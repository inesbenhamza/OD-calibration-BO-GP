import copy
import math
import torch
import gpytorch
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, Dict, List

from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal

from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError

from linear_operator.utils.errors import NotPSDError
from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator


def _ok_edge(eid: Optional[str], ignore_internal: bool) -> bool: #ignore internal edges (e.g. those starting with ":")
    if eid is None:
        return False
    if ignore_internal and eid.startswith(":"):
        return False
    return True





#####################################
#######################################
#######################################
#######################################

def adjacency_by_shared_node_weighted(  # use <edge id="..." from="..." to="..."/>.
# so based on if there is a connection between two edges meaning they share a junction node.
 # ex : <edge id="L15" from="6" to="3"
    #    <edge id="L16" from="3" to="6" 
    #    <edge id="L18" from="3" to="4"
    # so they share a junction node which is the node 3 and so they are neighbors and we can weight them accordingly
    net_xml_path: str,
    edge_ids: List[str],
    ignore_internal: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    normalize: str = "k-1",  # "k-1" or "k" or "none"
    clamp_max: Optional[float] = 1.0,  # set None to disable The “theoretical” maximum of 2 is reduced to 1 so that a bidirectional pair doesn’t get double weight compared to a pair that shares only one junction.
    # as it wouldnt make sense, it would give more weight to a bidirectional pair than a pair that shares only one junction.
) -> torch.Tensor:
    """
    Goal: Build an L×L adjacency matrix between edges (links). 
    Two edges are “neighbors” and get a non-zero entry in A if they share a junction node (same from or same to in the SUMO net).
    Undirected adjacency: two edges are neighbors if they share a 'from' or 'to' node.
    Weighted: for each node with k incident edges, connect all pairs with weight 1/(k-1) (or 1/k).

Step 1 — build node_to_edges

The loop goes through every edge and registers it at its from and to nodes:


    L15 (from=3, to=6) → node_to_edges[3].add(L15),  node_to_edges[6].add(L15)
    L16 (from=6, to=3) → node_to_edges[6].add(L16),  node_to_edges[3].add(L16)

    
    node_to_edges[3] = {L15, L16}   ← both edges meet at node 3
    node_to_edges[6] = {L15, L16}   ← both edges meet at node 6


    Step 2 — compute weights

For each node, connect all pairs with weight 1/(k-1) where k = number of edges at that node:


Node 3: k=2, w = 1/(2-1) = 1.0  → A[L15, L16] += 1.0
Node 6: k=2, w = 1/(2-1) = 1.0  → A[L15, L16] += 1.0

Total: A[L15, L16] = 2.0
L15 and L16 share both nodes so they accumulate the weight twice.



The clamp brings it back to 1.0:



A = torch.clamp(A, max=1.0)  # 2.0 → 1.0
Regular non-bidirectional pairs only share one node → weight stays at most 1.0 naturally → clamp has no effect on them.

"""
    tree = ET.parse(net_xml_path) # read the net.xml file and parse it into a tree
    root = tree.getroot()  # get the root element of the tree

    edge_to_idx = {e: i for i, e in enumerate(edge_ids)} # map each edge id to a unique integer index
    L = len(edge_ids) # number of links (edges)
    A = torch.zeros((L, L), dtype=dtype, device=device) # matrix of zeros with shape L×L

    node_to_edges: Dict[str, set] = {}

    for e in root.findall(".//edge"):
        eid = e.get("id")
        if not _ok_edge(eid, ignore_internal):
            continue
        if eid not in edge_to_idx:
            continue

        u = e.get("from") # get the start 
        v = e.get("to") # get the ending edge of the edge
        idx = edge_to_idx[eid] # get the index of the edge

        for n in (u, v):
            if n is None:
                continue
            node_to_edges.setdefault(n, set()).add(idx) # inserting a key-value pair into the dictionary, the key is the node and the value is a set of edges index. such as node_to_edges = { "3": {15, 16,18}, "6": {15, 16}, "4": {18} }

    for edges_set in node_to_edges.values():
        edges = list(edges_set)
        k = len(edges)
        if k <= 1:
            continue

        if normalize == "k-1":
            w = 1.0 / (k - 1)
        elif normalize == "k":
            w = 1.0 / k
        else:
            w = 1.0

        for i in edges:
            for j in edges:
                if i != j:
                    A[i, j] += w

    if clamp_max is not None:
        A = torch.clamp(A, max=float(clamp_max))

    A.fill_diagonal_(0.0)
    return A



###########################

def adjacency_route_overlap_weighted(
    routes_csv_path: str,
    edge_ids: List[str],
    route_edges_col: str = "route_edges",
    ratio_col: str = "ratio",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build L×L adjacency from route co-occurrence weighted by split ratios.

    W[i, r] = split ratio of route r  if edge i is on route r, else 0.
    A = W @ Wᵀ  (then zero diagonal).

    A[i,j] = Σ_r  ratio_r²  for all routes r containing both edge i and edge j.
    This gives soft co-occurrence: edges that share high-flow routes get higher weight.

    If the routes CSV has no ratio column (single-route file), binary W is used (ratio=1).
    """
    import csv as _csv
    L = len(edge_ids)
    edge_to_idx = {e: i for i, e in enumerate(edge_ids)}

    rows = []
    with open(routes_csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n_routes = len(rows)
    W = torch.zeros(L, n_routes, dtype=dtype, device=device)

    for r_idx, row in enumerate(rows):
        ratio = float(row[ratio_col]) if ratio_col in row and row[ratio_col] else 1.0
        for eid in str(row[route_edges_col]).split():
            if eid in edge_to_idx:
                W[edge_to_idx[eid], r_idx] = ratio

    A = W @ W.T


    if n_routes > 0:
        A = A / n_routes   # normalize by the number of routes, so that the scale of A is not too large when there are many routes. This way, the entries of A represent the average co-occurrence weighted by split ratios across all routes.
    A.fill_diagonal_(0.0)
    return A


########################################################
########################################################




def adjacency_route_overlap_weighted(
    routes_csv_path: str,
    edge_ids: List[str],
    route_edges_col: str = "route_edges",
    ratio_col: str = "ratio",
    from_taz_col: str = "fromTaz",
    to_taz_col: str = "toTaz",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Build the L×L task kernel matrix A for a multi-output GP.

    Supervisor definition:
        W  (L × R)  binary          W[i,r]  = 1 if edge i ∈ route r, else 0
        W2 (R × D)  probabilities   W2[r,d] = P(route r | OD d)  (split ratio)
        M  (L × D)  = W @ W2        M[i,d]  = P(edge i used | OD d)
        A  (L × L)  = (M @ Mᵀ) / D  A[i,j]  = (1/D) Σ_d P(i|d)·P(j|d), zero diagonal

    Dividing by D (number of OD pairs) keeps A in [0, 1] and makes the scale
    invariant to network size, preventing confounding with the GP output variance.

    If the CSV has no ratio column (single-route-per-OD), each route gets
    probability 1.0 for its OD pair.

    Raises ValueError if the OD columns are absent — W2 cannot be built.
    """
    L = len(edge_ids)
    edge_to_idx = {e: i for i, e in enumerate(edge_ids)}

    df = pd.read_csv(routes_csv_path).reset_index(drop=True)
    R = len(df)

    # ── Validate: OD columns are required to construct W2 ─────────────────
    missing = [c for c in (from_taz_col, to_taz_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"W2 cannot be built: column(s) {missing} not found in "
            f"'{routes_csv_path}'.\nAvailable: {list(df.columns)}"
        )

    # ── OD index: unique (fromTaz, toTaz) pairs, preserving row order ─────
    od_pairs = list(dict.fromkeys(zip(df[from_taz_col], df[to_taz_col])))
    od_to_idx = {od: d for d, od in enumerate(od_pairs)}
    D = len(od_pairs)

    # ── Split ratios: use column if present, else 1.0 per route ───────────
    has_ratio = ratio_col in df.columns
    ratios = df[ratio_col].astype(float).tolist() if has_ratio else [1.0] * R

    # ── W : binary edge-route incidence  (L × R) ──────────────────────────
    W = torch.zeros(L, R, dtype=dtype, device=device)
    for r, edges_str in enumerate(df[route_edges_col].astype(str)):
        for eid in edges_str.split():
            if eid in edge_to_idx:
                W[edge_to_idx[eid], r] = 1.0

    # ── W2 : route-OD probability matrix  (R × D) ─────────────────────────
    W2 = torch.zeros(R, D, dtype=dtype, device=device)
    d_indices = torch.tensor(
        [od_to_idx[(row[from_taz_col], row[to_taz_col])] for _, row in df.iterrows()],
        dtype=torch.long, device=device,
    )
    W2[torch.arange(R, device=device), d_indices] = torch.tensor(
        ratios, dtype=dtype, device=device
    )

    # ── M = W @ W2,  A = (M @ Mᵀ) / D,  zero diagonal ────────────────────
    M = W @ W2              # (L × D)  P(edge i used | OD d)
    A = (M @ M.T) / D      # (L × L)  average co-usage probability across ODs
    A.fill_diagonal_(0.0)

    if verbose:
        print(f"  W  : {tuple(W.shape)}   binary edge-route incidence")
        print(f"  W2 : {tuple(W2.shape)}   "
              f"{'split ratios from ratio col' if has_ratio else 'uniform 1.0 (no ratio col)'}")
        print(f"  M  : {tuple(M.shape)}   P(edge | OD)")
        print(f"  A  : {tuple(A.shape)}   task kernel / D={D}  (zero diagonal)")
        print(f"  A non-zero entries : {int((A > 0).sum())}")
        print(f"  A max : {A.max().item():.4f}")
        print(f"  A[0:3, 0:3] :\n{A[:3, :3]}")

    return A


##############################
##############################



#  building the Laplacian matrix L from the adjacency matrix A
def laplacian_from_adjacency(
    A: torch.Tensor,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build Laplacian from adjacency.
    - If normalized: L_sym = I - D^{-1/2} A D^{-1/2}
    - Else: L = D - A
    """
    A = 0.5 * (A + A.T)
    A = A.to(dtype=torch.float64)

    deg = A.sum(dim=1)

    if normalized:
        d = deg.pow(-0.5)  # D^{-1/2}, shape [L, 1], entries are 1/sqrt(degree)
        A_norm = (A * d[:, None]) * d[None, :] # D^{-1/2} A D^{-1/2}, we are broadcasting the d vector to the same size as A, and then multiplying the result by the transpose of d.
        # important * is element-wise multiplication, not matrix multiplication.
        # d is a column vector, so d[:, None] makes it a column vector of the same size as A ( [L, 1] ), and d[None, :] makes it a row vector of the same size as A ( [1, L] ).
        L = torch.eye(A.shape[0], dtype=A.dtype, device=A.device) - A_norm
    else:
        L = -A.clone()
        L.diagonal().copy_(deg)

    L = 0.5 * (L + L.T)
    return L



####
########################################################
########################################################    
def laplacian_eigh(
    A: torch.Tensor,
    normalized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition of Laplacian from adjacency.
    Returns (evals, evecs) with evals clamped to >= 0.

    torch.linalg.eigh is ascending and returns all eigenpairs.
    """
    L_mat = laplacian_from_adjacency(A, normalized=normalized)
    evals, evecs = torch.linalg.eigh(L_mat)
    evals = torch.clamp(evals, min=0.0)
    return evals, evecs




####
########################################################
########################################################   


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse softplus for y > 0: log(expm1(y))."""
    return torch.log(torch.expm1(y).clamp_min(1e-12))



class LearnableTauGraphTaskKernel(Kernel): # for diffusion kernel
    """
    Task kernel K = exp(-tau * L) with learnable tau; stores Laplacian eigenpairs as buffers.
    """
    is_stationary = False
    TAU_LB: float = 0.05
    TAU_UB: float = 10.0

    def __init__(
        self,
        Lap_evals: torch.Tensor,
        Lap_evecs: torch.Tensor,
        tau_init: float,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)
        self.register_buffer("Lap_evals", Lap_evals)  # λ_i — fixed from road network
        self.register_buffer("Lap_evecs", Lap_evecs)  # V   — fixed from road network

        tau0 = float(min(max(tau_init, self.TAU_LB), self.TAU_UB))
        t01 = max(1e-6, min((tau0 - self.TAU_LB) / (self.TAU_UB - self.TAU_LB), 1 - 1e-6))
        raw_tau0 = torch.tensor(math.log(t01 / (1.0 - t01)), dtype=dtype, device=device)
        self.register_parameter("raw_tau", torch.nn.Parameter(raw_tau0))

    @property
    def tau(self) -> torch.Tensor:
        return self.TAU_LB + (self.TAU_UB - self.TAU_LB) * torch.sigmoid(self.raw_tau)

    def _full_matrix(self) -> torch.Tensor:
        exp_evals = torch.exp(-self.tau * self.Lap_evals)
        Kfull = (self.Lap_evecs * exp_evals) @ self.Lap_evecs.T
        return 0.5 * (Kfull + Kfull.T)

    @property
    def covar_matrix(self):
        # Required by MultitaskKernel.forward() — it accesses task_covar_module.covar_matrix
        return DenseLinearOperator(self._full_matrix())

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)
        i2 = i2.long().squeeze(-1)
        Kfull = self._full_matrix()
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]
        return DenseLinearOperator(K)




class LearnableGammaGraphTaskKernel(Kernel):
    """
    Task kernel K = (I + gamma * L)^{-1} with learnable gamma; stores Laplacian eigenpairs as buffers.
    """
    is_stationary = False

    def __init__(
        self,
        Lap_evals: torch.Tensor,
        Lap_evecs: torch.Tensor,
        gamma_init: float,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)
        self.register_buffer("Lap_evals", Lap_evals)
        self.register_buffer("Lap_evecs", Lap_evecs)
        g0 = torch.tensor(max(float(gamma_init), 1e-12), dtype=dtype, device=device)
        raw_g0 = _softplus_inv(g0)
        self.register_parameter("raw_gamma", torch.nn.Parameter(raw_g0))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gamma)

    def _full_matrix(self) -> torch.Tensor:
        inv_evals = 1.0 / (1.0 + self.gamma * self.Lap_evals)
        Kfull = (self.Lap_evecs * inv_evals) @ self.Lap_evecs.T
        return 0.5 * (Kfull + Kfull.T)

    @property
    def covar_matrix(self):
        # Required by MultitaskKernel.forward() — it accesses task_covar_module.covar_matrix
        return DenseLinearOperator(self._full_matrix())

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)
        i2 = i2.long().squeeze(-1)
        Kfull = self._full_matrix()
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]
        return DenseLinearOperator(K)




##################################################
########################################################  


class GraphMaternTaskKernel(Kernel):
    """
    Spectral Matérn task kernel on a graph:
        K = V · diag((2ν/κ² + λ)^{-ν}) · Vᵀ
    with learnable smoothness ν and length-scale κ.
    Stores Laplacian eigenpairs as buffers (fixed from road network).
    """
    is_stationary = False

    def __init__(
        self,
        Lap_evals: torch.Tensor,
        Lap_evecs: torch.Tensor,
        nu_init: float = 1.5,
        kappa_init: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)
        self.register_buffer("Lap_evals", Lap_evals)
        self.register_buffer("Lap_evecs", Lap_evecs)

        nu0 = torch.tensor(max(float(nu_init), 1e-6), dtype=dtype, device=device)
        self.register_parameter("raw_nu", torch.nn.Parameter(_softplus_inv(nu0)))

        k0 = torch.tensor(max(float(kappa_init), 1e-6), dtype=dtype, device=device)
        self.register_parameter("raw_kappa", torch.nn.Parameter(_softplus_inv(k0)))

    @property
    def nu(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_nu)

    @property
    def kappa(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_kappa)

    def _full_matrix(self) -> torch.Tensor:
        Psi = (2.0 * self.nu / self.kappa ** 2 + self.Lap_evals).pow(-self.nu)
        Kfull = (self.Lap_evecs * Psi) @ self.Lap_evecs.T
        return 0.5 * (Kfull + Kfull.T)

    @property
    def covar_matrix(self):
        return DenseLinearOperator(self._full_matrix())

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)
        i2 = i2.long().squeeze(-1)
        Kfull = self._full_matrix()
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]
        return DenseLinearOperator(K)

########################################################
########################################################    



def build_graph_multitask_gp(
    net_xml_path: Optional[str] = None,   # required for connections/shared_node_weighted
    edge_ids: List[str] = None,
    train_X: torch.Tensor = None,         # (N, d+1), last column task_id
    train_Y: torch.Tensor = None,         # (N, 1)
    input_dim: int = None,                # d
    adjacency_type: str = "shared_node_weighted",
    routes_csv_path: Optional[str] = None,  # required for route_overlap_weighted
    graph_kernel_type: str = "diffusion",  # "diffusion" or "regularized_laplacian"
    ignore_internal: bool = True,
    matern_nu: float = 2.5,
    device: Optional[torch.device] = None,
    normalized_laplacian: bool = True,
    standardize: bool = True,  # set False when data is pre-standardized per task
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict]:
    """
    Returns (model, mll, info_dict). Model and mll are moved to device.
    graph_kernel_type: "diffusion" -> K = exp(-tau*L); "regularized_laplacian" -> K = (I+gamma*L)^{-1}.
    normalized_laplacian: use L_sym = I - D^{-1/2} A D^{-1/2} (True) or L = D - A (False).
    Uses SingleTaskGP so the product kernel x_kernel*task_kernel is used as-is;
    MultiTaskGP would add an extra IndexKernel on top.
    """
    if adjacency_type in ("connections", "shared_node_weighted") and net_xml_path is None:
        raise ValueError(f"net_xml_path is required for adjacency_type='{adjacency_type}'")
    if adjacency_type == "route_overlap_weighted" and routes_csv_path is None:
        raise ValueError("routes_csv_path is required for adjacency_type='route_overlap_weighted'")

    if device is None:
        device = train_X.device

    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    if train_Y.ndim == 1:
        train_Y = train_Y.view(-1, 1)

    #adjacency
    if adjacency_type == "shared_node_weighted":
        A = adjacency_by_shared_node_weighted(
            net_xml_path,
            edge_ids=edge_ids,
            ignore_internal=ignore_internal,
            device=device,
            dtype=torch.float64,
            normalize="k-1",
            clamp_max=1.0,
        )
    elif adjacency_type == "route_overlap_weighted":
        A = adjacency_route_overlap_weighted(
            routes_csv_path,
            edge_ids=edge_ids,
            device=device,
            dtype=torch.float64,
        )
    else:
        raise ValueError(f"Unknown adjacency_type: {adjacency_type}")

    A_density = float((A > 0).float().mean().item()) # about richiness of graph structur e, A_density = 1.0 → fully connected, every edge correlated with every other
    isolated = int((A.sum(dim=1) == 0).sum().item()) # to check if we have some edges that are isolated so not connected to any other edeg

    if graph_kernel_type == "diffusion":
        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
        task_kernel = LearnableTauGraphTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            tau_init=1.0, dtype=torch.float64, device=device,
        )
    elif graph_kernel_type == "regularized_laplacian":
        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
        task_kernel = LearnableGammaGraphTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            gamma_init=1.0, dtype=torch.float64, device=device,
        )
    elif graph_kernel_type == "matern":
        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
        task_kernel = GraphMaternTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            nu_init=1.5, kappa_init=1.0, dtype=torch.float64, device=device,
        )
    else:
        raise ValueError(f"Unknown graph_kernel_type: {graph_kernel_type}. Use 'diffusion', 'regularized_laplacian', or 'matern'.")
    

    x_kernel = ScaleKernel(MaternKernel(nu=matern_nu, lengthscale_constraint=Interval(0.005, 20.0)))
    x_kernel = x_kernel.to(device)
    task_kernel = task_kernel.to(device)
    x_kernel.active_dims = torch.tensor(
        list(range(input_dim)), dtype=torch.long, device=device
    )
    task_kernel.active_dims = torch.tensor(
        [input_dim], dtype=torch.long, device=device
    )
    covar = x_kernel * task_kernel

    task_mean = None
    task_std = None

    # SingleTaskGP is correct here: our product kernel already encodes both
    # input covariance (x_kernel) and task covariance (task_kernel via
    # active_dims). MultiTaskGP would add an extra IndexKernel on top, causing
    # the task column to be handled twice and distorting the graph kernel.

    # If standardize=True, do it PER TASK (edge) rather than a single global
    # Standardize(m=1) over all stacked (x, task) rows.
    #
    # We standardize train_Y directly and disable GP's own outcome_transform.
    if standardize:
        L_tasks = len(edge_ids)
        # train_X layout: (design features..., task_id) where task_id sits at index=input_dim
        task_ids = train_X[:, input_dim].long()
        y = train_Y.squeeze(-1)  # (N,)

        task_mean = torch.zeros(L_tasks, dtype=y.dtype, device=y.device)
        task_std = torch.ones(L_tasks, dtype=y.dtype, device=y.device)

        for t in range(L_tasks):
            mask = (task_ids == t) & (~torch.isnan(y))
            if mask.any():
                task_mean[t] = y[mask].mean()
                task_std[t] = y[mask].std(unbiased=False).clamp_min(1e-6)
            else:
                # If a task never appears (shouldn't happen), keep it identity.
                task_mean[t] = 0.0
                task_std[t] = 1.0

        y_std = (y - task_mean[task_ids]) / task_std[task_ids]
        train_Y = y_std.unsqueeze(-1)  # (N, 1)

    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 0.1))
    likelihood = likelihood.to(device)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar,
        likelihood=likelihood,
        outcome_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Attach per-task standardization params so GraphKernelAggregationModel can un-standardize.
    if standardize:
        model.register_buffer("task_mean", task_mean.view(-1))
        model.register_buffer("task_std", task_std.view(-1))

    model = model.to(device)
    mll = mll.to(device)

    info = {
        "A_density": A_density,
        "isolated_tasks": isolated,
        "L_tasks": len(edge_ids),
        "adjacency_type": adjacency_type,
        "graph_kernel_type": graph_kernel_type,
        "normalized_laplacian": normalized_laplacian,
    }
    return model, mll, info











def _uniform(shape, lb: float, ub: float, device, dtype) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lb, ub)

def randomize_graph_hyperparams(model: SingleTaskGP) -> None:
    device = model.train_inputs[0].device
    dtype  = model.train_inputs[0].dtype

    with torch.no_grad():
        x_kern    = model.covar_module.kernels[0]
        task_kern = model.covar_module.kernels[1]

        # outputscale
        ls_lb, ls_ub = 0.05, 2.0
        if hasattr(x_kern, "raw_outputscale_constraint"):
            c = x_kern.raw_outputscale_constraint
            if math.isfinite(float(c.lower_bound)): ls_lb = float(c.lower_bound)
            if math.isfinite(float(c.upper_bound)): ls_ub = float(c.upper_bound)
        x_kern.outputscale = _uniform(x_kern.outputscale.shape, ls_lb, ls_ub, device, dtype)

        # lengthscale
        l_lb, l_ub = 0.005, 20.0
        if hasattr(x_kern.base_kernel, "raw_lengthscale_constraint"):
            c = x_kern.base_kernel.raw_lengthscale_constraint
            if math.isfinite(float(c.lower_bound)): l_lb = float(c.lower_bound)
            if math.isfinite(float(c.upper_bound)): l_ub = float(c.upper_bound)
        x_kern.base_kernel.lengthscale = _uniform(
            x_kern.base_kernel.lengthscale.shape, l_lb, l_ub, device, dtype
        )

        # graph kernel scalar
        if hasattr(task_kern, "raw_gamma"):
            g_new = _uniform((), 0.1, 20.0, device, dtype)
            task_kern.raw_gamma.copy_(_softplus_inv(g_new))
        if hasattr(task_kern, "raw_tau"):
            t_new = _uniform((), LearnableTauGraphTaskKernel.TAU_LB, LearnableTauGraphTaskKernel.TAU_UB * 0.5, device, dtype)
            t01 = ((t_new - LearnableTauGraphTaskKernel.TAU_LB) / (LearnableTauGraphTaskKernel.TAU_UB - LearnableTauGraphTaskKernel.TAU_LB)).clamp(1e-6, 1 - 1e-6)
            task_kern.raw_tau.copy_(torch.log(t01 / (1.0 - t01)))
        if hasattr(task_kern, "raw_nu"):
            nu_new = _uniform((), 0.5, 3.0, device, dtype)
            task_kern.raw_nu.copy_(_softplus_inv(nu_new))
        if hasattr(task_kern, "raw_kappa"):
            k_new = _uniform((), 0.1, 5.0, device, dtype)
            task_kern.raw_kappa.copy_(_softplus_inv(k_new))

        # noise
        n_lb, n_ub = 1e-6, 0.1
        if hasattr(model.likelihood.noise_covar, "raw_noise_constraint"):
            c = model.likelihood.noise_covar.raw_noise_constraint
            if math.isfinite(float(c.lower_bound)): n_lb = float(c.lower_bound)
            if math.isfinite(float(c.upper_bound)): n_ub = float(c.upper_bound)
        model.likelihood.noise = _uniform(
            model.likelihood.noise.shape, n_lb, n_ub, device, dtype
        )





def fit_graph_mll_multistart(
    mll: ExactMarginalLogLikelihood,
    n_restarts: int = 5,
    verbose: bool = False,
) -> str:
    """
    Multi-start MLL fitting for the graph SingleTaskGP.
    Restart 0: smart initialization (lengthscale=0.5, noise=0.01, outputscale=1).
    Restarts 1+: randomized hyperparameters.
    Keeps the state with highest MLL.
    Falls back to Adam only if every LBFGS restart fails.
    Returns 'lbfgs-multistart' or 'adam-fallback'.
    """
    model          = mll.model
    original_state = copy.deepcopy(model.state_dict())
    best_state     = None
    best_mll_val   = -math.inf

    for r in range(n_restarts):
        model.load_state_dict(original_state)


        randomize_graph_hyperparams(model)

        
        model.train()
        mll.train()
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"  [graph restart {r+1}/{n_restarts}] LBFGS failed ({str(e)[:50]}), skipping.")
            continue
        model.train()
        mll.train()
        with torch.no_grad():
            try:
                out     = model(*model.train_inputs)
                mll_val = mll(out, model.train_targets).item()
            except Exception:
                mll_val = -math.inf
        # reject degenerate solutions before comparing MLL
        noise_val = model.likelihood.noise.item()
        os_val    = model.covar_module.kernels[0].outputscale.item() \
                    if hasattr(model.covar_module.kernels[0], "outputscale") else 1.0
        if noise_val > 0.08 or os_val < 1e-4:
            if verbose:
                print(f"  [graph restart {r+1}/{n_restarts}] degenerate (noise={noise_val:.3f}, os={os_val:.2e}), skipping.")
            continue
        if verbose:
            print(f"  [graph restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")
        if mll_val > best_mll_val:
            best_mll_val = mll_val
            best_state   = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        mll.eval()
        if verbose:
            print(f"  → best MLL={best_mll_val:.4f}")
        return "lbfgs-multistart"

    # all restarts failed — Adam fallback on original init
    model.load_state_dict(original_state)
    model.train()
    mll.train()
    opt      = torch.optim.Adam(mll.parameters(), lr=1e-2)
    stored_X = model.train_inputs[0]
    for _ in range(200):
        opt.zero_grad()
        try:
            loss = -mll(model(stored_X), model.train_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=1.0)
            opt.step()
        except NotPSDError:
            continue
    model.eval()
    mll.eval()
    return "adam-fallback"


#######################################
#######################################
#######################################
#######################################

def run_graph_multitask_gp(
    net_xml_path: Optional[str] = None,   # required for connections/shared_node_weighted
    edge_ids: List[str] = None,
    train_X: torch.Tensor = None,
    train_Y: torch.Tensor = None,
    input_dim: int = None,
    adjacency_type: str = "shared_node_weighted",
    routes_csv_path: Optional[str] = None,  # required for route_overlap_weighted
    graph_kernel_type: str = "diffusion",
    ignore_internal: bool = True,
    matern_nu: float = 2.5,
    device: Optional[torch.device] = None,
    normalized_laplacian: bool = True,
    standardize: bool = True,
    n_gp_restarts: int = 5,
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict, str]:
    """
    Build graph GP and fit. Returns (model, mll, info, fit_method).
    """
    if device is None:
        device = train_X.device

    model, mll, info = build_graph_multitask_gp(
        net_xml_path=net_xml_path,
        edge_ids=edge_ids,
        train_X=train_X,
        train_Y=train_Y,
        input_dim=input_dim,
        adjacency_type=adjacency_type,
        routes_csv_path=routes_csv_path,
        graph_kernel_type=graph_kernel_type,
        ignore_internal=ignore_internal,
        matern_nu=matern_nu,
        device=device,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
    )

    fit_method = fit_graph_mll_multistart(mll, n_restarts=n_gp_restarts)
    return model, mll, info, fit_method



#######################################
#######################################
#######################################
#######################################

# Graph aggregation model for BO

class GraphKernelAggregationModel(Model):
    """
    Wrap a fitted graph SingleTaskGP (train rows (x, task_idx), one scalar y per row) and expose
    scalar S(x) = w^T e(x) for EI. X is (b, d); for each x we stack [(x,0), (x,1), ..., (x,L-1)],
    get posterior over e(x), then aggregate.

    If task_mean/task_std are provided (per-task standardization was applied before GP fitting),
    the posterior is un-standardized per task before aggregation so that S(x) is in the original space.
    """
    def __init__(
        self,
        graph_model,
        weights: Tensor,
        task_mean: Optional[Tensor] = None,
        task_std: Optional[Tensor] = None,
    ):
        super().__init__()
        self.graph_model = graph_model
        self.register_buffer("w", weights.view(-1))  # (L,)
        # Per-task un-standardization buffers.
        # If the caller didn't provide them, try to read them from the fitted graph_model.
        if (task_mean is None or task_std is None) and hasattr(graph_model, "task_mean") and hasattr(graph_model, "task_std"):
            task_mean = getattr(graph_model, "task_mean")
            task_std = getattr(graph_model, "task_std")

        if task_mean is not None and task_std is not None:
            self.register_buffer("task_mean", task_mean.view(-1))   # (L,)
            self.register_buffer("task_std",  task_std.view(-1))    # (L,)
        else:
            self.task_mean = None
            self.task_std  = None

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        if X.dim() == 3:
            b, q, d = X.shape
            if q != 1:
                raise RuntimeError("q=1 only.")
            X = X.view(b, d)
        elif X.dim() == 2:
            b, d = X.shape
        else:
            raise RuntimeError(f"Unexpected X.dim()={X.dim()}")

        L = self.w.shape[0]
        self.graph_model.eval()
        if hasattr(self.graph_model, "likelihood"):
            self.graph_model.likelihood.eval()

        # Vectorized: build (b*L, d+1) in one shot, call posterior once.
        # Layout: first L rows = candidate 0 × all tasks, next L = candidate 1, ...
        task_col = torch.arange(L, dtype=X.dtype, device=X.device).view(1, L, 1).expand(b, L, 1)
        X_exp = X.unsqueeze(1).expand(b, L, d)           # (b, L, d)
        X_stack = torch.cat([X_exp, task_col], dim=-1)   # (b, L, d+1)
        X_flat = X_stack.reshape(b * L, d + 1)            # (b*L, d+1)

        post = self.graph_model.posterior(X_flat, observation_noise=observation_noise)
        mean_flat = post.mean.view(b * L)                 # (b*L,)
        cov_flat = post.covariance_matrix                 # (b*L, b*L)
        if hasattr(cov_flat, "evaluate"):
            cov_flat = cov_flat.evaluate()

        # Aggregate: mu_S[i] = w^T mean[i*L:(i+1)*L]
        mean_tasks = mean_flat.view(b, L)                 # (b, L)

        # var_S[i] = w^T Cov(Y(x_i,·), Y(x_i,·)) w — only the L×L block per x_i.
        # (Cross-candidate covariances are generally nonzero, but marginal Var(S(x_i))
        # depends only on that block; EI uses per-candidate mean/var anyway.)
        cov_blocks = cov_flat.view(b, L, b, L)           # (b, L, b, L)
        cov_diag = cov_blocks[torch.arange(b), :, torch.arange(b), :]  # (b, L, L)

        # Un-standardize per task if the GP was fitted on manually standardized targets
        if self.task_mean is not None:
            mean_tasks = mean_tasks * self.task_std.unsqueeze(0) + self.task_mean.unsqueeze(0)
            sigma_outer = self.task_std[:, None] * self.task_std[None, :]  # (L, L)
            cov_diag = cov_diag * sigma_outer.unsqueeze(0)                 # (b, L, L)

        mus = (mean_tasks * self.w.unsqueeze(0)).sum(-1)  # (b,)
        vars_ = torch.einsum("i,bij,j->b", self.w, cov_diag, self.w)   # (b,)
        vars_ = torch.clamp(vars_, min=0.0)

        mus = mus.view(b, 1)
        vars_ = vars_.view(b, 1, 1)
        mvn = MultivariateNormal(mus, vars_)
        return GPyTorchPosterior(mvn)



































########################################################
########################################################
# Kronecker Graph-MOGP
########################################################
########################################################


class GraphKroneckerGP(ExactGP):
    """
    Multi-output GP with graph Laplacian task kernel using Kronecker structure.

        K((θ, t), (θ', t')) = K_data(θ, θ') ⊗ K_graph[t, t']

    where K_graph = exp(-τL) or (I + γL)^{-1}.

    Uses MultitaskKernel (same as ICM) but replaces its internal IndexKernel
    with the graph task kernel so GPyTorch's Kronecker prediction strategies
    work correctly → O(N^3 + L^3) scaling.

    Parameters
    ----------
    train_X : torch.Tensor [N, d]  — normalized OD inputs
    train_Y : torch.Tensor [N, L]  — per-task targets (standardized outside)
    likelihood : MultitaskGaussianLikelihood
    graph_task_kernel : LearnableTauGraphTaskKernel or LearnableGammaGraphTaskKernel
    """

    def __init__(self, train_X, train_Y, likelihood, graph_task_kernel):
        super().__init__(train_X, train_Y, likelihood)
        self.num_tasks = train_Y.shape[-1]

        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_tasks)

        data_kernel = ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_constraint=Interval(0.005, 20.0),
            )
        )

        # MultitaskKernel handles Kronecker prediction strategies correctly.
        # We initialize it with rank=1 (creates a dummy IndexKernel) then
        # replace task_covar_module with the graph task kernel.
        # MultitaskKernel.forward() calls task_covar_module(task_idx, task_idx)
        # which our graph kernels accept directly.
        self.covar_module = MultitaskKernel(
            data_kernel,
            num_tasks=self.num_tasks,
            rank=1,
        )
        self.covar_module.task_covar_module = graph_task_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X=None, **kwargs):
        return X


def randomize_graph_kronecker_hyperparams(model: GraphKroneckerGP) -> None:
    """Randomize GraphKroneckerGP hyperparameters for multistart restarts."""
    with torch.no_grad():
        device = model.train_inputs[0].device
        dtype  = model.train_inputs[0].dtype

        # data_covar_module is now ScaleKernel(MaternKernel)
        data_kernel = model.covar_module.data_covar_module

        # Outputscale
        os_lb, os_ub = 0.05, 2.0
        if hasattr(data_kernel, "raw_outputscale_constraint"):
            c = data_kernel.raw_outputscale_constraint
            if math.isfinite(float(c.lower_bound)): os_lb = float(c.lower_bound)
            if math.isfinite(float(c.upper_bound)): os_ub = float(c.upper_bound)
        data_kernel.outputscale = _uniform(
            data_kernel.outputscale.shape, os_lb, os_ub, device, dtype
        )

        # Lengthscale (inside base_kernel)
        c = data_kernel.base_kernel.raw_lengthscale_constraint
        lb, ub = float(c.lower_bound), float(c.upper_bound)
        data_kernel.base_kernel.lengthscale = _uniform(
            data_kernel.base_kernel.lengthscale.shape, lb, ub, device, dtype
        )

        # Per-task noises
        if hasattr(model.likelihood, "task_noises"):
            c = getattr(model.likelihood, "raw_task_noises_constraint", None)
            lb = float(c.lower_bound) if c is not None else 1e-6
            ub = float(c.upper_bound) if c is not None else 1.0
            model.likelihood.task_noises = _uniform(
                model.likelihood.task_noises.shape, lb, ub, device, dtype
            )

        # Graph kernel scalar (tau or gamma)
        task_kern = model.covar_module.task_covar_module
        if hasattr(task_kern, "raw_tau"):
            t_new = _uniform((), LearnableTauGraphTaskKernel.TAU_LB, LearnableTauGraphTaskKernel.TAU_UB * 0.5, device, dtype)
            t01 = ((t_new - LearnableTauGraphTaskKernel.TAU_LB) / (LearnableTauGraphTaskKernel.TAU_UB - LearnableTauGraphTaskKernel.TAU_LB)).clamp(1e-6, 1 - 1e-6)
            task_kern.raw_tau.copy_(torch.log(t01 / (1.0 - t01)))
        if hasattr(task_kern, "raw_gamma"):
            g_new = _uniform((), 0.1, 20.0, device, dtype)
            task_kern.raw_gamma.copy_(_softplus_inv(g_new))


def fit_graph_kronecker_multistart(
    mll: ExactMarginalLogLikelihood,
    n_restarts: int = 5,
    verbose: bool = True,
) -> None:
    """Multi-start L-BFGS MLL fitting for GraphKroneckerGP."""
    model = mll.model
    original_state = copy.deepcopy(model.state_dict())
    best_state = None
    best_mll = -math.inf

    for r in range(n_restarts):
        model.load_state_dict(original_state)
        if r > 0:
            randomize_graph_kronecker_hyperparams(model)

        model.train()
        mll.train()

        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"[restart {r+1}] L-BFGS failed ({str(e)[:60]}), skipping.")
            continue

        model.train()
        mll.train()

        with torch.no_grad():
            try:
                output = model(*model.train_inputs)
                mll_val = mll(output, model.train_targets).item()
            except Exception:
                mll_val = -math.inf

        if verbose:
            print(f"[restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")

        if mll_val > best_mll:
            best_mll = mll_val
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        model.load_state_dict(original_state)

    model.eval()
    mll.eval()

    if verbose:
        print(f"→ best MLL={best_mll:.4f}")


def initialize_graph_kronecker_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    edge_ids: List[str],
    adjacency_type: str = "shared_node_weighted",
    net_xml_path: Optional[str] = None,
    routes_csv_path: Optional[str] = None,
    graph_kernel_type: str = "diffusion",
    normalized_laplacian: bool = True,
    ignore_internal: bool = True,
    device: Optional[torch.device] = None,
    Lap_evals: Optional[torch.Tensor] = None,
    Lap_evecs: Optional[torch.Tensor] = None,
) -> Tuple:
    """
    Initialize a GraphKroneckerGP.

    Parameters
    ----------
    train_X : [N, d]  normalized OD inputs
    train_Y : [N, L]  per-task targets (standardized outside, same as ICM)
    Lap_evals / Lap_evecs : precomputed Laplacian eigenpairs (optional).
        Pass these to skip the adjacency build + eigendecomposition when the
        road network is fixed across BO iterations.

    Returns
    -------
    model : GraphKroneckerGP
    mll   : ExactMarginalLogLikelihood
    """
    if device is None:
        device = train_X.device

    train_X = train_X.to(device)
    train_Y = train_Y.to(device)

    if train_X.dim() != 2 or train_Y.dim() != 2:
        raise ValueError("Expected train_X [N, d] and train_Y [N, L].")
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must have the same number of rows.")

    L = train_Y.shape[-1]

    if Lap_evals is None or Lap_evecs is None:
        # Build adjacency and decompose — only needed when eigenpairs aren't cached.
        if adjacency_type == "shared_node_weighted":
            A = adjacency_by_shared_node_weighted(
                net_xml_path, edge_ids,
                ignore_internal=ignore_internal,
                device=device, dtype=torch.float64,
            )
        elif adjacency_type == "route_overlap_weighted":
            A = adjacency_route_overlap_weighted(
                routes_csv_path, edge_ids,
                device=device, dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown adjacency_type: {adjacency_type}")

        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
    else:
        Lap_evals = Lap_evals.to(device=device, dtype=torch.float64)
        Lap_evecs = Lap_evecs.to(device=device, dtype=torch.float64)

    if graph_kernel_type == "diffusion":
        graph_task_kernel = LearnableTauGraphTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            tau_init=1.0, dtype=torch.float64, device=device,
        )
    elif graph_kernel_type == "regularized_laplacian":
        graph_task_kernel = LearnableGammaGraphTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            gamma_init=1.0, dtype=torch.float64, device=device,
        )
    else:
        raise ValueError(f"Unknown graph_kernel_type: {graph_kernel_type}")

    likelihood = MultitaskGaussianLikelihood(
        num_tasks=L,
        has_global_noise=False,
        noise_constraint=Interval(1e-6, 0.1),
    ).to(device)

    model = GraphKroneckerGP(
        train_X=train_X,
        train_Y=train_Y,
        likelihood=likelihood,
        graph_task_kernel=graph_task_kernel,
    ).to(device)

    mll = ExactMarginalLogLikelihood(likelihood, model)
    return model, mll
