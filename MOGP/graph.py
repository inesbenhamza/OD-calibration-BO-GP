import torch
import gpytorch
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from typing import Optional, Tuple, Dict, List

from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from botorch.fit import fit_gpytorch_mll
from botorch.exceptions import ModelFittingError
from botorch.optim.closures import ForwardBackwardClosure
from botorch.optim.utils import get_parameters

from linear_operator.utils.errors import NotPSDError
from linear_operator.operators import DenseLinearOperator

"""
Graph from SUMO net.xml → adjacency A → diffusion kernel K = exp(-τ L).
where L is the Laplacian matrix of the graph (D-A) or (I-D^{-1/2} A D^{-1/2}) which is the normalized Laplacian matrix.
Task kernel uses K over the L edges

two different Adjacency options:
  - "connections": from <connection from="..." to="..."> (driving connectivity)
  - "shared_node_weighted": edges share a junction node (topological smoothness)

The file builds a multi-task Gaussian process where:

Inputs: OD (origin–destination) parameters x and a task index (edge id).

Output: One scalar per (x, edge) (e.g. error on that edge).

Task similarity: Defined by the road network: edges that are “close” in the graph are more correlated. (e.g. edges that share a junction node are more correlated)
That’s done by turning the SUMO network into a graph, then into a diffusion kernel/regularized Laplacian kernel used as the task kernel of the GP.
So: SUMO net → adjacency → Laplacian → diffusion kernel → task kernel, combined with
Matern on x in a SingleTaskGP (product kernel on (x, task).



2 things i didnt try but i think are interesting:


graph_kernel_type "random_walk": K = V diag((a - λ_i)^p) V^T (spectral polynomial kernel).
"""

class FixedGraphTaskKernel(Kernel):
    """Task kernel K_G over task indices (edge ids)
     GPyTorch kernel that defines covariance between tasks
    answers: “how correlated are edge i and edge j?” using a precomputed L×L matrix K_graph.
    K_graph : [L, L]  (diffusion kernel or regularized laplacian kernel)

    This kernel takes two sets of task indices i1 and i2, indexes into the fixed matrix K_graph to get the covariance between those tasks, and returns it as a GPyTorch linear operator.
    """
    is_stationary = False # this means that the kernel is not stationary, i.e. it depends on the input x.

    def __init__(self, K_graph: torch.Tensor):
        super().__init__()
        self.register_buffer("K_graph", K_graph) # store the positional encodings in state_dict()
        # not trainable, so it is a buffer. ( fixed kernel for a specific x, not learnable)
        # we can save and load it and be part of the model 

#for every pair (a, b) with a from i1 and b from i2, take K_graph[a, b]. So we get the L×L covariance
#it returns the task–task covariance for the requested task

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1) # .long() is used to convert the input to a long tensor.
        i2 = i2.long().squeeze(-1) # .squeeze(-1) is used to remove the last dimension of the tensor. so from (3,1) to (3,)
        K = self.K_graph[i1.unsqueeze(-1), i2.unsqueeze(-2)] # .unsqueeze(-1) is used to add a new dimension to the tensor.
        return DenseLinearOperator(K) # Wraps that matrix as a linear operator so the rest of GPyTorch can use it in the joint covariance (e.g. for the marginal log likelihood or predictions). as gpytorch does not support matrices directly, it needs to be wrapped in a linear operator.
        # in a model, there are both trainable param and buffer (not trainable),  that can be saved and loaded and be part of the model 

def _ok_edge(eid: Optional[str], ignore_internal: bool) -> bool: #ignore internal edges (e.g. those starting with ":")
    if eid is None:
        return False
    if ignore_internal and eid.startswith(":"):
        return False
    return True


############
###########################
###########################
########################################################

def adjacency_from_connections(  # use <connection from="..." to="..."/>. --> so flow is allowed 
    # ex :     <connection from="L9" to="L15"
    #    <connection from="L15" to="L18" 
    net_xml_path: str,
    edge_ids: List[str],
    ignore_internal: bool = True,
    undirected: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build L×L adjacency from SUMO net.xml <connection from="..." to="..."/>.

    A[i,j] = 1 if edge i connects to edge j (optionally symmetrized).
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    L = len(edge_ids)
    edge_to_idx = {e: i for i, e in enumerate(edge_ids)} # this is a dictionary that maps each edge id to a unique integer index
    A = torch.zeros((L, L), dtype=dtype, device=device) # matrix of zeros with shape L×L

    for conn in root.findall(".//connection"): # loop through all connections in the net.xml file
        f = conn.get("from")
        t = conn.get("to")

        if not (_ok_edge(f, ignore_internal) and _ok_edge(t, ignore_internal)):
            continue
        if f not in edge_to_idx or t not in edge_to_idx:
            continue
        if f == t:
            continue

        i, j = edge_to_idx[f], edge_to_idx[t]
        A[i, j] = 1.0
        if undirected:
            A[j, i] = 1.0

    A.fill_diagonal_(0.0)
    return A

#######################################
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
            node_to_edges.setdefault(n, set()).add(idx) # inserting a key-value pair into the dictionary, the key is the node and the value is a set of edges.

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




########################################################
########################################################    



# building the Laplacian matrix L from the adjacency matrix A
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

    deg = A.sum(dim=1).clamp_min(1e-12)  # prevent inf for isolated nodes

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



def _auto_scale_from_evals(evals: torch.Tensor) -> torch.Tensor:
    """Stable scale: median of positive eigenvalues.

    Used to get the initial value of the tau or gamma parameter.
    Diffusion: tau_used = tau_scale / scale → τ is inversely proportional to this scale.
    Regularized Laplacian: gamma_used = gamma_scale / scale → γ is inversely proportional to this scale.

    If there is no positive λ to scale from, we fall back to a neutral scale of 1.0 so
    tau_used ≈ tau_scale / 1 and similarly for γ.
    """
    pos = evals[evals > 1e-12]
    if pos.numel() > 0:
        return pos.median().clamp_min(1e-12)
    return torch.tensor(1.0, dtype=evals.dtype, device=evals.device)


















########################################################
########################################################    
########################################################
########################################################    
# Diffusion kernel
def diffusion_kernel_from_adjacency_eigh(
    A: torch.Tensor,
    tau: Optional[float] = None,
    tau_mode: str = "auto",
    tau_scale: float = 1.0,
    normalized_laplacian: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Compute K = exp(-tau * Laplacian) via eigendecomposition.
    Returns (K, tau_used). K is float64.

    """
    evals, evecs = laplacian_eigh(A, normalized=normalized_laplacian)

    if tau_mode == "auto":
        scale = _auto_scale_from_evals(evals)
        tau_used = float((tau_scale / scale).item())
    else:
        if tau is None:
            raise ValueError("tau must be provided when tau_mode='fixed'")
        tau_used = float(tau)

    exp_evals = torch.exp(-tau_used * evals)
    K = (evecs * exp_evals) @ evecs.T
    K = 0.5 * (K + K.T)
    return K, tau_used



############
########################################################
########################################################    

def regularized_laplacian_kernel_from_adjacency_eigh(
    A: torch.Tensor,
    gamma: Optional[float] = None,
    gamma_mode: str = "auto",
    gamma_scale: float = 1.0,
    normalized_laplacian: bool = True,
) -> Tuple[torch.Tensor, float]:
    """
    Compute K = (I + gamma * Laplacian)^{-1} via eigendecomposition.
    Returns (K, gamma_used). K is float64.

    """
    evals, evecs = laplacian_eigh(A, normalized=normalized_laplacian)

    if gamma_mode == "auto":
        scale = _auto_scale_from_evals(evals)
        gamma_used = float((gamma_scale / scale).item())
    else:
        if gamma is None:
            raise ValueError("gamma must be provided when gamma_mode='fixed'")
        gamma_used = float(gamma)

    inv_evals = 1.0 / (1.0 + gamma_used * evals)
    K = (evecs * inv_evals) @ evecs.T
    K = 0.5 * (K + K.T)
    return K, gamma_used




################
########################

#############

def random_walk_kernel_from_adjacency_eigh(
    A: torch.Tensor,
    p: int = 2,
    a: float = 2.0,
    normalized_laplacian: bool = True,
) -> Tuple[torch.Tensor, int, float]:
    """
    Compute spectral random-walk style kernel K = (aI - L)^p via eigendecomposition.
    Returns (K, p_used, a_used). K is float64.

    With normalized Laplacian on an undirected graph, λ in [0, 2], so a >= 2 ensures
    nonnegative spectral weights (a - λ)^p and PSD.
    For unnormalized Laplacian, we enforce a >= max kept eigenvalue.
    """
    if int(p) < 1:
        raise ValueError(f"p must be >= 1, got {p}")
    p = int(p)
    a = float(a)

    evals, evecs = laplacian_eigh(A, normalized=normalized_laplacian)
    lam_max = float(evals.max().item())
    if a < lam_max - 1e-9:
        raise ValueError(
            f"random_walk kernel requires a >= max Laplacian eigenvalue ({lam_max:.6g}); got a={a}"
        )

    rw_evals = (a - evals).pow(p)
    K = (evecs * rw_evals) @ evecs.T
    K = 0.5 * (K + K.T)
    # Normalize overall scale so rw_p changes shape more than magnitude.
    diag_mean = K.diagonal().mean().clamp_min(1e-12)
    K = K / diag_mean
    return K, p, a



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

def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Stable inverse softplus for y > 0: log(expm1(y))."""
    return torch.log(torch.expm1(y).clamp_min(1e-12))





class LearnableTauGraphTaskKernel(Kernel): # for diffusion kernel
    """
    Task kernel K = exp(-tau * L) with learnable tau; stores Laplacian eigenpairs as buffers.
    """
    is_stationary = False

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
        self.register_buffer("Lap_evals", Lap_evals)
        self.register_buffer("Lap_evecs", Lap_evecs)
        tau0 = torch.tensor(max(float(tau_init), 1e-6), dtype=dtype, device=device)
        raw_tau0 = _softplus_inv(tau0)
        self.register_parameter("raw_tau", torch.nn.Parameter(raw_tau0))

    @property
    def tau(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_tau)

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)  # (..., n)
        i2 = i2.long().squeeze(-1)  # (..., m)
        exp_evals = torch.exp(-self.tau * self.Lap_evals)
        Kfull = (self.Lap_evecs * exp_evals) @ self.Lap_evecs.T
        Kfull = 0.5 * (Kfull + Kfull.T)
        # Use direct 2D indexing: i1 and i2 are flat after squeeze
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

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)
        i2 = i2.long().squeeze(-1)
        inv_evals = 1.0 / (1.0 + self.gamma * self.Lap_evals)
        Kfull = (self.Lap_evecs * inv_evals) @ self.Lap_evecs.T
        Kfull = 0.5 * (Kfull + Kfull.T)
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]
        return DenseLinearOperator(K)






########################################################
########################################################    








def build_graph_multitask_gp(
    net_xml_path: str,
    edge_ids: List[str],
    train_X: torch.Tensor,     # (N, d+1), last column task_id
    train_Y: torch.Tensor,     # (N, 1)
    input_dim: int,            # d
    adjacency_type: str = "shared_node_weighted",
    graph_kernel_type: str = "diffusion",  # "diffusion", "regularized_laplacian", or "random_walk"
    rw_p: int = 2,
    rw_a: float = 2.0,
    tau_mode: str = "auto",
    tau: Optional[float] = None,
    tau_scale: float = 1.0,
    gamma: Optional[float] = None,
    gamma_mode: str = "auto",
    gamma_scale: float = 1.0,
    ignore_internal: bool = True,
    matern_nu: float = 2.5,
    device: Optional[torch.device] = None,
    learnable_tau: bool = False,
    learnable_gamma: bool = False,
    normalized_laplacian: bool = True,
    standardize: bool = True,  # set False when data is pre-standardized per task
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict]:
    """
    Returns (model, mll, info_dict). Model and mll are moved to device.
    graph_kernel_type: "diffusion" -> K = exp(-tau*L); "regularized_laplacian" -> K = (I+gamma*L)^{-1};
    "random_walk" -> K = (rw_a I - L)^rw_p in the Laplacian eigenbasis.
    normalized_laplacian: use L_sym = I - D^{-1/2} A D^{-1/2} (True) or L = D - A (False).
    Uses SingleTaskGP so the product kernel x_kernel*task_kernel is used as-is;
    MultiTaskGP would add an extra IndexKernel on top.
    """
    if device is None:
        device = train_X.device

    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    if train_Y.ndim == 1:
        train_Y = train_Y.view(-1, 1)

    # ---- adjacency
    if adjacency_type == "connections":
        A = adjacency_from_connections(
            net_xml_path,
            edge_ids=edge_ids,
            ignore_internal=ignore_internal,
            undirected=True,
            device=device,
            dtype=torch.float64,
        )
    elif adjacency_type == "shared_node_weighted":
        A = adjacency_by_shared_node_weighted(
            net_xml_path,
            edge_ids=edge_ids,
            ignore_internal=ignore_internal,
            device=device,
            dtype=torch.float64,
            normalize="k-1",
            clamp_max=1.0,
        )
    else:
        raise ValueError(f"Unknown adjacency_type: {adjacency_type}")

    A_density = float((A > 0).float().mean().item())
    isolated = int((A.sum(dim=1) == 0).sum().item())
    tau_used: Optional[float] = None
    gamma_used: Optional[float] = None

    if graph_kernel_type == "diffusion":
        K_graph, tau_used = diffusion_kernel_from_adjacency_eigh(
            A,
            tau=tau,
            tau_mode=tau_mode,
            tau_scale=tau_scale,
            normalized_laplacian=normalized_laplacian,
        )
        K_graph = K_graph.to(device)
        if learnable_tau:
            Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
            task_kernel = LearnableTauGraphTaskKernel(
                Lap_evals=Lap_evals,
                Lap_evecs=Lap_evecs,
                tau_init=tau_used,
                dtype=torch.float64,
                device=device,
            )
        else:
            task_kernel = FixedGraphTaskKernel(K_graph)
    elif graph_kernel_type == "regularized_laplacian":
        K_graph, gamma_used = regularized_laplacian_kernel_from_adjacency_eigh(
            A,
            gamma=gamma,
            gamma_mode=gamma_mode,
            gamma_scale=gamma_scale,
            normalized_laplacian=normalized_laplacian,
        )
        K_graph = K_graph.to(device)
        if learnable_gamma:
            Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
            task_kernel = LearnableGammaGraphTaskKernel(
                Lap_evals=Lap_evals,
                Lap_evecs=Lap_evecs,
                gamma_init=gamma_used,
                dtype=torch.float64,
                device=device,
            )
        else:
            task_kernel = FixedGraphTaskKernel(K_graph)
    elif graph_kernel_type == "random_walk":
        if learnable_tau or learnable_gamma:
            raise ValueError(
                "graph_kernel_type='random_walk' does not support learnable_tau or learnable_gamma."
            )
        K_graph, _, _ = random_walk_kernel_from_adjacency_eigh(
            A,
            p=rw_p,
            a=rw_a,
            normalized_laplacian=normalized_laplacian,
        )
        K_graph = K_graph.to(device)
        task_kernel = FixedGraphTaskKernel(K_graph)
    else:
        raise ValueError(f"Unknown graph_kernel_type: {graph_kernel_type}")
    x_kernel = ScaleKernel(MaternKernel(nu=matern_nu, ard_num_dims=input_dim))
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

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar,
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
        "tau_used": tau_used,
        "gamma_used": gamma_used,
        "A_density": A_density,
        "isolated_tasks": isolated,
        "L_tasks": len(edge_ids),
        "adjacency_type": adjacency_type,
        "graph_kernel_type": graph_kernel_type,
        "learnable_tau": learnable_tau,
        "learnable_gamma": learnable_gamma,
        "normalized_laplacian": normalized_laplacian,
        "rw_p": rw_p if graph_kernel_type == "random_walk" else None,
        "rw_a": rw_a if graph_kernel_type == "random_walk" else None,
    }
    return model, mll, info

def fit_mll_with_fallback(
    mll: ExactMarginalLogLikelihood,
    train_X: torch.Tensor,
    max_adam_iters: int = 200,
    lr: float = 1e-2,
) -> str:
    """
    Fit with LBFGS; on failure fall back to Adam.
    Returns 'lbfgs' or 'adam'.
    SingleTaskGP uses standard GaussianLikelihood; LBFGS via BoTorch, Adam on failure.
    """
    model_device = next(mll.model.parameters()).device
    train_X = train_X.to(model_device)
    model = mll.model

    def _loss_forward(**kwargs):
        model_output = model(*model.train_inputs)
        return -mll(model_output, model.train_targets, **kwargs)

    closure = ForwardBackwardClosure(
        forward=_loss_forward,
        parameters=get_parameters(mll, requires_grad=True),
    )

    try:
        fit_gpytorch_mll(mll, closure=closure)
        return "lbfgs"
    except (NotPSDError, RuntimeError, ModelFittingError):
        model.train()
        model.likelihood.train()
        opt = torch.optim.Adam(mll.parameters(), lr=lr)
        stored_X = model.train_inputs[0]  # use stored inputs, not the external tensor
        for _ in range(max_adam_iters):
            opt.zero_grad()
            try:
                out = model(stored_X)
                loss = -mll(out, model.train_targets)
                loss.backward()
                opt.step()
            except NotPSDError:
                # Skip this step: no backward, no parameter update; continue to next iteration
                continue
        return "adam"

def run_graph_multitask_gp(
    net_xml_path: str,
    edge_ids: List[str],
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    input_dim: int,
    adjacency_type: str = "shared_node_weighted",
    graph_kernel_type: str = "diffusion",
    rw_p: int = 2,
    rw_a: float = 2.0,
    tau_mode: str = "auto",
    tau: Optional[float] = None,
    tau_scale: float = 1.0,
    gamma: Optional[float] = None,
    gamma_mode: str = "auto",
    gamma_scale: float = 1.0,
    ignore_internal: bool = True,
    matern_nu: float = 2.5,
    device: Optional[torch.device] = None,
    learnable_tau: bool = False,
    learnable_gamma: bool = False,
    normalized_laplacian: bool = True,
    standardize: bool = True,  # set False when data is pre-standardized per task
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict, str]:
    """
    Build graph GP and fit.
    Returns (model, mll, info, fit_method).

    rw_p, rw_a: random-walk kernel hyperparameters when graph_kernel_type == "random_walk".
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
        graph_kernel_type=graph_kernel_type,
        rw_p=rw_p,
        rw_a=rw_a,
        tau_mode=tau_mode,
        tau=tau,
        tau_scale=tau_scale,
        gamma=gamma,
        gamma_mode=gamma_mode,
        gamma_scale=gamma_scale,
        ignore_internal=ignore_internal,
        matern_nu=matern_nu,
        device=device,
        learnable_tau=learnable_tau,
        learnable_gamma=learnable_gamma,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
    )

    fit_method = fit_mll_with_fallback(mll, train_X, max_adam_iters=200, lr=1e-2)
    return model, mll, info, fit_method

def edge_ids_from_routes_csv(routes_csv_path: str, column: str = "route_edges") -> List[str]:
    """
    Parse routes CSV: column contains space-separated edge IDs per route.
    Returns sorted list of unique edge IDs.
    """
    df = pd.read_csv(routes_csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in {list(df.columns)}")

    all_edges = set()
    for s in df[column].dropna().astype(str):
        # If your CSV uses commas, uncomment:
        # s = s.replace(",", " ")
        all_edges.update(s.split())

    return sorted(all_edges)

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

if __name__ == "__main__":
    # Adjust these paths to your project layout
    base = Path(__file__).resolve().parent.parent
    net_xml = base / "network" / "quickstart" / "net.xml"

    # Use the 6 representative edges, NOT all raw edges from routes.csv.
    # edge_ids_from_routes_csv gives 14 raw edges (D2, L2, L12, ...) which would
    # create 14 GP tasks. The 6 representatives are the meaningful link-level tasks.
    from src.utils.edge_transition_matrix import QUICKSTART_EDGE_GROUPS
    edge_ids = sorted(QUICKSTART_EDGE_GROUPS.keys())  # ['L10', 'L12', 'L14', 'L15', 'L16', 'L18']

    N = 10
    d = 4
    L = len(edge_ids)
    torch.manual_seed(0)

    X_cont = torch.rand(N, d)
    task_ids = torch.randint(low=0, high=L, size=(N, 1)).float()
    train_X = torch.cat([X_cont, task_ids], dim=1)
    train_Y = torch.randn(N, 1)

    device = torch.device("cpu")

    model, mll, info, fit_method = run_graph_multitask_gp(
        net_xml_path=str(net_xml),
        edge_ids=edge_ids,
        train_X=train_X,
        train_Y=train_Y,
        input_dim=d,
        adjacency_type="shared_node_weighted",
        tau_mode="auto",
        tau_scale=1.0,
        ignore_internal=True,
        matern_nu=2.5,
        device=device,
        normalized_laplacian=True,
    )

    print("Graph kernel info:", info)
    print("Fit method used:", fit_method)

