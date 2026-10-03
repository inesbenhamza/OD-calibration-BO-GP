import copy                          # for deepcopy of model state dicts during multi-restart fitting
import math                          # for math.inf and math.isfinite
import torch                         # main tensor library
import gpytorch                      # GP framework built on PyTorch
import xml.etree.ElementTree as ET   # parse SUMO net.xml to extract edge topology
from pathlib import Path             # filesystem path utilities
import pandas as pd                  # read route CSV files
from typing import Optional, Tuple, Dict, List  # type hints

from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel, MultitaskKernel         # kernel building blocks
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood        # noise models
from gpytorch.constraints import Interval                                                # box constraints on hyperparameters
from gpytorch.priors import HalfNormalPrior                                              # soft penalty on rho^2 (SharedPrivateGraphKernel)
from gpytorch.mlls import ExactMarginalLogLikelihood                                    # MLL objective for hyperparameter optimization
from gpytorch.models import ExactGP                                                      # base class for exact GPs
from gpytorch.means import MultitaskMean, ConstantMean                                  # prior mean functions
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal      # GP output distributions

from torch import Tensor                                    # tensor type alias for type hints
from botorch.models import SingleTaskGP                     # BoTorch single-output GP (used for stacked graph GP)
from botorch.models.model import Model                      # BoTorch base model class
from botorch.models.gpytorch import GPyTorchModel           # BoTorch mixin that adds .posterior() to GPyTorch models
from botorch.posteriors.gpytorch import GPyTorchPosterior   # BoTorch posterior wrapper, required by acquisition functions
from botorch.fit import fit_gpytorch_mll                    # L-BFGS optimizer for MLL
from botorch.exceptions import ModelFittingError            # raised when L-BFGS fails to converge

from linear_operator.utils.errors import NotPSDError                              # raised when covariance matrix is not positive semi-definite
from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator, DiagLinearOperator  # lazy linear operators for efficient matrix ops


def _ok_edge(eid: Optional[str], ignore_internal: bool) -> bool:
    # return False for None or for internal SUMO connector edges (ids starting with ":")
    if eid is None:
        return False                              # skip edges with no id
    if ignore_internal and eid.startswith(":"):
        return False                              # skip SUMO internal junction connector edges
    return True                                   # edge is valid

def adjacency_route_overlap_weighted(
    routes_csv_path: str,
    edge_ids: List[str],
    route_edges_col: str = "route_edges",   # column in CSV listing the edge IDs on each route
    ratio_col: str = "ratio",               # column with split ratios (probability of each route per OD)
    from_taz_col: str = "fromTaz",          # origin TAZ column
    to_taz_col: str = "toTaz",             # destination TAZ column
    edge_groups: Optional[Dict[str, List[str]]] = None,  # maps group label → list of underlying SUMO edge IDs
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    verbose: bool = False,
    zero_diagonal: bool = True,
) -> torch.Tensor:
    """
    Build the L×L task kernel matrix A for a multi-output GP.

        W  (L × R)  binary          W[i,r]  = 1 if edge i ∈ route r, else 0
        W2 (R × D)  probabilities   W2[r,d] = P(route r | OD d)  (split ratio)
        M  (L × D)  = W @ W2        M[i,d]  = P(edge i used | OD d)
        A  (L × L)  = (M @ Mᵀ) / D  A[i,j]  = (1/D) Σ_d P(i|d)·P(j|d), zero diagonal

    Dividing by D (number of OD pairs) keeps A in [0, 1] and makes the scale
    invariant to network size, preventing confounding with the GP output variance.

    If the CSV has no ratio column (single-route-per-OD), each route gets
    probability 1.0 for its OD pair.

    edge_groups : optional dict mapping each entry of `edge_ids` (a group
        label, e.g. "c0_in_91") to the list of underlying SUMO edge IDs that
        make up that group (e.g. ["c0_D2", "c0_L2", "c0_L12"]). Needed when a
        group label is a synthetic name that never appears verbatim in
        `route_edges` — without it, such groups get an all-zero row in A.

    Raises ValueError if the OD columns are absent — W2 cannot be built.
    """
    L = len(edge_ids)                              # number of edges/links
    edge_to_idx = {e: i for i, e in enumerate(edge_ids)}  # edge id → integer index

    # build token_to_idx: maps every underlying SUMO edge id AND every group label to its row index
    # this lets route_edges tokens resolve correctly even for merged-chain groups
    token_to_idx: Dict[str, int] = dict(edge_to_idx)
    if edge_groups is not None:
        for label, idx in edge_to_idx.items():
            for member in edge_groups.get(label, [label]):
                token_to_idx[member] = idx  # every member of the group maps to the group's index

    df = pd.read_csv(routes_csv_path).reset_index(drop=True)  # load routes CSV, reset index for clean iteration
    R = len(df)  # number of routes

    # validate that OD columns are present — needed to build W2
    missing = [c for c in (from_taz_col, to_taz_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"W2 cannot be built: column(s) {missing} not found in "
            f"'{routes_csv_path}'.\nAvailable: {list(df.columns)}"
        )

    # build OD pair index: unique (fromTaz, toTaz) pairs in row order
    od_pairs  = list(dict.fromkeys(zip(df[from_taz_col], df[to_taz_col])))  # preserves first-occurrence order
    od_to_idx = {od: d for d, od in enumerate(od_pairs)}                    # (fromTaz, toTaz) → integer index
    D = len(od_pairs)  # number of unique OD pairs

    # split ratios: use ratio column if present, else 1.0 per route (single route per OD)
    has_ratio = ratio_col in df.columns
    ratios = df[ratio_col].astype(float).tolist() if has_ratio else [1.0] * R

    # W: binary edge-route incidence matrix (L × R), W[i,r]=1 if edge i is on route r
    W = torch.zeros(L, R, dtype=dtype, device=device)
    for r, edges_str in enumerate(df[route_edges_col].astype(str)):
        for eid in edges_str.split():           # each route lists its edges space-separated
            if eid in token_to_idx:
                W[token_to_idx[eid], r] = 1.0  # mark this edge as part of this route

    # W2: route-OD probability matrix (R × D), W2[r,d] = P(route r | OD d)
    W2 = torch.zeros(R, D, dtype=dtype, device=device)
    d_indices = torch.tensor(
        [od_to_idx[(row[from_taz_col], row[to_taz_col])] for _, row in df.iterrows()],
        dtype=torch.long, device=device,
    )  # for each route row, which OD column it belongs to
    W2[torch.arange(R, device=device), d_indices] = torch.tensor(
        ratios, dtype=dtype, device=device
    )  # fill in the split ratios at the correct (route, OD) positions

    # M = W @ W2 → (L × D): P(edge i used | OD d)
    # A = (M @ Mᵀ) / D → (L × L): average co-usage probability across all OD pairs
    M = W @ W2          # (L × D) — probability that edge i is used for OD d
    A = (M @ M.T) / D  # (L × L) — A[i,j] = mean over ODs of P(i|d)*P(j|d)
    if zero_diagonal:
        A.fill_diagonal_(0.0)  # zero the diagonal — self-correlation not meaningful for task kernel
    # else: A stays a proper Gram matrix (M @ M.T, PSD by construction) with
    # its diagonal intact — used by build_route_overlap_correlation_matrix to
    # correlation-normalize without re-deriving M from scratch.

    if verbose:
        print(f"  W  : {tuple(W.shape)}   binary edge-route incidence")
        print(f"  W2 : {tuple(W2.shape)}   "
              f"{'split ratios from ratio col' if has_ratio else 'uniform 1.0 (no ratio col)'}")
        print(f"  M  : {tuple(M.shape)}   P(edge | OD)")
        print(f"  A  : {tuple(A.shape)}   task kernel / D={D}  (zero diagonal)")
        print(f"  A non-zero entries : {int((A > 0).sum())}")
        print(f"  A max : {A.max().item():.4f}")
        print(f"  A[0:3, 0:3] :\n{A[:3, :3]}")

    return A  # L×L task adjacency matrix based on route overlap


def laplacian_from_adjacency(
    A: torch.Tensor,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Build Laplacian from adjacency.
    - If normalized: L_sym = I - D^{-1/2} A D^{-1/2}
    - Else: L = D - A
    """
    A = 0.5 * (A + A.T)          # symmetrize A to handle any floating-point asymmetry
    A = A.to(dtype=torch.float64) # use float64 for numerical stability in eigendecomposition

    deg = A.sum(dim=1)  # degree of each node: deg[i] = sum of row i of A

    if normalized:
        d = deg.pow(-0.5)              # D^{-1/2}: entries are 1/sqrt(degree_i)
        d[torch.isinf(d)] = 0.0       # isolated nodes (degree=0): set 1/sqrt(0) = 0 to avoid NaN
        A_norm = (A * d[:, None]) * d[None, :]  # D^{-1/2} A D^{-1/2}: scale each entry A[i,j] by 1/sqrt(d_i*d_j)
        L = torch.eye(A.shape[0], dtype=A.dtype, device=A.device) - A_norm  # L_sym = I - D^{-1/2} A D^{-1/2}
    else:
        L = -A.clone()         # start with -A
        L.diagonal().copy_(deg)  # set diagonal to degree: L = D - A (unnormalized Laplacian)

    L = 0.5 * (L + L.T)  # re-symmetrize to eliminate any tiny numerical asymmetry
    return L              # L×L Laplacian matrix


def laplacian_eigh(
    A: torch.Tensor,
    normalized: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecomposition of Laplacian from adjacency.
    Returns (evals, evecs) with evals clamped to >= 0.

    torch.linalg.eigh is ascending and returns all eigenpairs.
    """
    L_mat = laplacian_from_adjacency(A, normalized=normalized)    # build the Laplacian matrix
    evals, evecs = torch.linalg.eigh(L_mat)                       # eigendecomposition: evals ascending, evecs orthonormal
    evals = torch.clamp(evals, min=0.0)                           # clamp to >= 0 (numerical errors can give tiny negatives)
    return evals, evecs  # evals: Laplacian eigenvalues [L], evecs: eigenvectors [L, L]


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """
    Stable inverse softplus for y > 0: log(expm1(y)) → log(1 + exp(raw_y)).

    The optimizer works in unconstrained space of raw_param, so we need to transform
    the initial value to raw_param using the inverse softplus function.
    _softplus_inv is only called at initialization to set the starting point of raw_param
    such that softplus(raw_param) = param_init. After that, the optimizer takes over.

    Training flow:
        1. INIT: raw_γ ≈ 0.541  (so that softplus(raw_γ) = 1.0)
        2. LOOP: γ = softplus(raw_γ) → K = (I + γL)⁻¹ → MLL → backprop → update raw_γ
        3. CONVERGE: γ_final = softplus(raw_γ_final) → used in predictions
    """
    return torch.log(torch.expm1(y).clamp_min(1e-12))  # clamp_min avoids log(0) for very small y


class LearnableTauGraphTaskKernel(Kernel):
    """
    Task kernel K = exp(-tau * L) with learnable tau; stores Laplacian eigenpairs as buffers.
    This is the diffusion kernel: models how information spreads on the graph over time tau.
    """
    is_stationary = False    # kernel depends on graph structure, not just distance
    TAU_LB: float = 0.001   # minimum tau: high diffusion (edges are very similar)
    TAU_UB: float = 10.0    # maximum tau: low diffusion (edges are nearly independent)

    def __init__(
        self,
        Lap_evals: torch.Tensor,   # Laplacian eigenvalues [L], fixed from road network
        Lap_evecs: torch.Tensor,   # Laplacian eigenvectors [L, L], fixed from road network
        tau_init: float,           # initial value of tau (diffusion time)
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)  # move eigenvalues to correct dtype/device
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)  # move eigenvectors to correct dtype/device
        self.register_buffer("Lap_evals", Lap_evals)  # λ_i — fixed from road network, not trained
        self.register_buffer("Lap_evecs", Lap_evecs)  # V   — fixed from road network, not trained

        tau0 = float(min(max(tau_init, self.TAU_LB), self.TAU_UB))  # clip tau_init to valid range
        t01 = max(1e-6, min((tau0 - self.TAU_LB) / (self.TAU_UB - self.TAU_LB), 1 - 1e-6))  # normalize to (0,1) for sigmoid
        raw_tau0 = torch.tensor(math.log(t01 / (1.0 - t01)), dtype=dtype, device=device)  # inverse sigmoid: unconstrained init
        self.register_parameter("raw_tau", torch.nn.Parameter(raw_tau0))  # raw_tau is the trainable parameter

    @property
    def tau(self) -> torch.Tensor:
        # constrain tau to [TAU_LB, TAU_UB] via sigmoid: tau = LB + (UB-LB)*sigmoid(raw_tau)
        return self.TAU_LB + (self.TAU_UB - self.TAU_LB) * torch.sigmoid(self.raw_tau)

    def _full_matrix(self) -> torch.Tensor:
        exp_evals = torch.exp(-self.tau * self.Lap_evals)       # spectral: exp(-τλ_i) for each eigenvalue
        Kfull = (self.Lap_evecs * exp_evals) @ self.Lap_evecs.T  # K = V diag(exp(-τλ)) Vᵀ
        return 0.5 * (Kfull + Kfull.T)                           # symmetrize for numerical stability

    @property
    def covar_matrix(self):
        # required by MultitaskKernel.forward() — it accesses task_covar_module.covar_matrix
        return DenseLinearOperator(self._full_matrix())  # wrap in lazy operator for GPyTorch compatibility

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)               # task indices for first set of inputs
        i2 = i2.long().squeeze(-1)               # task indices for second set of inputs
        Kfull = self._full_matrix()              # compute full L×L kernel matrix
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]  # index into K to get the requested (i1, i2) entries
        return DenseLinearOperator(K)            # return as lazy operator


class LearnableGammaGraphTaskKernel(Kernel):
    """
    Task kernel K = (I + gamma * L)^{-1} with learnable gamma; stores Laplacian eigenpairs as buffers.
    This is the regularized Laplacian kernel: models spatial smoothness with strength controlled by gamma.
    """
    is_stationary = False  # kernel depends on graph structure, not just distance

    def __init__(
        self,
        Lap_evals: torch.Tensor,   # Laplacian eigenvalues [L]
        Lap_evecs: torch.Tensor,   # Laplacian eigenvectors [L, L]
        gamma_init: float,         # initial value of gamma (regularization strength)
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)  # move to correct dtype/device
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)  # move to correct dtype/device
        self.register_buffer("Lap_evals", Lap_evals)           # fixed from road network
        self.register_buffer("Lap_evecs", Lap_evecs)           # fixed from road network
        g0 = torch.tensor(max(float(gamma_init), 1e-12), dtype=dtype, device=device)  # initial gamma, clipped away from 0
        raw_g0 = _softplus_inv(g0)                              # transform to unconstrained space
        self.register_parameter("raw_gamma", torch.nn.Parameter(raw_g0))  # trainable unconstrained parameter

    @property
    def gamma(self) -> torch.Tensor:
        # constrain gamma > 0 via softplus: gamma = log(1 + exp(raw_gamma))
        return torch.nn.functional.softplus(self.raw_gamma)

    def _full_matrix(self) -> torch.Tensor:
        inv_evals = 1.0 / (1.0 + self.gamma * self.Lap_evals)   # spectral: 1/(1 + γλ_i) for each eigenvalue
        Kfull = (self.Lap_evecs * inv_evals) @ self.Lap_evecs.T  # K = V diag(1/(1+γλ)) Vᵀ
        return 0.5 * (Kfull + Kfull.T)                           # symmetrize for numerical stability

    @property
    def covar_matrix(self):
        # required by MultitaskKernel.forward() — it accesses task_covar_module.covar_matrix
        return DenseLinearOperator(self._full_matrix())

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)               # task indices for first set of inputs
        i2 = i2.long().squeeze(-1)               # task indices for second set of inputs
        Kfull = self._full_matrix()              # compute full L×L kernel matrix
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]  # index into K for requested (i1, i2) pairs
        return DenseLinearOperator(K)


class GraphMaternTaskKernel(Kernel):
    """
    Spectral Matérn task kernel on a graph:
        K = V · diag((2ν/κ² + λ)^{-ν}) · Vᵀ
    with learnable smoothness ν and length-scale κ.
    Stores Laplacian eigenpairs as buffers (fixed from road network).
    """
    is_stationary = False  # kernel depends on graph structure

    def __init__(
        self,
        Lap_evals: torch.Tensor,   # Laplacian eigenvalues [L]
        Lap_evecs: torch.Tensor,   # Laplacian eigenvectors [L, L]
        nu_init: float = 1.5,      # initial smoothness parameter ν
        kappa_init: float = 1.0,   # initial length-scale κ
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        Lap_evals = Lap_evals.to(dtype=dtype, device=device)  # move to correct dtype/device
        Lap_evecs = Lap_evecs.to(dtype=dtype, device=device)  # move to correct dtype/device
        self.register_buffer("Lap_evals", Lap_evals)           # fixed eigenvalues from road network
        self.register_buffer("Lap_evecs", Lap_evecs)           # fixed eigenvectors from road network

        nu0 = torch.tensor(max(float(nu_init), 1e-6), dtype=dtype, device=device)  # initial ν, clipped away from 0
        self.register_parameter("raw_nu", torch.nn.Parameter(_softplus_inv(nu0)))  # trainable unconstrained ν

        k0 = torch.tensor(max(float(kappa_init), 1e-6), dtype=dtype, device=device)  # initial κ, clipped away from 0
        self.register_parameter("raw_kappa", torch.nn.Parameter(_softplus_inv(k0))) # trainable unconstrained κ

    @property
    def nu(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_nu)    # constrain ν > 0 via softplus

    @property
    def kappa(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_kappa) # constrain κ > 0 via softplus

    def _full_matrix(self) -> torch.Tensor:
        Psi = (2.0 * self.nu / self.kappa ** 2 + self.Lap_evals).pow(-self.nu)  # spectral weights: (2ν/κ² + λ)^{-ν}
        Kfull = (self.Lap_evecs * Psi) @ self.Lap_evecs.T                        # K = V diag(Psi) Vᵀ
        return 0.5 * (Kfull + Kfull.T)                                           # symmetrize for numerical stability

    @property
    def covar_matrix(self):
        return DenseLinearOperator(self._full_matrix())  # required by MultitaskKernel.forward()

    def forward(self, i1, i2, **params):
        i1 = i1.long().squeeze(-1)               # task indices for first set of inputs
        i2 = i2.long().squeeze(-1)               # task indices for second set of inputs
        Kfull = self._full_matrix()              # compute full L×L kernel matrix
        K = Kfull[i1.unsqueeze(-1), i2.unsqueeze(-2)]  # index into K for requested (i1, i2) pairs
        return DenseLinearOperator(K)


def build_graph_multitask_gp(
    net_xml_path: Optional[str] = None,    # required for shared_node_weighted adjacency
    edge_ids: List[str] = None,
    train_X: torch.Tensor = None,          # (N, d+1): first d cols = OD inputs, last col = task_id
    train_Y: torch.Tensor = None,          # (N, 1): scalar target per row
    input_dim: int = None,                 # d: number of OD input dimensions (excluding task column)
    adjacency_type: str = "shared_node_weighted",
    routes_csv_path: Optional[str] = None,      # required for route_overlap_weighted adjacency
    adjacency_tensor: Optional[torch.Tensor] = None,  # required for precomputed adjacency
    edge_groups: Optional[Dict[str, List[str]]] = None,  # group label → underlying SUMO edge IDs
    graph_kernel_type: str = "diffusion",  # "diffusion", "regularized_laplacian", or "matern"
    ignore_internal: bool = True,
    matern_nu: float = 2.5,
    device: Optional[torch.device] = None,
    normalized_laplacian: bool = True,
    standardize: bool = True,              # if True, standardize per task before fitting
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict]:
    """
    Returns (model, mll, info_dict). Model and mll are moved to device.
    graph_kernel_type: "diffusion" -> K = exp(-tau*L); "regularized_laplacian" -> K = (I+gamma*L)^{-1}.
    normalized_laplacian: use L_sym = I - D^{-1/2} A D^{-1/2} (True) or L = D - A (False).
    Uses SingleTaskGP so the product kernel x_kernel*task_kernel is used as-is;
    MultiTaskGP would add an extra IndexKernel on top.


    Gp receive train_X (6, d+1)          
    train_Y (6, 1)
[x1, 0]  ← x1, edge 0     [e1_0]
[x1, 1]  ← x1, edge 1     [e1_1]
[x2, 0]  ← x2, edge 0     [e2_0]
[x2, 1]  ← x2, edge 1     [e2_1]
[x3, 0]  ← x3, edge 0     [e3_0]
[x3, 1]  ← x3, edge 1     [e3_1]

So build_graph_multitask_gp / SingleTaskGP never sees the original (N, L) shape — it receives already-stacked (N*L, d+1) data. The stacking is the caller's responsibility, not the model's.

This is the key difference from ICMMultiOutputGP and GraphKroneckerGP which both accept train_X as (N, d) and train_Y as (N, L) natively.
    """
    if adjacency_type in ("connections", "shared_node_weighted") and net_xml_path is None:
        raise ValueError(f"net_xml_path is required for adjacency_type='{adjacency_type}'")
    if adjacency_type == "route_overlap_weighted" and routes_csv_path is None:
        raise ValueError("routes_csv_path is required for adjacency_type='route_overlap_weighted'")
    if adjacency_type == "precomputed" and adjacency_tensor is None:
        raise ValueError("adjacency_tensor is required for adjacency_type='precomputed'")

    if device is None:
        device = train_X.device  # infer device from training data if not provided

    train_X = train_X.to(device)  # move training inputs to target device
    train_Y = train_Y.to(device)  # move training targets to target device
    if train_Y.ndim == 1:          # if targets are 1D (N,), add output dim to get (N, 1)
        train_Y = train_Y.view(-1, 1)  # SingleTaskGP expects 2D targets

    # build the adjacency matrix A from the road network
 
    if adjacency_type == "route_overlap_weighted":
        A = adjacency_route_overlap_weighted(
            routes_csv_path,
            edge_ids=edge_ids,
            edge_groups=edge_groups,
            device=device,
            dtype=torch.float64,
        )
    elif adjacency_type == "precomputed":
        A  = adjacency_tensor.to(dtype=torch.float64, device=device)
    else:
        raise ValueError(f"Unknown adjacency_type: {adjacency_type}")

    A_density = float((A > 0).float().mean().item())  # fraction of non-zero entries: 1.0 = fully connected
    isolated  = int((A.sum(dim=1) == 0).sum().item()) # number of edges with no neighbors in A

    # build the graph task kernel from the Laplacian eigendecomposition
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

    # no ScaleKernel: an outputscale on x_kernel would be redundant with task_kernel's own
    # scale parameters (tau/gamma/kappa), creating a non-identifiable direction in the MLL
    x_kernel = MaternKernel(nu=matern_nu, lengthscale_constraint=Interval(0.005, 20.0))  # shared input kernel
    x_kernel    = x_kernel.to(device)    # move to target device
    task_kernel = task_kernel.to(device) # move to target device



    x_kernel.active_dims = torch.tensor(
        list(range(input_dim)), dtype=torch.long, device=device
    ) # register which column x ekernle should care about 
    
    
    # stack the kernels: product kernel K((x,t),(x',t')) = k_data(x,x') * k_task(t,t')
    # x_kernel acts on the first d columns of train_X (the OD inputs)
    task_kernel.active_dims = torch.tensor(
        [input_dim], dtype=torch.long, device=device
    )  # task_kernel acts on the last column of train_X (the task/edge index)
    covar = x_kernel * task_kernel  # product kernel: K((x,t),(x',t')) = k_data(x,x') * k_task(t,t')

    task_mean = None  # will hold per-task mean if standardize=True
    task_std  = None  # will hold per-task std if standardize=True

    # per-task standardization: subtract per-edge mean and divide by per-edge std
    # done manually (not via botorch's Standardize) so we control the un-standardization in posterior()
    if standardize:
        L_tasks  = len(edge_ids)
        task_ids = train_X[:, input_dim].long()  # extract task index column from train_X
        y        = train_Y.squeeze(-1)           # flatten targets to (N,) for per-task computation

        task_mean = torch.zeros(L_tasks, dtype=y.dtype, device=y.device)  # one mean per task
        task_std  = torch.ones(L_tasks, dtype=y.dtype, device=y.device)   # one std per task

        for t in range(L_tasks):
            mask = (task_ids == t) & (~torch.isnan(y))  # select rows belonging to task t, excluding NaNs
            if mask.any():
                task_mean[t] = y[mask].mean()                       # per-task mean
                task_std[t]  = y[mask].std(unbiased=False).clamp_min(1e-6)  # per-task std, clamped to avoid divide-by-zero
            else:
                task_mean[t] = 0.0  # no data for this task: keep identity transform
                task_std[t]  = 1.0

        y_std    = (y - task_mean[task_ids]) / task_std[task_ids]  # standardize each row by its task's stats
        train_Y  = y_std.unsqueeze(-1)                             # reshape back to (N, 1)

    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1))  # single global noise (shared across all tasks)
    likelihood = likelihood.to(device)  # move to target device

    # SingleTaskGP: uses our product kernel directly; correct here since the kernel already handles tasks via active_dims
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        covar_module=covar,
        likelihood=likelihood,
        outcome_transform=None,  # no botorch Standardize — we did it manually above
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)  # MLL objective wiring model and likelihood

    # attach per-task standardization stats so GraphKernelAggregationModel can un-standardize at prediction time
    if standardize:
        model.register_buffer("task_mean", task_mean.view(-1))  # [L] — registered as buffer (not trainable)
        model.register_buffer("task_std",  task_std.view(-1))   # [L]

    model = model.to(device)  # move model to target device
    mll   = mll.to(device)   # move MLL to target device

    info = {
        "A_density":           A_density,          # fraction of non-zero entries in A
        "isolated_tasks":      isolated,            # number of links with no neighbors
        "L_tasks":             len(edge_ids),       # total number of links
        "adjacency_type":      adjacency_type,      # "shared_node_weighted" or "route_overlap_weighted"
        "graph_kernel_type":   graph_kernel_type,   # "diffusion", "regularized_laplacian", or "matern"
        "normalized_laplacian": normalized_laplacian,  # True/False
    }
    return model, mll, info  # return model for predictions, mll for fitting, info for diagnostics









def _uniform(shape, lb: float, ub: float, device, dtype) -> torch.Tensor:
    # create a tensor of given shape filled with uniform random values in [lb, ub]
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lb, ub)


def randomize_graph_hyperparams(model: SingleTaskGP) -> None:
    # randomize all hyperparameters of the stacked graph GP within their valid constrained ranges
    device = model.train_inputs[0].device  # get device from training data
    dtype  = model.train_inputs[0].dtype   # get dtype from training data

    with torch.no_grad():  # no gradients needed — just setting values
        x_kern    = model.covar_module.kernels[0]  # the Matérn input kernel (first factor of product)
        task_kern = model.covar_module.kernels[1]  # the graph task kernel (second factor of product)

        # randomize outputscale if x_kern is wrapped in a ScaleKernel (optional)
        if hasattr(x_kern, "outputscale"):
            ls_lb, ls_ub = 0.05, 2.0  # default outputscale range
            if hasattr(x_kern, "raw_outputscale_constraint"):
                c = x_kern.raw_outputscale_constraint
                if math.isfinite(float(c.lower_bound)): ls_lb = float(c.lower_bound)
                if math.isfinite(float(c.upper_bound)): ls_ub = float(c.upper_bound)
            x_kern.outputscale = _uniform(x_kern.outputscale.shape, ls_lb, ls_ub, device, dtype)

        # randomize lengthscale of the base Matérn kernel
        base_kern = x_kern.base_kernel if hasattr(x_kern, "base_kernel") else x_kern  # unwrap ScaleKernel if present
        l_lb, l_ub = 0.005, 20.0  # default lengthscale range
        if hasattr(base_kern, "raw_lengthscale_constraint"):
            c = base_kern.raw_lengthscale_constraint
            if math.isfinite(float(c.lower_bound)): l_lb = float(c.lower_bound)
            if math.isfinite(float(c.upper_bound)): l_ub = float(c.upper_bound)
        base_kern.lengthscale = _uniform(
            base_kern.lengthscale.shape, l_lb, l_ub, device, dtype
        )

        # randomize graph kernel scalar parameter (gamma, tau, or nu/kappa depending on kernel type)
        if hasattr(task_kern, "raw_gamma"):   # regularized Laplacian kernel
            g_new = _uniform((), 0.1, 20.0, device, dtype)
            task_kern.raw_gamma.copy_(_softplus_inv(g_new))  # set in unconstrained space
        if hasattr(task_kern, "raw_tau"):     # diffusion kernel
            t_new = _uniform((), LearnableTauGraphTaskKernel.TAU_LB, LearnableTauGraphTaskKernel.TAU_UB * 0.5, device, dtype)
            t01 = ((t_new - LearnableTauGraphTaskKernel.TAU_LB) / (LearnableTauGraphTaskKernel.TAU_UB - LearnableTauGraphTaskKernel.TAU_LB)).clamp(1e-6, 1 - 1e-6)
            task_kern.raw_tau.copy_(torch.log(t01 / (1.0 - t01)))  # inverse sigmoid for sigmoid-constrained tau
        if hasattr(task_kern, "raw_nu"):      # graph Matérn: randomize smoothness ν
            nu_new = _uniform((), 0.5, 3.0, device, dtype)
            task_kern.raw_nu.copy_(_softplus_inv(nu_new))
        if hasattr(task_kern, "raw_kappa"):   # graph Matérn: randomize length-scale κ
            k_new = _uniform((), 0.1, 5.0, device, dtype)
            task_kern.raw_kappa.copy_(_softplus_inv(k_new))

        # randomize observation noise within its constrained range
        n_lb, n_ub = 1e-6, 0.1  # default noise range
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
    model          = mll.model                            # extract the GP model
    original_state = copy.deepcopy(model.state_dict())   # save initial state to reset between restarts
    best_state     = None                                 # holds the best state dict found so far
    best_mll_val   = -math.inf                            # tracks the best MLL value seen

    for r in range(n_restarts):
        model.load_state_dict(original_state)   # reset to original init before each restart
        randomize_graph_hyperparams(model)      # always randomize (restart 0 also randomizes from init)

        model.train()  # training mode: enables gradient computation
        mll.train()    # same for MLL
        try:
            fit_gpytorch_mll(mll)  # run L-BFGS to maximize MLL
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"  [graph restart {r+1}/{n_restarts}] LBFGS failed ({str(e)[:50]}), skipping.")
            continue  # skip this restart
        model.train()  # re-enter train mode to evaluate MLL (fit_gpytorch_mll leaves model in eval)
        mll.train()    # same for MLL
        with torch.no_grad():  # no gradients needed for MLL evaluation
            try:
                out     = model(*model.train_inputs)           # forward pass on training data
                mll_val = mll(out, model.train_targets).item() # compute MLL as a scalar
            except Exception:
                mll_val = -math.inf  # treat failed evaluation as worst possible

        # reject degenerate solutions: very high noise or very low outputscale → poor surrogate
        noise_val = model.likelihood.noise.item()
        os_val    = model.covar_module.kernels[0].outputscale.item() \
                    if hasattr(model.covar_module.kernels[0], "outputscale") else 1.0
        if noise_val > 0.08 or os_val < 1e-4:  # degenerate: all noise or no signal
            if verbose:
                print(f"  [graph restart {r+1}/{n_restarts}] degenerate (noise={noise_val:.3f}, os={os_val:.2e}), skipping.")
            continue  # skip degenerate solutions

        if verbose:
            print(f"  [graph restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")

        if mll_val > best_mll_val:            # new best restart found
            best_mll_val = mll_val            # update best MLL
            best_state   = copy.deepcopy(model.state_dict())  # save this restart's state

    if best_state is not None:               # at least one L-BFGS restart succeeded
        model.load_state_dict(best_state)    # restore best hyperparameters
        model.eval()                         # switch to prediction mode
        mll.eval()                           # same for MLL
        if verbose:
            print(f"  → best MLL={best_mll_val:.4f}")
        return "lbfgs-multistart"            # signal L-BFGS succeeded

    # all restarts failed — Adam fallback on original init
    model.load_state_dict(original_state)   # reset to original init for Adam
    model.train()                           # training mode for Adam
    mll.train()                             # same for MLL
    opt      = torch.optim.Adam(mll.parameters(), lr=1e-2)  # Adam optimizer over all hyperparameters
    stored_X = model.train_inputs[0]                        # cache training inputs
    for _ in range(200):               # 200 Adam steps
        opt.zero_grad()                # clear gradients
        try:
            loss = -mll(model(stored_X), model.train_targets)              # negative MLL = loss to minimize
            loss.backward()                                                 # compute gradients
            torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=1.0) # clip gradients to prevent exploding updates
            opt.step()                                                      # update hyperparameters
        except NotPSDError:
            continue  # skip step if covariance becomes non-PSD mid-optimization
    model.eval()  # prediction mode after Adam
    mll.eval()    # same for MLL
    return "adam-fallback"  # signal Adam was used


def run_graph_multitask_gp(
    net_xml_path: Optional[str] = None,
    edge_ids: List[str] = None,
    train_X: torch.Tensor = None,
    train_Y: torch.Tensor = None,
    input_dim: int = None,
    adjacency_type: str = "shared_node_weighted",
    routes_csv_path: Optional[str] = None,
    adjacency_csv_path: Optional[str] = None,
    adjacency_tensor: Optional[torch.Tensor] = None,
    edge_groups: Optional[Dict[str, List[str]]] = None,
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
        device = train_X.device  # infer device from training data

    # build and initialize the graph GP
    model, mll, info = build_graph_multitask_gp(
        net_xml_path=net_xml_path,
        edge_ids=edge_ids,
        train_X=train_X,
        train_Y=train_Y,
        input_dim=input_dim,
        adjacency_type=adjacency_type,
        routes_csv_path=routes_csv_path,
        adjacency_csv_path=adjacency_csv_path,
        adjacency_tensor=adjacency_tensor,
        edge_groups=edge_groups,
        graph_kernel_type=graph_kernel_type,
        ignore_internal=ignore_internal,
        matern_nu=matern_nu,
        device=device,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
    )

    fit_method = fit_graph_mll_multistart(mll, n_restarts=n_gp_restarts)  # fit hyperparameters via multi-start L-BFGS
    return model, mll, info, fit_method  # return fitted model, mll, diagnostics, and fit method used


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
        super().__init__()                                    # initialize BoTorch Model base class
        self.graph_model = graph_model                        # the fitted stacked graph GP
        self.register_buffer("w", weights.view(-1))          # (L,) fixed aggregation weights, not trainable

        # if task_mean/task_std not provided explicitly, try to read them from the fitted model
        if (task_mean is None or task_std is None) and hasattr(graph_model, "task_mean") and hasattr(graph_model, "task_std"):
            task_mean = getattr(graph_model, "task_mean")
            task_std  = getattr(graph_model, "task_std")

        if task_mean is not None and task_std is not None:
            self.register_buffer("task_mean", task_mean.view(-1))  # (L,) per-task mean for un-standardization
            self.register_buffer("task_std",  task_std.view(-1))   # (L,) per-task std for un-standardization
        else:
            self.task_mean = None  # no standardization was applied
            self.task_std  = None

    @property
    def num_outputs(self) -> int:
        return 1  # scalar S(x) = w^T e(x), required by BoTorch acquisition functions

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        if X.dim() == 3:
            b, q, d = X.shape   # b = candidates, q = points per candidate, d = input dim
            if q != 1:
                raise RuntimeError("q=1 only.")  # only single-point acquisition supported
            X = X.view(b, d)    # squeeze out q=1: [b, 1, d] → [b, d]
        elif X.dim() == 2:
            b, d = X.shape      # already [b, d], nothing to do
        else:
            raise RuntimeError(f"Unexpected X.dim()={X.dim()}")

        L = self.w.shape[0]     # number of tasks/edges
        self.graph_model.eval() # switch to prediction mode
        if hasattr(self.graph_model, "likelihood"):
            self.graph_model.likelihood.eval()  # same for likelihood

        # vectorized: build (b*L, d+1) in one shot, then call posterior once for all candidates and tasks
        # layout: rows 0..L-1 = candidate 0 × tasks 0..L-1, rows L..2L-1 = candidate 1, ...
        task_col = torch.arange(L, dtype=X.dtype, device=X.device).view(1, L, 1).expand(b, L, 1)  # (b, L, 1) task indices
        X_exp    = X.unsqueeze(1).expand(b, L, d)          # (b, L, d): repeat each candidate L times
        X_stack  = torch.cat([X_exp, task_col], dim=-1)    # (b, L, d+1): append task index column
        X_flat   = X_stack.reshape(b * L, d + 1)           # (b*L, d+1): flatten for SingleTaskGP

        post      = self.graph_model.posterior(X_flat, observation_noise=observation_noise)  # GP posterior at all (candidate, task) pairs
        mean_flat = post.mean.view(b * L)                   # (b*L,): flatten posterior mean
        cov_flat  = post.covariance_matrix                  # (b*L, b*L): full joint covariance
        if hasattr(cov_flat, "evaluate"):
            cov_flat = cov_flat.evaluate()                  # materialize lazy tensor to dense

        # aggregate: mu_S[i] = w^T mean[i*L:(i+1)*L]
        mean_tasks = mean_flat.view(b, L)                  # (b, L): reshape to per-candidate, per-task

        # extract only the per-candidate L×L diagonal blocks from the (b*L, b*L) covariance
        # var_S[i] = w^T Cov(Y(x_i,·), Y(x_i,·)) w — cross-candidate blocks not needed for EI
        cov_blocks = cov_flat.view(b, L, b, L)             # (b, L, b, L): expose block structure
        cov_diag   = cov_blocks[torch.arange(b), :, torch.arange(b), :]  # (b, L, L): one L×L block per candidate

        # un-standardize per task if GP was fitted on standardized targets
        if self.task_mean is not None:
            mean_tasks  = mean_tasks * self.task_std.unsqueeze(0) + self.task_mean.unsqueeze(0)  # (b, L) back to original scale
            sigma_outer = self.task_std[:, None] * self.task_std[None, :]   # (L, L) outer product of stds
            cov_diag    = cov_diag * sigma_outer.unsqueeze(0)               # (b, L, L) scale covariance back

        mus   = (mean_tasks * self.w.unsqueeze(0)).sum(-1)                  # (b,) scalar mean S(x) = w^T μ per candidate
        vars_ = torch.einsum("i,bij,j->b", self.w, cov_diag, self.w)       # (b,) scalar variance w^T Σ w per candidate
        vars_ = torch.clamp(vars_, min=0.0)                                 # numerical safety: variance must be non-negative

        mus   = mus.view(b, 1)       # (b, 1): required shape for MultivariateNormal
        vars_ = vars_.view(b, 1, 1)  # (b, 1, 1): covariance matrix shape for scalar output
        mvn   = MultivariateNormal(mus, vars_)     # scalar Gaussian per candidate
        return GPyTorchPosterior(mvn)              # BoTorch-compatible posterior object


class GraphKroneckerGP(GPyTorchModel, ExactGP):
    """
    Multi-output GP with graph Laplacian task kernel using Kronecker structure.

        K((θ, t), (θ', t')) = K_data(θ, θ') ⊗ K_graph[t, t']

    where K_graph = exp(-τL) or (I + γL)^{-1}.

    Uses MultitaskKernel (same as ICM) but replaces its internal IndexKernel
    with the graph task kernel so GPyTorch's Kronecker prediction strategies
    work correctly → O(N^3 + L^3) scaling.

    Inherits from botorch's GPyTorchModel mixin so it exposes `.posterior()`
    and `.num_outputs` like ICM's MultiTaskGP — this lets it be used directly
    with ScalarizedPosteriorTransform / make_aggregation_transform for EI.

    No ScaleKernel on the data kernel and no botorch Standardize outcome_transform
    (mirrors ICMMultiOutputGP and build_graph_multitask_gp): an outputscale on
    the data kernel would be redundant with both the task kernel's own scale
    parameter (tau/gamma/kappa) and the per-task noise in MultitaskGaussianLikelihood,
    creating a flat, non-identifiable direction in the likelihood that slows
    L-BFGS convergence. Per-task standardization is instead done by hand
    (task_mean/task_std buffers) and undone manually in posterior().

    Parameters
    ----------
    train_X : torch.Tensor [N, d]  — normalized OD inputs
    train_Y : torch.Tensor [N, L]  — per-task targets, already per-task
        standardized by the caller (see initialize_graph_kronecker_gp)
    likelihood : MultitaskGaussianLikelihood
    graph_task_kernel : LearnableTauGraphTaskKernel, LearnableGammaGraphTaskKernel,
        or GraphMaternTaskKernel
    task_mean, task_std : torch.Tensor [L] or None — per-task standardization
        constants used to un-standardize the posterior in posterior().
    """

    def __init__(self, train_X, train_Y, likelihood, graph_task_kernel, task_mean=None, task_std=None):
        self.num_tasks   = train_Y.shape[-1]  # L = number of tasks/edges
        self._num_outputs = self.num_tasks    # required by GPyTorchModel mixin for .posterior()

        super().__init__(train_X, train_Y, likelihood)  # initialize ExactGP with data and likelihood

        if task_mean is not None and task_std is not None:
            self.register_buffer("task_mean", task_mean.view(-1))  # (L,) per-task mean for un-standardization
            self.register_buffer("task_std",  task_std.view(-1))   # (L,) per-task std for un-standardization
        else:
            self.task_mean = None  # no standardization
            self.task_std  = None

        # per-task constant mean: one learnable constant c_l per edge, matching ICMMultiOutputGP's convention
        # per-task standardization already centers targets near 0, so c_l ≈ 0 at convergence
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=self.num_tasks)

        # bare Matérn kernel, no ScaleKernel — mirrors ICMMultiOutputGP's data_kernel
        data_kernel = MaternKernel(
            nu=2.5,                                        # twice-differentiable smoothness
            lengthscale_constraint=Interval(0.005, 20.0), # valid lengthscale range
        )

        # MultitaskKernel handles Kronecker prediction strategies correctly
        # initialize with rank=1 (creates a dummy IndexKernel), then replace with graph task kernel
        # MultitaskKernel.forward() calls task_covar_module(task_idx, task_idx) which our kernels accept
        self.covar_module = MultitaskKernel(
            data_kernel,
            num_tasks=self.num_tasks,
            rank=1,  # dummy rank — only used to initialize IndexKernel, which we replace below
        )
        self.covar_module.task_covar_module = graph_task_kernel  # replace IndexKernel with graph kernel

    def forward(self, x):
        mean_x  = self.mean_module(x)   # prior mean: [b, L] one constant per task per candidate
        covar_x = self.covar_module(x)  # prior covariance: Kronecker product K_data ⊗ K_graph
        return MultitaskMultivariateNormal(mean_x, covar_x)  # joint GP distribution over all tasks

    def transform_inputs(self, X=None, **kwargs):
        return X  # no-op: input normalization is done before calling the model

    def posterior(self, X, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        post = super().posterior(X, observation_noise=observation_noise, **kwargs)  # get standardized posterior
        if self.task_mean is None:
            return post  # no standardization was applied — return as-is

        # un-standardize: the GP predicted in standardized space, convert back to original scale
        mvn   = post.distribution
        mean  = mvn.mean             # [..., n, L] predicted mean in standardized space
        covar = mvn.covariance_matrix  # [..., n*L, n*L] covariance in standardized space
        if hasattr(covar, "evaluate"):
            covar = covar.evaluate() # materialize lazy tensor

        ts = self.task_std   # (L,) per-task standard deviations
        tm = self.task_mean  # (L,) per-task means
        mean_un = mean * ts + tm  # un-standardize mean: μ_orig = μ_std * σ + mean

        n           = mean.shape[-2]       # number of test points
        L_          = mean.shape[-1]       # number of tasks
        batch_shape = covar.shape[:-2]     # any leading batch dimensions
        covar_r     = covar.reshape(*batch_shape, n, L_, n, L_)  # expose (test_point, task) block structure
        sigma_outer = (ts[:, None] * ts[None, :]).view(1, L_, 1, L_)   # (1, L, 1, L): broadcast over n dims
        covar_un    = (covar_r * sigma_outer).reshape(*batch_shape, n * L_, n * L_)  # Cov(aX, bY) = ab*Cov(X,Y)

        mvn_un = MultitaskMultivariateNormal(mean_un, covar_un, interleaved=True)  # un-standardized joint distribution
        return GPyTorchPosterior(mvn_un)  # BoTorch-compatible posterior in original scale


def randomize_graph_kronecker_hyperparams(model: GraphKroneckerGP) -> None:
    """Randomize GraphKroneckerGP hyperparameters for multistart restarts."""
    with torch.no_grad():  # no gradients needed — just setting values
        device = model.train_inputs[0].device  # get device from training data
        dtype  = model.train_inputs[0].dtype   # get dtype from training data

        # randomize lengthscale of the bare Matérn data kernel
        data_kernel = model.covar_module.data_covar_module  # the Matérn kernel (no ScaleKernel wrapper)
        c  = data_kernel.raw_lengthscale_constraint         # box constraint on lengthscale
        lb, ub = float(c.lower_bound), float(c.upper_bound)
        data_kernel.lengthscale = _uniform(
            data_kernel.lengthscale.shape, lb, ub, device, dtype  # draw new lengthscale in [lb, ub]
        )

        # randomize per-task noise levels
        if hasattr(model.likelihood, "task_noises"):
            c  = getattr(model.likelihood, "raw_task_noises_constraint", None)
            lb = float(c.lower_bound) if c is not None else 1e-6  # fallback lower bound
            ub = float(c.upper_bound) if c is not None else 1.0   # fallback upper bound
            model.likelihood.task_noises = _uniform(
                model.likelihood.task_noises.shape, lb, ub, device, dtype  # draw new per-task noises
            )

        # randomize graph kernel scalar parameter depending on kernel type
        task_kern = model.covar_module.task_covar_module
        if hasattr(task_kern, "raw_tau"):    # diffusion kernel: randomize tau
            t_new = _uniform((), LearnableTauGraphTaskKernel.TAU_LB, LearnableTauGraphTaskKernel.TAU_UB * 0.5, device, dtype)
            t01   = ((t_new - LearnableTauGraphTaskKernel.TAU_LB) / (LearnableTauGraphTaskKernel.TAU_UB - LearnableTauGraphTaskKernel.TAU_LB)).clamp(1e-6, 1 - 1e-6)
            task_kern.raw_tau.copy_(torch.log(t01 / (1.0 - t01)))  # set in unconstrained sigmoid space
        if hasattr(task_kern, "raw_gamma"):  # regularized Laplacian kernel: randomize gamma
            g_new = _uniform((), 0.1, 20.0, device, dtype)
            task_kern.raw_gamma.copy_(_softplus_inv(g_new))  # set in unconstrained softplus space
        if hasattr(task_kern, "raw_nu"):     # graph Matérn: randomize smoothness ν
            nu_new = _uniform((), 0.5, 3.0, device, dtype)
            task_kern.raw_nu.copy_(_softplus_inv(nu_new))
        if hasattr(task_kern, "raw_kappa"):  # graph Matérn: randomize length-scale κ
            k_new = _uniform((), 0.1, 5.0, device, dtype)
            task_kern.raw_kappa.copy_(_softplus_inv(k_new))


def fit_graph_kronecker_multistart(
    mll: ExactMarginalLogLikelihood,
    n_restarts: int = 5,
    verbose: bool = True,
) -> None:
    """Multi-start L-BFGS MLL fitting for GraphKroneckerGP."""
    model          = mll.model                           # extract the GP model
    original_state = copy.deepcopy(model.state_dict())  # save initial state to reset between restarts
    best_state     = None                               # holds the best state dict found so far
    best_mll       = -math.inf                          # tracks the best MLL value seen

    for r in range(n_restarts):
        # keep_transforms=False: plain nn.Module state-dict load
        # GPyTorchModel.load_state_dict (keep_transforms=True) assumes train_targets has shape [m, n]
        # but GraphKroneckerGP stores train_targets as [n, m] (MultitaskMultivariateNormal convention)
        # same model instance and data throughout, so a plain reload is correct and simpler
        model.load_state_dict(original_state, keep_transforms=False)
        if r > 0:
            randomize_graph_kronecker_hyperparams(model)  # randomize for restarts 2+ to escape local optima

        model.train()  # training mode: enables gradient computation
        mll.train()    # same for MLL

        try:
            fit_gpytorch_mll(mll)  # run L-BFGS to maximize MLL
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"[restart {r+1}] L-BFGS failed ({str(e)[:60]}), skipping.")
            continue  # skip this restart

        model.train()  # re-enter train mode to evaluate MLL
        mll.train()    # same for MLL

        with torch.no_grad():  # no gradients needed for MLL evaluation
            try:
                output  = model(*model.train_inputs)               # forward pass on training data
                mll_val = mll(output, model.train_targets).item()  # compute MLL as a scalar
            except Exception:
                mll_val = -math.inf  # failed evaluation → treat as worst possible

        if verbose:
            print(f"[restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")

        if mll_val > best_mll:                         # new best restart found
            best_mll   = mll_val                       # update best MLL
            best_state = copy.deepcopy(model.state_dict())  # save this restart's state

    if best_state is not None:
        model.load_state_dict(best_state, keep_transforms=False)  # restore best hyperparameters
    else:
        model.load_state_dict(original_state, keep_transforms=False)  # all failed: fall back to original init

    model.eval()  # switch to prediction mode
    mll.eval()    # same for MLL

    if verbose:
        print(f"→ best MLL={best_mll:.4f}")


def initialize_graph_kronecker_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    edge_ids: List[str],
    adjacency_type: str = "shared_node_weighted",
    net_xml_path: Optional[str] = None,
    routes_csv_path: Optional[str] = None,
    edge_groups: Optional[Dict[str, List[str]]] = None,
    graph_kernel_type: str = "diffusion",
    normalized_laplacian: bool = True,
    ignore_internal: bool = True,
    standardize: bool = True,
    device: Optional[torch.device] = None,
    Lap_evals: Optional[torch.Tensor] = None,  # pass precomputed eigenpairs to skip adjacency build
    Lap_evecs: Optional[torch.Tensor] = None,
) -> Tuple:
    """
    Initialize a GraphKroneckerGP.

    Parameters
    ----------
    train_X : [N, d]  normalized OD inputs
    train_Y : [N, L]  per-task targets (raw, unstandardized)
    standardize : if True, standardize train_Y per task BY HAND (mean/std
        computed per column, unbiased=False) before fitting -- mirrors
        build_graph_multitask_gp's manual standardization, not botorch's
        Standardize outcome_transform. GraphKroneckerGP.posterior() undoes
        this manually using the stored task_mean/task_std buffers.
    Lap_evals / Lap_evecs : precomputed Laplacian eigenpairs (optional).
        Pass these to skip the adjacency build + eigendecomposition when the
        road network is fixed across BO iterations.

    Returns
    -------
    model : GraphKroneckerGP
    mll   : ExactMarginalLogLikelihood
    """
    if device is None:
        device = train_X.device  # infer device from training data

    train_X = train_X.to(device)  # move to target device
    train_Y = train_Y.to(device)  # move to target device

    if train_X.dim() != 2 or train_Y.dim() != 2:
        raise ValueError("Expected train_X [N, d] and train_Y [N, L].")  # enforce 2D tensors
    if train_X.shape[0] != train_Y.shape[0]:
        raise ValueError("train_X and train_Y must have the same number of rows.")  # N must match

    L = train_Y.shape[-1]  # number of tasks/edges

    if Lap_evals is None or Lap_evecs is None:
        # eigenpairs not cached — build adjacency and decompose
        if adjacency_type == "shared_node_weighted":
            A = adjacency_by_shared_node_weighted(
                net_xml_path, edge_ids,
                ignore_internal=ignore_internal,
                device=device, dtype=torch.float64,
            )
        elif adjacency_type == "route_overlap_weighted":
            A = adjacency_route_overlap_weighted(
                routes_csv_path, edge_ids,
                edge_groups=edge_groups,
                device=device, dtype=torch.float64,
            )
        else:
            raise ValueError(f"Unknown adjacency_type: {adjacency_type}")

        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)  # eigendecompose the Laplacian
    else:
        Lap_evals = Lap_evals.to(device=device, dtype=torch.float64)  # move precomputed eigenpairs to device
        Lap_evecs = Lap_evecs.to(device=device, dtype=torch.float64)

    # build the graph task kernel from the eigenpairs
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
    elif graph_kernel_type == "matern":
        graph_task_kernel = GraphMaternTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            nu_init=1.5, kappa_init=1.0, dtype=torch.float64, device=device,
        )
    else:
        raise ValueError(f"Unknown graph_kernel_type: {graph_kernel_type}. Use 'diffusion', 'regularized_laplacian', or 'matern'.")

    # per-task noise: matches ICMMultiOutputGP's convention
    # noise floor 1e-4 (vs 1e-6) guards against per-task noise collapsing to ~0 when N < L (early BO)
    likelihood = MultitaskGaussianLikelihood(
        num_tasks=L,
        has_global_noise=False,              # no shared global noise
        has_task_noise=True,                 # per-task noise σ²_l for each edge
        noise_constraint=Interval(1e-4, 1),  # same bound as ICM
    ).to(device)

    if standardize:
        task_mean    = train_Y.mean(dim=0)                               # (L,) per-task mean
        task_std     = train_Y.std(dim=0, unbiased=False).clamp_min(1e-6)  # (L,) per-task std, clamped to avoid divide-by-zero
        train_Y_fit  = (train_Y - task_mean) / task_std                  # standardized targets
    else:
        task_mean   = None   # no standardization
        task_std    = None
        train_Y_fit = train_Y

    model = GraphKroneckerGP(
        train_X=train_X,
        train_Y=train_Y_fit,           # standardized targets (or raw if standardize=False)
        likelihood=likelihood,
        graph_task_kernel=graph_task_kernel,
        task_mean=task_mean,           # stored for un-standardization in posterior()
        task_std=task_std,
    ).to(device)

    mll = ExactMarginalLogLikelihood(likelihood, model)  # MLL objective for hyperparameter optimization
    return model, mll  # return model for predictions, mll for fitting


def run_graph_kronecker_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    edge_ids: List[str],
    adjacency_type: str = "shared_node_weighted",
    net_xml_path: Optional[str] = None,
    routes_csv_path: Optional[str] = None,
    edge_groups: Optional[Dict[str, List[str]]] = None,
    graph_kernel_type: str = "diffusion",
    normalized_laplacian: bool = True,
    ignore_internal: bool = True,
    standardize: bool = True,
    device: Optional[torch.device] = None,
    Lap_evals: Optional[torch.Tensor] = None,  # precomputed Laplacian eigenpairs (optional)
    Lap_evecs: Optional[torch.Tensor] = None,
    n_gp_restarts: int = 5,
) -> Tuple[GraphKroneckerGP, ExactMarginalLogLikelihood]:
    """
    Build a GraphKroneckerGP (per-task noise via MultitaskGaussianLikelihood
    with has_global_noise=False) and fit it with multi-start L-BFGS.

    Drop-in per-task-noise counterpart to run_graph_multitask_gp. The returned
    model is a botorch Model (GPyTorchModel + ExactGP) with num_outputs=L,
    so it can be used directly with make_aggregation_transform /
    ScalarizedPosteriorTransform + ExpectedImprovement, exactly like ICM's
    MultiTaskGP.

    Returns
    -------
    model : GraphKroneckerGP
    mll   : ExactMarginalLogLikelihood
    """
    # initialize the Kronecker graph GP
    model, mll = initialize_graph_kronecker_gp(
        train_X=train_X,
        train_Y=train_Y,
        edge_ids=edge_ids,
        adjacency_type=adjacency_type,
        net_xml_path=net_xml_path,
        routes_csv_path=routes_csv_path,
        edge_groups=edge_groups,
        graph_kernel_type=graph_kernel_type,
        normalized_laplacian=normalized_laplacian,
        ignore_internal=ignore_internal,
        standardize=standardize,
        device=device,
        Lap_evals=Lap_evals,
        Lap_evecs=Lap_evecs,
    )

    fit_graph_kronecker_multistart(mll, n_restarts=n_gp_restarts)  # fit hyperparameters via multi-start L-BFGS
    return model, mll  # return fitted model and MLL
























# ============================================================================
# Shared + private multitask GP.
#
#   K((x,l),(x',m)) = delta_lm * private_l(x,x')  +  rho^2 * k_shared(x,x') * B[l,m]
#
# private_l = ScaleKernel(MaternKernel) per task l, identical construction
#   (kernel family, lengthscale_constraint) to IndGP's own per-link kernel
#   (src/models/gp_models.py:set_covar_module) -- each task keeps its own
#   lengthscale theta_l and outputscale sigma_l^2. This is the key difference
#   from the existing product-kernel GraphGP above (build_graph_multitask_gp),
#   whose single x_kernel forces one lengthscale across all L tasks.

# k_shared = one bare MaternKernel with its own lengthscale phi, separate
#   from every theta_l. No outputscale of its own -- rho^2 carries the
#   overall shared amplitude so there is no redundant scale x rho^2 pair.

# B: fixed (not learned), L x L, unit-diagonal correlation matrix from route
#   overlap (see build_route_overlap_correlation_matrix). 
# 
# As rho^2 -> 0, the
#   off-diagonal terms vanish and every diagonal block approaches
#   private_l -- L independent GPs. Paired with PerTaskGaussianLikelihood
#   (its own noise per task, matching IndGP's L separate GaussianLikelihoods
#   instead of this file's usual single global noise) and a fixed ZeroMean
#   (matching IndGP's own ZeroMean under standardization -- see
#   build_shared_private_graph_gp), the covariance, noise, and mean all nest
#   IndGP's model exactly in that limit. Not literally *at* rho^2=0, though:
#   rho2_constraint = Interval(1e-6, 20.0) keeps rho^2 strictly positive
#   (see SharedPrivateGraphKernel.__init__ for why 1e-6 rather than 0 --
#   GPyTorch's sigmoid-based Interval.inverse_transform(0.0) is -inf, and
#   floating-point noise near the lower bound can produce NaN besides). In
#   practice this floor is negligible: gradients of the constrained rho^2
#   w.r.t. its unconstrained raw parameter are already ~1e-15 there, so the
#   model is not distinguishable from the true rho^2=0 limit for any
#   practical purpose.
# ============================================================================

from gpytorch.likelihoods.noise_models import Noise                    # base class for custom noise models
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase  # lets us plug in a custom noise_covar


def build_route_overlap_correlation_matrix(
    routes_csv_path: str,
    edge_ids: List[str],
    edge_groups: Optional[Dict[str, List[str]]] = None,
    kind: str = "gram",           # "gram" (default) or "matern_spectral"
    nu_g: float = 1.5,
    kappa_g: float = 1.0,
    normalized_laplacian: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build the fixed L x L unit-diagonal task-correlation matrix B used by
    SharedPrivateGraphKernel's shared term.

    kind="gram" (default): the route-incidence Gram matrix M @ M.T from
        adjacency_route_overlap_weighted (zero_diagonal=False), correlation-
        normalized to unit diagonal. Already PSD by construction (a Gram
        matrix of route-incidence probability profiles), so this needs zero
        extra hyperparameters beyond rho^2 -- the simplest, most identifiable
        choice, and the recommended default at N=20-70 evaluations.

    kind="matern_spectral": reuse GraphMaternTaskKernel's spectral filter on
        top of the (zero-diagonal) route-overlap adjacency at *fixed*
        nu_g/kappa_g, then correlation-normalize. More flexible (lets
        correlation propagate transitively through the graph, not just
        between links with direct route incidence overlap) but not fit
        jointly with the BO loop here -- a documented fallback to try if the
        plain Gram matrix underperforms, not the default.
    """
    if kind == "gram":
        M_gram = adjacency_route_overlap_weighted(
            routes_csv_path, edge_ids, edge_groups=edge_groups,
            device=device, dtype=dtype, zero_diagonal=False,
        )
        K = M_gram


    elif kind == "matern_spectral":
        A = adjacency_route_overlap_weighted(
            routes_csv_path, edge_ids, edge_groups=edge_groups,
            device=device, dtype=dtype, zero_diagonal=True,
        )

        Lap_evals, Lap_evecs = laplacian_eigh(A, normalized=normalized_laplacian)
        task_kernel = GraphMaternTaskKernel(
            Lap_evals=Lap_evals, Lap_evecs=Lap_evecs,
            nu_init=nu_g, kappa_init=kappa_g, dtype=dtype, device=device,
        )
        with torch.no_grad():
            K = task_kernel._full_matrix()


    else:
        raise ValueError(f"Unknown kind: {kind}. Use 'gram' or 'matern_spectral'.")

    diag = torch.diagonal(K).clamp_min(1e-8)   # avoid divide-by-zero for a link with no route incidence at all

    inv_sqrt = diag.rsqrt() #  1-dimensional list of 3 numbers, 

    B = K * inv_sqrt.unsqueeze(0) * inv_sqrt.unsqueeze(1) # convert the K matri to a correlation matrix 

    B = 0.5 * (B + B.T)               # symmetrize away floating-point asymmetry
    B.diagonal().fill_(1.0)           # exact unit diagonal 

    
    return B


class TaskIndexNoise(Noise):
    """
    Per-task homoskedastic noise. Reads the task index from the last column
    of the raw input x (same (x, task_idx) stacking convention used
    throughout this file) and looks up that row's own noise scalar.

    With one free noise parameter per task, this exactly matches the L
    independent GaussianLikelihood(noise_constraint=Interval(1e-4,1))
    objects IndGP fits per link (src/models/gp_models.py), instead of the
    single global GaussianLikelihood used by build_graph_multitask_gp.
    """

    def __init__(self, num_tasks: int, noise_constraint=None,
                 dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        super().__init__()
        if noise_constraint is None:
            noise_constraint = Interval(1e-4, 1)
        self.num_tasks = num_tasks
        self.register_parameter("raw_noise", torch.nn.Parameter(torch.zeros(num_tasks, dtype=dtype, device=device)))
        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self) -> torch.Tensor:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(self, *params, shape=None, **kwargs) -> DiagLinearOperator:
        if "noise" in kwargs:
            return DiagLinearOperator(kwargs["noise"])
        x = params[0]
        if not torch.is_tensor(x):
            x = x[0]
        task_idx = x[..., -1].round().long()
        noise_diag = self.noise[task_idx]
        return DiagLinearOperator(noise_diag)


class PerTaskGaussianLikelihood(_GaussianLikelihoodBase):
    """
    GaussianLikelihood whose noise is looked up per-row from the task index
    in the last input column, via TaskIndexNoise -- see that class for why
    this (rather than this file's usual single global-noise
    GaussianLikelihood) is needed for SharedPrivateGraphKernel's noise
    structure to nest IndGP's as rho^2 -> 0 (see the module-level docstring
    above for the rho^2 -> 0 vs. rho^2 = 0 caveat).
    """

    def __init__(self, num_tasks: int, noise_constraint=None,
                 dtype: torch.dtype = torch.float64, device: Optional[torch.device] = None):
        noise_covar = TaskIndexNoise(num_tasks, noise_constraint=noise_constraint, dtype=dtype, device=device)
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.noise = value


class SharedPrivateGraphKernel(Kernel):
    """Additive shared+private multitask covariance -- see module-level docstring above."""

    is_stationary = False

    def __init__(
        self,
        L: int,
        input_dim: int,
        B: torch.Tensor,                    # (L, L) fixed unit-diagonal correlation matrix
        matern_nu: float = 2.5,
        rho2_init: float = 0.3,
        lengthscale_constraint=None,
        rho2_constraint=None,
        rho2_prior=None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        if lengthscale_constraint is None:
            lengthscale_constraint = Interval(0.005, 20.0)

        self.register_buffer("B", B.to(dtype=dtype, device=device))

        # private: one ScaleKernel(Matern) per task -- identical construction
        # to set_covar_module("matern-2p5") used by IndGP.
        self.private_kernels = torch.nn.ModuleList([
            ScaleKernel(MaternKernel(nu=matern_nu, lengthscale_constraint=lengthscale_constraint)).to(
                dtype=dtype, device=device)
            for _ in range(L)
        ])

        # shared: one bare Matern with its own lengthscale phi.
        self.shared_kernel = MaternKernel(nu=matern_nu, lengthscale_constraint=lengthscale_constraint).to(
            dtype=dtype, device=device)

        # rho^2 is bounded (Interval, like every other scale-type parameter in
        # this file) rather than a bare softplus. Since B has unit diagonal
        # and targets are standardized to ~unit per-task variance, rho^2 IS
        # the shared term's own contribution to a task's marginal variance --
        # left unconstrained above, an early smoke test on real grid_2d_3x3
        # data converged to rho^2~48 with private outputscales collapsed to
        # ~0, i.e. the exact oversharing failure mode this model exists to
        # avoid, just moved into the shared term instead of a rank-1 ICM
        # latent factor.
        #
        # The Interval upper bound alone turned out to be a hard wall, not a
        # safety margin: on real data rho^2 pinned at whatever bound was set
        # (4, then 20, then 100 in a diagnostic sweep), never settling at an
        # interior value -- a genuine MLE degeneracy (almost certainly the
        # diagonal confound: B_ll=1 means rho^2 inflates marginal variance
        # with nothing in the likelihood capping it) rather than real
        # correlation strength saturating a generous bound. A hard cap can
        # only hide that, not fix it. So in addition to the Interval (kept as
        # an outer numerical safety net, wide enough to rarely bind), rho^2
        # gets a half-Normal *prior* -- growing it costs marginal likelihood
        # via ExactMarginalLogLikelihood's registered-prior term, so the
        # fitted value becomes a genuine data-driven trade-off instead of
        # "as large as whatever wall happens to be in place."
        #
        # Lower bound is 1e-6, not 0: Interval.inverse_transform(0.0) is -inf
        # under GPyTorch's sigmoid-based transform (verified directly), which
        # would break gradient-based fitting if rho2 were ever set to literal
        # 0 (e.g. via randomize_shared_private_hyperparams's uniform draw).
        # Pushing the floor even lower than 1e-6 doesn't help either: the
        # gradient of constrained rho^2 w.r.t. its raw parameter is already
        # ~1e-15 at 1e-6 (i.e. already gradient-starved, L-BFGS was never
        # going to reach meaningfully lower on its own), while floats that
        # close to a boundary risk hitting the same -inf/NaN failure via
        # ordinary floating-point rounding, not just literal 0. So the
        # "exactly at rho^2=0" nesting claim should really be read as "as
        # rho^2 -> 0" -- see the module docstring above.
        if rho2_constraint is None:
            rho2_constraint = Interval(1e-6, 20.0)
        init = torch.tensor(
            min(max(float(rho2_init), 1e-6), float(rho2_constraint.upper_bound) - 1e-6),
            dtype=dtype, device=device,
        )
        self.register_parameter("raw_rho2", torch.nn.Parameter(rho2_constraint.inverse_transform(init)))
        self.register_constraint("raw_rho2", rho2_constraint)

        if rho2_prior is None:
            rho2_prior = HalfNormalPrior(scale=torch.tensor(1.0, dtype=dtype, device=device))
        self.register_prior(
            "rho2_prior", rho2_prior,
            lambda m: m.rho2,
            lambda m, v: m._set_rho2(v),
        )

    @property
    def rho2(self) -> torch.Tensor:
        return self.raw_rho2_constraint.transform(self.raw_rho2)

    @rho2.setter
    def rho2(self, value) -> None:
        self._set_rho2(value)

    def _set_rho2(self, value) -> None:
        # separate from the property setter (rather than folding into it) to
        # match gpytorch's own convention (e.g. _HomoskedasticNoiseBase's
        # _set_noise) for the setting_closure a register_prior call needs.
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho2)
        self.initialize(raw_rho2=self.raw_rho2_constraint.inverse_transform(value))

    def _private_block(self, x1d: Tensor, t1: Tensor, x2d: Tensor, t2: Tensor) -> Tensor:
        # Only compute the (much smaller) per-task submatrices actually needed --
        # the private term is exactly zero off the task-matched blocks, so
        # looping per task and scattering avoids an O(L * n1 * n2) blow-up.
        n1, n2 = x1d.shape[-2], x2d.shape[-2]
        K = x1d.new_zeros(n1, n2)
        for ell in range(self.L):
            i1 = torch.nonzero(t1 == ell, as_tuple=True)[0]
            i2 = torch.nonzero(t2 == ell, as_tuple=True)[0]
            if i1.numel() == 0 or i2.numel() == 0:
                continue
            sub = self.private_kernels[ell](x1d[i1], x2d[i2])
            sub = sub.to_dense() if hasattr(sub, "to_dense") else sub.evaluate()
            K[i1.unsqueeze(-1), i2.unsqueeze(-2)] = sub
        return K

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        d = self.input_dim
        x1d, t1 = x1[..., :d], x1[..., d].round().long()
        x2d, t2 = x2[..., :d], x2[..., d].round().long()

        if diag:
            n = x1d.shape[-2]
            priv = x1d.new_zeros(n)
            for ell in range(self.L):
                m = (t1 == ell) & (t2 == ell)
                if m.any():
                    priv[m] = self.private_kernels[ell](x1d[m], x2d[m], diag=True)
            shared_diag = self.shared_kernel(x1d, x2d, diag=True)
            if hasattr(shared_diag, "to_dense"):
                shared_diag = shared_diag.to_dense()
            b_diag = self.B[t1, t2]
            return priv + self.rho2 * shared_diag * b_diag

        priv = self._private_block(x1d, t1, x2d, t2)
        shared = self.shared_kernel(x1d, x2d)
        shared = shared.to_dense() if hasattr(shared, "to_dense") else shared.evaluate()
        b_block = self.B[t1.unsqueeze(-1), t2.unsqueeze(-2)]
        K = priv + self.rho2 * shared * b_block
        return DenseLinearOperator(K)


def randomize_shared_private_hyperparams(model: SingleTaskGP) -> None:
    """Randomize SharedPrivateGraphKernel hyperparameters for multistart restarts."""
    device = model.train_inputs[0].device
    dtype = model.train_inputs[0].dtype
    kern = model.covar_module

    with torch.no_grad():
        # private per-task kernels: randomize each task's own lengthscale/outputscale
        for pk in kern.private_kernels:
            c = pk.base_kernel.raw_lengthscale_constraint
            l_lb = float(c.lower_bound) if math.isfinite(float(c.lower_bound)) else 0.005
            l_ub = float(c.upper_bound) if math.isfinite(float(c.upper_bound)) else 20.0
            pk.base_kernel.lengthscale = _uniform(pk.base_kernel.lengthscale.shape, l_lb, l_ub, device, dtype)
            oc = pk.raw_outputscale_constraint
            o_lb = float(oc.lower_bound) if math.isfinite(float(oc.lower_bound)) else 0.05
            o_ub = float(oc.upper_bound) if math.isfinite(float(oc.upper_bound)) else 2.0
            pk.outputscale = _uniform(pk.outputscale.shape, o_lb, o_ub, device, dtype)

        # shared kernel: randomize phi
        c = kern.shared_kernel.raw_lengthscale_constraint
        l_lb = float(c.lower_bound) if math.isfinite(float(c.lower_bound)) else 0.005
        l_ub = float(c.upper_bound) if math.isfinite(float(c.upper_bound)) else 20.0
        kern.shared_kernel.lengthscale = _uniform(kern.shared_kernel.lengthscale.shape, l_lb, l_ub, device, dtype)

        # rho^2: draw restart-init values from the kernel's own registered
        # rho2_prior (HalfNormalPrior(1.0) by default) rather than uniformly
        # over the full constrained range. A uniform draw over [1e-6, 20]
        # gives ~0.5% probability of landing below 0.1 -- restarts almost
        # never actually start in the "sharing doesn't help" regime the
        # prior itself says is quite plausible (HalfNormal(1) puts ~8% mass
        # below 0.1, ~68% below 1). Sampling from the prior directly closes
        # that gap and stays correct even if a custom rho2_prior is passed
        # to SharedPrivateGraphKernel.__init__ instead of the default.
        rc = kern.raw_rho2_constraint
        r_lb = float(rc.lower_bound) if math.isfinite(float(rc.lower_bound)) else 1e-6
        r_ub = float(rc.upper_bound) if math.isfinite(float(rc.upper_bound)) else 20.0
        rho2_sample = kern.rho2_prior.sample(())
        kern.rho2 = rho2_sample.clamp(min=r_lb, max=r_ub)

        # per-task noise: randomize each task's own noise within its constraint
        nc = model.likelihood.noise_covar.raw_noise_constraint
        n_lb = float(nc.lower_bound) if math.isfinite(float(nc.lower_bound)) else 1e-6
        n_ub = float(nc.upper_bound) if math.isfinite(float(nc.upper_bound)) else 0.1
        model.likelihood.noise = _uniform(model.likelihood.noise.shape, n_lb, n_ub, device, dtype)


def fit_shared_private_mll_multistart(
    mll: ExactMarginalLogLikelihood,
    n_restarts: int = 5,
    verbose: bool = False,
) -> str:
    """
    Multi-start MLL fitting for SharedPrivateGraphKernel -- mirrors
    fit_graph_mll_multistart's structure (smart init on restart 0, randomized
    thereafter, keep the highest-MLL state, Adam fallback if every L-BFGS
    restart fails) but checks this model's own parameters for the
    degenerate-solution guard instead of the product-kernel's kernels[0].
    """
    model = mll.model
    original_state = copy.deepcopy(model.state_dict())
    best_state = None
    best_mll_val = -math.inf

    for r in range(n_restarts):
        model.load_state_dict(original_state)
        randomize_shared_private_hyperparams(model)

        model.train()
        mll.train()
        try:
            fit_gpytorch_mll(mll)
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"  [shared-private restart {r+1}/{n_restarts}] LBFGS failed ({str(e)[:50]}), skipping.")
            continue
        model.train()
        mll.train()
        with torch.no_grad():
            try:
                out = model(*model.train_inputs)
                # PerTaskGaussianLikelihood's noise depends on x (task index in the
                # last column), so -- unlike this file's other single-global-noise
                # graph GPs -- the raw train_inputs must be forwarded here too,
                # matching how botorch's fit_gpytorch_mll internally calls mll().
                mll_val = mll(out, model.train_targets, *model.train_inputs).item()
            except Exception:
                mll_val = -math.inf

        # reject degenerate solutions:
        #  - every private outputscale collapsed to ~0 with no shared signal
        #    either (nothing explains the data), or
        #  - rho^2 pinned at (or essentially at) its own ceiling *while*
        #    private outputscale has also collapsed: the specific oversharing
        #    signature actually observed on real data (rho2~48 unconstrained,
        #    then rho2 pinned at whatever ceiling was set -- see the rho^2
        #    constraint docstring in SharedPrivateGraphKernel.__init__).
        #
        # An earlier version of this guard also rejected noise>0.08 and
        # rho2>0.05-with-collapsed-private outright. Both turned out to fire
        # on essentially every restart at small N (e.g. N=5 real init data):
        # with only 5 points to fit 24 separate private kernels from, private
        # outputscale collapsing toward 0 and noise rising toward its ceiling
        # is often the *correct* MLE answer (there is genuinely not enough
        # data yet to support 24 private functions), not oversharing -- and
        # rho2~0.6-0.9 is nowhere near its own ceiling. Rejecting those
        # legitimate small-N fits was what forced adam-fallback on ~every
        # iteration of the first real run instead of the intended occasional
        # fallback. Both checks now specifically target ceiling-pinning
        # rather than "noise/rho2 is merely elevated."
        noise_val = model.likelihood.noise.max().item()
        noise_ub = float(model.likelihood.noise_covar.raw_noise_constraint.upper_bound)
        priv_os_vals = [float(pk.outputscale.max()) for pk in model.covar_module.private_kernels]
        max_priv_os = max(priv_os_vals)
        mean_priv_os = sum(priv_os_vals) / len(priv_os_vals)
        rho2_val = float(model.covar_module.rho2)
        rho2_ub = float(model.covar_module.raw_rho2_constraint.upper_bound)
        degenerate_empty = max_priv_os < 1e-4 and rho2_val < 1e-4
        degenerate_oversharing = mean_priv_os < 1e-3 and rho2_val > 0.98 * rho2_ub
        degenerate_all_noise = noise_val > 0.98 * noise_ub and mean_priv_os < 1e-3 and rho2_val < 0.98 * rho2_ub
        if degenerate_empty or degenerate_oversharing or degenerate_all_noise:
            if verbose:
                print(f"  [shared-private restart {r+1}/{n_restarts}] degenerate "
                      f"(noise={noise_val:.3f}, mean_priv_os={mean_priv_os:.2e}, rho2={rho2_val:.3f}), skipping.")
            continue

        if verbose:
            print(f"  [shared-private restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")

        if mll_val > best_mll_val:
            best_mll_val = mll_val
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        mll.eval()
        if verbose:
            print(f"  → best MLL={best_mll_val:.4f}")
        return "lbfgs-multistart"

    model.load_state_dict(original_state)
    model.train()
    mll.train()
    opt = torch.optim.Adam(mll.parameters(), lr=1e-2)
    stored_X = model.train_inputs[0]
    for _ in range(200):
        opt.zero_grad()
        try:
            # same param-forwarding requirement as the L-BFGS branch above --
            # PerTaskGaussianLikelihood needs x to look up per-task noise.
            loss = -mll(model(stored_X), model.train_targets, stored_X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=1.0)
            opt.step()
        except NotPSDError:
            continue
    model.eval()
    mll.eval()
    return "adam-fallback"


def build_shared_private_graph_gp(
    edge_ids: List[str],
    train_X: torch.Tensor,          # (N, d+1): first d cols = OD inputs, last col = task_id
    train_Y: torch.Tensor,          # (N, 1): scalar target per row
    input_dim: int,
    routes_csv_path: str,
    edge_groups: Optional[Dict[str, List[str]]] = None,
    shared_kernel_type: str = "gram",     # "gram" (default) or "matern_spectral"
    matern_nu: float = 2.5,
    rho2_init: float = 1.0,
    device: Optional[torch.device] = None,
    standardize: bool = True,
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict]:
    """
    Build the shared+private graph GP (see SharedPrivateGraphKernel). Mirrors
    build_graph_multitask_gp's I/O and per-task standardization convention so
    it drops into the same BO-loop call sites.
    """
    if device is None:
        device = train_X.device
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    if train_Y.ndim == 1:
        train_Y = train_Y.view(-1, 1)

    L = len(edge_ids)
    B = build_route_overlap_correlation_matrix(
        routes_csv_path, edge_ids, edge_groups=edge_groups,
        kind=shared_kernel_type, device=device, dtype=torch.float64,
    )

    covar_module = SharedPrivateGraphKernel(
        L=L, input_dim=input_dim, B=B, matern_nu=matern_nu, rho2_init=rho2_init,
        dtype=torch.float64, device=device,
    )
    likelihood = PerTaskGaussianLikelihood(num_tasks=L, dtype=torch.float64, device=device)

    task_mean = None
    task_std = None
    if standardize:
        task_ids = train_X[:, input_dim].long()
        y = train_Y.squeeze(-1)
        task_mean = torch.zeros(L, dtype=y.dtype, device=y.device)
        task_std = torch.ones(L, dtype=y.dtype, device=y.device)
        for t in range(L):
            mask = (task_ids == t) & (~torch.isnan(y))
            if mask.any():
                task_mean[t] = y[mask].mean()
                task_std[t] = y[mask].std(unbiased=False).clamp_min(1e-6)
        y_std = (y - task_mean[task_ids]) / task_std[task_ids]
        train_Y = y_std.unsqueeze(-1)

    model = SingleTaskGP(
        train_X=train_X, train_Y=train_Y,
        covar_module=covar_module, likelihood=likelihood,
        # fixed ZeroMean (not a learned ConstantMean): per-task standardization
        # already exactly zero-centers each task's targets at its own training
        # points, so a learned constant is redundant and -- because SingleTaskGP
        # would otherwise default to ONE shared ConstantMean across all L tasks,
        # unlike IndGP's L independently-fit constants -- an unconditional,
        # ungated sharing channel outside the rho^2/B structure. ZeroMean makes
        # both models' mean identically 0 for every task, so the "exactly nests
        # IndGP as rho^2 -> 0" claim holds for the mean too, not just the
        # covariance/noise. See IndGP's matching change in
        # src/models/gp_models.py (initialize_independent_gp_models_with_modellist).
        mean_module=gpytorch.means.ZeroMean(),
        outcome_transform=None,
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if standardize:
        model.register_buffer("task_mean", task_mean.view(-1))
        model.register_buffer("task_std", task_std.view(-1))

    model = model.to(device)
    mll = mll.to(device)

    info = {
        "L_tasks": L,
        "shared_kernel_type": shared_kernel_type,
        "B_offdiag_max": float(B.masked_select(~torch.eye(L, dtype=torch.bool, device=device)).max()),
    }
    return model, mll, info


def run_shared_private_graph_gp(
    edge_ids: List[str],
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    input_dim: int,
    routes_csv_path: str,
    edge_groups: Optional[Dict[str, List[str]]] = None,
    shared_kernel_type: str = "gram",
    matern_nu: float = 2.5,
    rho2_init: float = 1.0,
    device: Optional[torch.device] = None,
    standardize: bool = True,
    n_gp_restarts: int = 5,
) -> Tuple[SingleTaskGP, ExactMarginalLogLikelihood, Dict, str]:
    """Build and fit the shared+private graph GP. Returns (model, mll, info, fit_method)."""
    if device is None:
        device = train_X.device
    model, mll, info = build_shared_private_graph_gp(
        edge_ids=edge_ids, train_X=train_X, train_Y=train_Y, input_dim=input_dim,
        routes_csv_path=routes_csv_path, edge_groups=edge_groups,
        shared_kernel_type=shared_kernel_type, matern_nu=matern_nu, rho2_init=rho2_init,
        device=device, standardize=standardize,
    )
    fit_method = fit_shared_private_mll_multistart(mll, n_restarts=n_gp_restarts)
    return model, mll, info, fit_method
