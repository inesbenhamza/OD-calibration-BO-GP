"""
ICMMultiOutputGP is a pure gpytorch.ExactGP — no outcome_transform, no Standardize.
Input normalization and output standardization are handled outside the model.
"""

import copy          # for deepcopy of model state dicts during multi-restart fitting
import math          # for math.inf used to initialize best_mll
import torch         # main tensor library
import gpytorch      # GP framework built on PyTorch

from gpytorch.models import ExactGP                          # base class for exact (non-sparse) GPs
from gpytorch.means import MultitaskMean, ConstantMean       # mean functions: one constant per task
from gpytorch.kernels import MaternKernel, MultitaskKernel   # input kernel and ICM multitask kernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood # Gaussian likelihood with per-task noise
from gpytorch.constraints import Interval                    # box constraint on hyperparameters
from gpytorch.mlls import ExactMarginalLogLikelihood         # objective for hyperparameter optimization
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal  # GP output distributions

from botorch.posteriors.gpytorch import GPyTorchPosterior  # BoTorch posterior wrapper, required by acquisition functions
from botorch.models.model import Model                     # BoTorch base model class
from botorch.fit import fit_gpytorch_mll                   # L-BFGS optimizer for MLL
from botorch.exceptions import ModelFittingError           # raised when L-BFGS fails to converge
from linear_operator.utils.errors import NotPSDError       # raised when covariance matrix is not positive semi-definite


def _uniform_tensor(shape, lb, ub, device, dtype):
    # create a tensor of given shape filled with uniform random values in [lb, ub]
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lb, ub)


def randomize_icm_hyperparams(model):
    """
    Randomize ICM hyperparameters inside valid constrained space.
    For ICMMultiOutputGP with:
      - MultitaskKernel
      - Matern data_covar_module
      - MultitaskGaussianLikelihood with one shared global noise (has_task_noise=False) or per-task noises (has_task_noise=True).
    """
    with torch.no_grad():  # no gradients needed — we're just setting values, not optimizing
        device = model.train_inputs[0].device  # get device (cpu/gpu) from training data
        dtype  = model.train_inputs[0].dtype   # get dtype (float32/float64) from training data

        # randomize shared input lengthscale within its constrained range
        data_kernel = model.covar_module.data_covar_module    # the Matérn kernel over inputs x
        c  = data_kernel.raw_lengthscale_constraint           # box constraint on lengthscale
        lb = float(c.lower_bound)                             # lower bound of constraint
        ub = float(c.upper_bound)                             # upper bound of constraint
        data_kernel.lengthscale = _uniform_tensor(
            data_kernel.lengthscale.shape, lb, ub, device, dtype  # draw new lengthscale uniformly in [lb, ub]
        )

        # randomize per-task noise levels σ²_l for each edge
        if hasattr(model.likelihood, "task_noises"):                              # check if model uses per-task noise
            c  = getattr(model.likelihood, "raw_task_noises_constraint", None)   # get constraint if it exists
            lb = float(c.lower_bound) if c is not None else 1e-6                 # fallback lower bound
            ub = float(c.upper_bound) if c is not None else 1.0                  # fallback upper bound
            model.likelihood.task_noises = _uniform_tensor(
                model.likelihood.task_noises.shape, lb, ub, device, dtype        # draw new per-task noises in [lb, ub]
            )

        # randomize global shared noise if present (only when has_global_noise=True)
        if hasattr(model.likelihood, "noise_covar"):                              # check if global noise exists
            c  = model.likelihood.noise_covar.raw_noise_constraint                # get constraint
            lb = float(c.lower_bound) if c is not None else 1e-6                 # fallback lower bound
            ub = float(c.upper_bound) if c is not None else 1.0                  # fallback upper bound
            model.likelihood.noise = _uniform_tensor(
                model.likelihood.noise.shape, lb, ub, device, dtype              # draw new global noise in [lb, ub]
            )

        # randomize ICM task covariance factor W (B = W Wᵀ + diag(κ))
        task_kernel = model.covar_module.task_covar_module         # the IndexKernel that stores B
        if hasattr(task_kernel, "covar_factor"):                   # covar_factor is GPyTorch's internal name for W
            task_kernel.covar_factor.normal_(mean=0.0, std=0.5)   # draw W entries from N(0, 0.5²)

        # randomize ICM diagonal task variance κ (the diag(κ) part of B)
        if hasattr(task_kernel, "var"):            # check if diagonal variance exists
            c  = task_kernel.raw_var_constraint    # get constraint on κ
            lb = max(float(c.lower_bound), 1e-4)  # ensure lower bound is at least 1e-4 for stability
            ub = float(c.upper_bound)              # upper bound of constraint
            if not math.isfinite(ub):              # if upper bound is inf, cap it at 1.0
                ub = 1.0
            task_kernel.var = _uniform_tensor(
                task_kernel.var.shape, lb, ub, device, dtype  # draw new diagonal variances in [lb, ub]
            )


def fit_icm_multistart(
    mll,
    n_restarts: int = 5,
    verbose: bool = True,
):
    """
    Multi-start L-BFGS MLL fitting for ICMMultiOutputGP.

    First start = current initialization.
    Other starts = randomized ICM hyperparameters.
    Keeps the state with the highest MLL.
    Falls back to Adam (200 steps, lr=1e-2) if every L-BFGS restart fails.
    Returns 'lbfgs-multistart' or 'adam-fallback'.
    """
    model = mll.model  # extract the GP model from the MLL object

    original_state = copy.deepcopy(model.state_dict())  # save initial hyperparameters to reset between restarts
    best_state     = None                               # will hold the state dict of the best restart found so far
    best_mll       = -math.inf                          # track the best MLL value seen across restarts

    for r in range(n_restarts):
        model.load_state_dict(original_state)  # reset to original init before each restart

        if r > 0:
            randomize_icm_hyperparams(model)  # for restarts 2+, randomize to escape local optima

        model.train()  # put model in training mode (enables gradients, disables prediction caching)
        mll.train()    # put MLL in training mode

        try:
            fit_gpytorch_mll(mll)  # run L-BFGS to maximize MLL w.r.t. all hyperparameters
        except (RuntimeError, NotPSDError, ModelFittingError) as e:
            if verbose:
                print(f"[restart {r+1}] L-BFGS failed ({str(e)[:60]}), skipping.")  # log the failure
            continue  # skip this restart and try the next one

        model.train()  # re-enter train mode to evaluate MLL (fit_gpytorch_mll leaves model in eval)
        mll.train()    # same for MLL

        with torch.no_grad():  # no gradients needed for MLL evaluation
            try:
                output  = model(*model.train_inputs)               # forward pass on training data
                mll_val = mll(output, model.train_targets).item()  # compute MLL value as a scalar
            except Exception:
                mll_val = -math.inf  # if evaluation fails, treat this restart as worst possible

        if verbose:
            print(f"[restart {r+1}/{n_restarts}] MLL={mll_val:.4f}")  # log MLL for this restart

        if mll_val > best_mll:                          # if this restart is better than all previous ones
            best_mll   = mll_val                        # update best MLL
            best_state = copy.deepcopy(model.state_dict())  # save this restart's hyperparameters

    if best_state is not None:                    # at least one L-BFGS restart succeeded
        model.load_state_dict(best_state)         # restore the best hyperparameters found
        model.eval()                              # switch to prediction mode
        mll.eval()                                # same for MLL
        if verbose:
            print(f"→ best MLL={best_mll:.4f}")  # log the best MLL value
        return "lbfgs-multistart"                 # signal that L-BFGS succeeded

    # all restarts failed — Adam fallback on original init
    if verbose:
        print("[fit_icm_multistart] all L-BFGS restarts failed, falling back to Adam.")
    model.load_state_dict(original_state)  # reset to original init for Adam
    model.train()                          # training mode for Adam
    mll.train()                            # same for MLL
    opt      = torch.optim.Adam(mll.parameters(), lr=1e-2)  # Adam optimizer over all hyperparameters
    stored_X = model.train_inputs[0]                        # cache training inputs to avoid repeated access
    for _ in range(200):               # run 200 gradient steps
        opt.zero_grad()                # clear gradients from previous step
        try:
            loss = -mll(model(stored_X), model.train_targets)              # negative MLL = loss to minimize
            loss.backward()                                                 # compute gradients
            torch.nn.utils.clip_grad_norm_(mll.parameters(), max_norm=1.0) # clip gradients to prevent exploding updates
            opt.step()                                                      # update hyperparameters
        except NotPSDError:
            continue  # skip this step if covariance matrix becomes non-PSD mid-optimization
    model.eval()  # switch to prediction mode after Adam
    mll.eval()    # same for MLL
    if verbose:
        print("→ Adam fallback complete.")
    return "adam-fallback"  # signal that Adam was used


class ICMMultiOutputGP(ExactGP):
    """
    Multi-output Exact GP using an ICM / MultitaskKernel construction.

    Parameters
    ----------
    train_X : torch.Tensor
        Shape [N, d], normalized inputs.
    train_Y : torch.Tensor
        Shape [N, L], multi-task targets.
    likelihood : MultitaskGaussianLikelihood
        GPyTorch multitask likelihood.
    rank : int
        Rank of the task covariance factorization.
    """

    def __init__(self, train_X, train_Y, likelihood, rank: int = 1):
        super().__init__(train_X, train_Y, likelihood)  # pass data and likelihood to ExactGP base class

        self.num_tasks = train_Y.shape[-1]  # L = number of edges/tasks, inferred from target shape

        # one learnable constant mean per task: m(x) = [c_1, ..., c_L], learned via MLL
        self.mean_module = MultitaskMean(
            ConstantMean(),           # base mean function: a single learnable scalar
            num_tasks=self.num_tasks, # replicated L times, one constant per task
        )

        # shared Matérn input kernel k(x, x') over OD matrices — same kernel for all edges
        data_kernel = MaternKernel(
            nu=2.5,                                       # smoothness: 2.5 = twice-differentiable functions
            lengthscale_constraint=Interval(0.005, 20.0), # keep lengthscale in a reasonable range
        )

        # ICM multitask kernel: K[(x,t),(x',t')] = k_data(x,x') * B[t,t']
        # B = W Wᵀ + diag(κ) captures inter-edge correlations (rank controls expressiveness)
        self.covar_module = MultitaskKernel(
            data_kernel,              # the shared input kernel
            num_tasks=self.num_tasks, # number of tasks/edges L
            rank=rank,                # rank of W in B = W Wᵀ + diag(κ)
        )

    def forward(self, x):  # defines the prior GP distribution p(f | x, θ)
        mean_x  = self.mean_module(x)   # compute prior mean: [b, L]
        covar_x = self.covar_module(x)  # compute prior covariance: [b*L, b*L] ICM kernel matrix
        return MultitaskMultivariateNormal(mean_x, covar_x)  # package into a joint GP distribution over all tasks

    def transform_inputs(self, X=None, **kwargs):
        """
        BoTorch may call model.transform_inputs(X=...).
        Since normalization is handled outside, return X unchanged.
        """
        return X  # no-op: input normalization is done before calling the model


def initialize_icm_gp(
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    verbose: bool = False,
    rank: int = 1,
):
    """
    Initialize an ICM multitask GP.

    Parameters
    ----------
    train_X_norm : torch.Tensor
        Shape [N, d], normalized inputs in [0, 1]^d.
    train_Y : torch.Tensor
        Shape [N, L], per-task targets.
    verbose : bool
        Unused placeholder for diagnostics.
    rank : int
        Rank of the task covariance factorization.

    Returns
    -------
    model : ICMMultiOutputGP
    mll : ExactMarginalLogLikelihood
    """
    if train_X_norm.dim() != 2 or train_Y.dim() != 2:
        raise ValueError("Expected train_X_norm [N, d] and train_Y [N, L].")  # enforce expected tensor shapes

    N, _  = train_X_norm.shape  # N = number of training points, _ = input dimension d
    N2, L = train_Y.shape       # N2 = same N (checked below), L = number of tasks/edges
    if N != N2:
        raise ValueError(f"Mismatch in number of rows: X has {N}, Y has {N2}.")  # X and Y must have same number of rows

    # per-edge Gaussian noise: each edge l gets its own noise σ²_l, no shared global noise
    likelihood = MultitaskGaussianLikelihood(
        num_tasks=L,                         # one noise parameter per edge
        has_global_noise=False,              # drops the shared scalar σ²I_L term
        has_task_noise=True,                 # full D = diag(σ²_1,…,σ²_L), each edge gets its own noise level
        noise_constraint=Interval(1e-4, 1),  # keep noise in a reasonable range, avoids numerical issues
    )

    # build the ICM GP: combines a shared input kernel k(x,x') with a learned task covariance B
    model = ICMMultiOutputGP(
        train_X=train_X_norm,  # normalized inputs [N, d]
        train_Y=train_Y,       # per-edge targets [N, L]
        likelihood=likelihood, # per-task Gaussian noise model
        rank=rank,             # rank of B = W Wᵀ + diag(κ), controls inter-task correlation complexity
    )

    # wire likelihood and model together so mll(output, targets) computes log p(y | X, θ)
    # this is what L-BFGS/Adam maximizes to learn all hyperparameters (lengthscales, B, noise, mean)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll  # return both: model for predictions, mll for fitting


def initialize_icm_gp_from_adjacency(
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    adjacency_tensor: torch.Tensor,
    rank: int = 1,
    verbose: bool = False,
):
    """
    ICM GP warm-started from a graph adjacency matrix.

    The task covariance B = W Wᵀ + diag(κ) is initialised so that B ≈ A:
      W  ← U[:, :rank] * sqrt(S[:rank])   (rank-R SVD of A)
      κ  ← diag(A - W Wᵀ).clamp(1e-4)    (residual diagonal, kept positive)

    MLL optimisation then refines W and κ freely — the graph is a warm start,
    not a constraint.  For rank=1 this encodes the dominant correlation mode of A.
    """
    model, mll = initialize_icm_gp(train_X_norm, train_Y, verbose=verbose, rank=rank)

    dtype  = train_X_norm.dtype
    device = train_X_norm.device
    A = adjacency_tensor.to(dtype=dtype, device=device)
    A = (A + A.T) / 2  # symmetrize (adjacency should already be symmetric)
    A = A.clamp(min=0)

    U, S, _ = torch.linalg.svd(A, full_matrices=False)
    W_init  = U[:, :rank] * S[:rank].sqrt().unsqueeze(0)  # [L, rank]

    with torch.no_grad():
        task_kernel = model.covar_module.task_covar_module
        task_kernel.covar_factor.data.copy_(W_init)
        B_approx    = W_init @ W_init.T
        diag_resid  = (torch.diag(A) - torch.diag(B_approx)).clamp_min(1e-4)
        if hasattr(task_kernel, "var"):
            task_kernel.var.data.copy_(diag_resid)

    return model, mll


class LinearAggregationICM(Model):
    """
    Wrap an ICMMultiOutputGP with L outputs and expose a scalar posterior:

        S(x) = sum_l w_l * e_l(x)

    where e(x) are the task outputs and w are fixed weights.

    This is intended for scalar acquisitions such as Expected Improvement.
    Supports q=1 only.
    """

    def __init__(
        self,
        base_icm_model: ICMMultiOutputGP,
        weights: torch.Tensor,
        normalize_weights: bool = False,
    ):
        super().__init__()                 # initialize BoTorch Model base class
        self.base_model = base_icm_model  # the fitted ICM GP with L task outputs

        # cast weights to the same dtype/device as the model to avoid type mismatches
        w = torch.as_tensor(
            weights,
            dtype=base_icm_model.train_targets.dtype,    # match model's float precision
            device=base_icm_model.train_targets.device,  # match model's device (cpu/gpu)
        ).view(-1)  # flatten to 1D: shape [L]

        if w.numel() != base_icm_model.num_tasks:  # weights must have exactly one entry per task
            raise ValueError(
                f"weights has length {w.numel()}, but base model has "
                f"{base_icm_model.num_tasks} tasks."
            )

        # register_buffer: saves w as part of model state (moves with .to(device), saved in checkpoints)
        # but not a trainable parameter — weights are fixed, not learned
        self.register_buffer("w", w)

        # read per-task standardization stats from the model if they were stored after fitting
        # needed to un-standardize predictions back to original error scale before aggregating
        if hasattr(base_icm_model, "task_mean") and hasattr(base_icm_model, "task_std"):
            self.register_buffer("task_mean", base_icm_model.task_mean.view(-1))  # [L] mean per task
            self.register_buffer("task_std",  base_icm_model.task_std.view(-1))   # [L] std per task
        else:
            self.task_mean = None  # no standardization was applied, predictions already in original scale
            self.task_std  = None  # same

    @property
    def num_outputs(self) -> int:
        return 1  # scalar output S(x) = w^T e(x), required by BoTorch acquisition functions

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: bool = False,
        **kwargs,
    ) -> GPyTorchPosterior:
        """
        Compute the scalar posterior for S(x) = w^T e(x).

        Parameters
        ----------
        X : torch.Tensor
            Shape [b, d] or [b, 1, d].
        observation_noise : bool
            Whether to include observation noise in the posterior.

        Returns
        -------
        GPyTorchPosterior
            Scalar posterior with one output.
        """
        if X.dim() == 3:
            b, q, d = X.shape  # b = batch of candidates, q = points per candidate, d = input dim
            if q != 1:         # only q=1 supported: we aggregate per candidate, not across a joint batch
                raise RuntimeError("LinearAggregationICM supports q=1 only.")
            X = X.view(b, d)   # squeeze out the q=1 dimension: [b, 1, d] → [b, d]
        elif X.dim() == 2:
            b, d = X.shape     # already [b, d], nothing to do
        else:
            raise RuntimeError(f"Unexpected X.dim()={X.dim()}, expected 2 or 3.")

        self.base_model.eval()            # switch to prediction mode (disables gradient tracking for params)
        self.base_model.likelihood.eval() # same for likelihood

        L = self.w.shape[0]  # number of tasks/edges

        # single batched forward pass: calls ICMMultiOutputGP.forward(X)
        # returns a MultitaskMultivariateNormal with mean [b, L] and cov [b*L, b*L]
        # GPyTorch uses task-last ordering: flat index i*L + l → candidate i, task l
        dist_all = self.base_model(X)
        if observation_noise:
            dist_all = self.base_model.likelihood(dist_all)  # adds per-task noise D=diag(σ²_1,...,σ²_L) → posterior predictive

        mean_all = dist_all.mean               # [b, L] — GP mean for each candidate and each task
        cov_full = dist_all.covariance_matrix  # [b*L, b*L] — full joint covariance (lazy tensor)
        if hasattr(cov_full, "evaluate"):
            cov_full = cov_full.evaluate()     # materialize lazy tensor to dense [b*L, b*L]

        # the full covariance [b*L, b*L] mixes candidates and tasks
        # reshape to [b, L, b, L] to separate them, then extract only the [b, L, L] diagonal blocks
        # block [i, :, i, :] = per-candidate task covariance for candidate i — needed for w^T Σ w
        cov_blocks = cov_full.view(b, L, b, L)        # reshape to expose block structure
        idx        = torch.arange(b, device=X.device) # indices [0, 1, ..., b-1]
        cov_diag   = cov_blocks[idx, :, idx, :]       # [b, L, L] — one L×L task cov per candidate

        # if the ICM was trained on standardized targets, GP predicts in standardized space
        # un-standardize: μ_orig = μ_std * σ + mean,  Σ_orig = Σ_std * (σ σ^T)
        if self.task_mean is not None:
            mean_all    = mean_all * self.task_std.unsqueeze(0) + self.task_mean.unsqueeze(0)  # [b, L] back to original scale
            sigma_outer = self.task_std[:, None] * self.task_std[None, :]   # [L, L] outer product of per-task stds
            cov_diag    = cov_diag * sigma_outer.unsqueeze(0)               # [b, L, L] scale covariance back

        w     = self.w
        mus   = (mean_all * w.unsqueeze(0)).sum(-1)          # [b] — scalar mean S(x) = w^T μ, one per candidate
        vars_ = torch.einsum("i,bij,j->b", w, cov_diag, w)  # [b] — scalar variance w^T Σ w, one per candidate
        vars_ = torch.clamp(vars_, min=0.0)                  # numerical safety: variance must be non-negative

        # wrap into MultivariateNormal then GPyTorchPosterior so BoTorch acquisition functions can consume it
        mvn = MultivariateNormal(mus.view(b, 1), vars_.view(b, 1, 1))  # scalar Gaussian per candidate
        return GPyTorchPosterior(mvn)  # BoTorch-compatible posterior object
