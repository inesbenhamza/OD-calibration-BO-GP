"""
ICMMultiOutputGP is a pure gpytorch.ExactGP — no outcome_transform, no Standardize.
Input normalization and output standardization are handled outside the model.
"""

import torch
import gpytorch

from gpytorch.models import ExactGP
from gpytorch.means import MultitaskMean, ConstantMean
from gpytorch.kernels import MaternKernel, MultitaskKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.model import Model
from botorch.fit import fit_gpytorch_mll


def fit_gpytorch_mll_with_adam_fallback(
    mll: ExactMarginalLogLikelihood,
    max_adam_iters: int = 200,
    lr: float = 0.1,
    verbose: bool = True,
) -> str:
    """
    Fit the GP marginal log likelihood with fit_gpytorch_mll (typically L-BFGS-B).
    If that fails, fall back to Adam.

    Returns
    -------
    str
        'lbfgs' if L-BFGS-B succeeded, 'adam' if the fallback was used.
    """
    try:
        fit_gpytorch_mll(mll)
        if verbose:
            print("GP fit successful (fit_gpytorch_mll)")
        return "lbfgs"
    except Exception as e:
        if verbose:
            print(f"fit_gpytorch_mll failed ({e}), falling back to Adam ({max_adam_iters} steps)")

        mll.train()
        model = mll.model
        optimizer = torch.optim.Adam(mll.parameters(), lr=lr)

        for _ in range(max_adam_iters):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

        mll.eval()
        if verbose:
            print("Adam fallback finished")
        return "adam"


#########################
####################################
########################


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
        super().__init__(train_X, train_Y, likelihood)

        self.num_tasks = train_Y.shape[-1]

        # One learnable constant mean per task through MultitaskMean wrapping ConstantMean
        self.mean_module = MultitaskMean(
            ConstantMean(),
            num_tasks=self.num_tasks,
        )

        # Shared input kernel over x
        data_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=train_X.shape[-1],
            lengthscale_constraint=Interval(0.005, 4.0),
        )

        # ICM multitask kernel = data kernel x task kernel
        self.covar_module = MultitaskKernel(
            data_kernel,
            num_tasks=self.num_tasks,
            rank=rank,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X=None, **kwargs):
        """
        BoTorch may call model.transform_inputs(X=...).
        Since normalization is handled outside, return X unchanged.
        """
        return X


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
        raise ValueError("Expected train_X_norm [N, d] and train_Y [N, L].")

    N, _ = train_X_norm.shape
    N2, L = train_Y.shape
    if N != N2:
        raise ValueError(f"Mismatch in number of rows: X has {N}, Y has {N2}.")

    likelihood = MultitaskGaussianLikelihood(
        num_tasks=L,
        noise_constraint=Interval(1e-4, 1.0),
    ) 

    model = ICMMultiOutputGP(
        train_X=train_X_norm,
        train_Y=train_Y,
        likelihood=likelihood,
        rank=rank,
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def select_icm_rank(
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    ranks: list = [1, 2, 3],
    verbose: bool = True,
) -> int:
    """
    Select the ICM task covariance rank by marginal log-likelihood.

    For each candidate rank, fits an ICMMultiOutputGP and evaluates the MLL
    on the training data. The rank with the highest MLL is returned.

    Parameters
    ----------
    train_X_norm : torch.Tensor
        Shape [N, d], normalized inputs.
    train_Y : torch.Tensor
        Shape [N, L], multi-task targets.
    ranks : list of int
        Candidate ranks to evaluate.
    verbose : bool
        Print MLL scores per rank.

    Returns
    -------
    int
        Best rank.
    """
    best_rank = ranks[0]
    best_score = float("-inf")

    for rank in ranks:
        model, mll = initialize_icm_gp(train_X_norm, train_Y, rank=rank, verbose=False)
        model = model.to(dtype=train_X_norm.dtype)
        mll = mll.to(dtype=train_X_norm.dtype)
        fit_gpytorch_mll_with_adam_fallback(mll, verbose=False)
        model.eval()
        with torch.no_grad():
            output = model(train_X_norm)
            score = mll(output, model.train_targets).item()
        if verbose:
            print(f"  rank={rank}: MLL={score:.4f}")
        if score > best_score:
            best_score = score
            best_rank = rank

    if verbose:
        print(f"  → selected rank={best_rank} (MLL={best_score:.4f})")
    return best_rank


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
        super().__init__()
        self.base_model = base_icm_model

        w = torch.as_tensor(
            weights,
            dtype=base_icm_model.train_targets.dtype, # same as base model targets for consistency, which is the num_tasks dimension of the multitask GP
            device=base_icm_model.train_targets.device, # same device as base model targets for consistency
        ).view(-1)

        if w.numel() != base_icm_model.num_tasks:
            raise ValueError(
                f"weights has length {w.numel()}, but base model has "
                f"{base_icm_model.num_tasks} tasks."
            )

        self.register_buffer("w", w)

    @property
    def num_outputs(self) -> int:
        return 1

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
            b, q, d = X.shape 
            if q != 1: # as we are not evlaluating multiple points in parallel as we are using EI and not a q-batch acquisition functions 
                # model compute the scalar aggregate points by points 
                raise RuntimeError("LinearAggregationICM supports q=1 only.")
            X = X.view(b, d) # b is the batch size (nb of training points), d is the input dimension (tasks)
        elif X.dim() == 2:
            b, d = X.shape
        else:
            raise RuntimeError(f"Unexpected X.dim()={X.dim()}, expected 2 or 3.")

        self.base_model.eval()
        self.base_model.likelihood.eval()

        w = self.w
        mus = []
        vars_ = []

        # Safer: compute one-point multitask posterior at a time
        # rather than slicing a large flattened covariance matrix.
        for i in range(b): # for each data point
            # base_model is the GP over L outputs jointly. So if we call it on one input gives a gaussian dist over the vector [e_1(x),...,e_L(x)] of the L outputs. Then we can compute the mean and covariance of S(x) = w^T e(x) using the mean and covariance of e(x) and the weights w.
            dist_i = self.base_model(X[i : i + 1])  # one x (candidate point passed into posterior by the acquisition function), multitask posterior
            if observation_noise:
                dist_i = self.base_model.likelihood(dist_i)


            #reminder GP gives a joint distribution over all tasks output simultaneously at the same input point 
            mean_i = dist_i.mean.squeeze(0)          # [L] mean of the L outputs at this point
            cov_i = dist_i.covariance_matrix         # [L, L] for one input point, covaariance between each task output at that point




            #K_** = K_data(x_i, x_i) ⊗ K_task    shape: [L, L]
            #  K_** = K_data(x_i, x_i) ⊗ K_task            shape: [L, N·L]




            #Diagonal = how uncertain the GP is about each task at x_i
             #Off-diagonal = how correlated two tasks are at x_i (learned by ICM from the task kernel)
            


            # aggregation
            mu_S = (mean_i * w).sum()
            var_S = w @ cov_i @ w # single scalar variance of the aggregate S(x) = w^T e(x)
            #variance of the weighted sum S = w_1·e_1 + w_2·e_2 + w_3·e_3, which is the standard formula for variance of a linear combination of correlated random variable

            if var_S.item() < -1e-12:
                raise RuntimeError(f"Negative scalar variance encountered: {var_S.item()}")

            var_S = torch.clamp(var_S, min=0.0) # no negative variance

            mus.append(mu_S)
            vars_.append(var_S)

        mus = torch.stack(mus).view(b, 1)         # batch of scalar means
        vars_ = torch.stack(vars_).view(b, 1, 1)  # batch of 1x1 covariances

        mvn = MultivariateNormal(mus, vars_)
        return GPyTorchPosterior(mvn)
