from abc import abstractmethod

import beartype.typing as tp
from cola.annotations import PSD
from cola.linalg.algorithm_base import Algorithm
from cola.linalg.decompositions.decompositions import Cholesky
from cola.linalg.inverse.inv import solve
from cola.ops.operators import I_like
from flax import nnx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import (
    Float,
    Num,
)

from gpjax.dataset import Dataset
from gpjax.distributions import GaussianDistribution
from gpjax.kernels import RFF
from gpjax.kernels.base import AbstractKernel
from gpjax.likelihoods import (
    AbstractLikelihood,
    Gaussian,
    NonGaussian,
)
from gpjax.lower_cholesky import lower_cholesky
from gpjax.mean_functions import AbstractMeanFunction
from gpjax.parameters import (
    Parameter,
    Real,
    Static,
)
from gpjax.typing import (
    Array,
    FunctionalSample,
    KeyArray,
)
from gpjax.gps import Prior
import beartype.typing as tp
import jax.random as jr
from jaxtyping import Float

from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import BasisFunctionComputation
from gpjax.kernels.stationary.base import StationaryKernel
from gpjax.parameters import Static
from gpjax.typing import (
    Array,
    KeyArray,
)

K = tp.TypeVar("K", bound=AbstractKernel)
M = tp.TypeVar("M", bound=AbstractMeanFunction)
L = tp.TypeVar("L", bound=AbstractLikelihood)
NGL = tp.TypeVar("NGL", bound=NonGaussian)
GL = tp.TypeVar("GL", bound=Gaussian)


def adj_sample_approx(posterior,
        num_samples: int,
        train_data: Dataset,
        key: KeyArray,
        num_features: int | None = 100,
        solver_algorithm: tp.Optional[Algorithm] = Cholesky(),
) -> FunctionalSample:
    r"""Draw approximate samples from the Gaussian process posterior.

    Build an approximate sample from the Gaussian process posterior. This method
    provides a function that returns the evaluations of a sample across any given
    inputs.

    Unlike when building approximate samples from a Gaussian process prior, decompositions
    based on Fourier features alone rarely give accurate samples. Therefore, we must also
    include an additional set of features (known as canonical features) to better model the
    transition from Gaussian process prior to Gaussian process posterior. For more details
    see [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309).

    In particular, we approximate the Gaussian processes' posterior as the finite
    feature approximation
    $\hat{f}(x) = \sum_{i=1}^m \phi_i(x)\theta_i + \sum{j=1}^N v_jk(.,x_j)$
    where $\phi_i$ are m features sampled from the Fourier feature decomposition of
    the model's kernel and $k(., x_j)$ are N canonical features. The Fourier
    weights $\theta_i$ are samples from a unit Gaussian. See
    [Wilson et. al. (2020)](https://arxiv.org/abs/2002.09309) for expressions
    for the canonical weights $v_j$.

    A key property of such functional samples is that the same sample draw is
    evaluated for all queries. Consistency is a property that is prohibitively costly
    to ensure when sampling exactly from the GP prior, as the cost of exact sampling
    scales cubically with the size of the sample. In contrast, finite feature representations
    can be evaluated with constant cost regardless of the required number of queries.

    Args:
        num_samples (int): The desired number of samples.
        key (KeyArray): The random seed used for the sample(s).
        num_features (int): The number of features used when approximating the
            kernel.
        solver_algorithm (Optional[Algorithm], optional): The algorithm to use for the solves of
            the inverse of the covariance matrix. See the
            [CoLA documentation](https://cola.readthedocs.io/en/latest/package/cola.linalg.html#algorithms)
            for which solver to pick. For PSD matrices, CoLA currently recommends Cholesky() for small
            matrices and CG() for larger matrices. Select Auto() to let CoLA decide. Defaults to Cholesky().

    Returns:
        FunctionalSample: A function representing an approximate sample from the Gaussian
        process prior.
    """
    if (not isinstance(num_samples, int)) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # sample fourier features
    fourier_feature_fn = _build_fourier_features_fn(posterior.prior, num_features, key)

    # sample fourier weights
    fourier_weights = jr.normal(key, [num_samples, 2 * num_features])  # [B, L]

    # sample weights v for canonical features
    # v = Σ⁻¹ (y + ε - ɸ⍵) for  Σ = Kxx + Io² and ε ᯈ N(0, o²)
    obs_var = posterior.likelihood.obs_stddev.value ** 2
    Kxx = posterior.prior.kernel.gram(train_data.X)  # [N, N]
    Sigma = Kxx + I_like(Kxx) * (obs_var + posterior.jitter)  # [N, N]
    eps = jnp.sqrt(obs_var) * jr.normal(key, [train_data.n, num_samples])  # [N, B]
    y = train_data.y - posterior.prior.mean_function(train_data.X)  # account for mean
    Phi = fourier_feature_fn(train_data.X)
    canonical_weights = solve(
        Sigma,
        y + eps - jnp.inner(Phi, fourier_weights),
        solver_algorithm,
    )  # [N, B]

    def sample_fn(test_inputs: Float[Array, "n D"]) -> Float[Array, "n B"]:
        fourier_features = fourier_feature_fn(test_inputs)  # [n, L]
        weight_space_contribution = jnp.inner(
            fourier_features, fourier_weights
        )  # [n, B]
        canonical_features = posterior.prior.kernel.cross_covariance(
            test_inputs, train_data.X
        )  # [n, N]
        function_space_contribution = jnp.matmul(
            canonical_features, canonical_weights
        )

        return (
                posterior.prior.mean_function(test_inputs)
                + weight_space_contribution
                + function_space_contribution
        )

    return sample_fn


def _build_fourier_features_fn(
    prior: Prior, num_features: int, key: KeyArray
) -> tp.Callable[[Float[Array, "N D"]], Float[Array, "N L"]]:
    r"""Return a function that evaluates features sampled from the Fourier feature
    decomposition of the prior's kernel.

    Args:
        prior (Prior): The Prior distribution.
        num_features (int): The number of feature functions to be sampled.
        key (KeyArray): The random seed used.

    Returns
    -------
        Callable: A callable function evaluation the sampled feature functions.
    """
    if (not isinstance(num_features, int)) or num_features <= 0:
        raise ValueError("num_features must be a positive integer")

    # Approximate kernel with feature decomposition
    approximate_kernel = MO_RFF(kernel0=prior.kernel.kernel0, kernel1=prior.kernel.kernel1, num_basis_fns=num_features, key=key)

    def eval_fourier_features(test_inputs: Float[Array, "N D"]) -> Float[Array, "N L"]:
        Phi = approximate_kernel.compute_features(x=test_inputs)
        return Phi

    return eval_fourier_features


class MO_RFF(AbstractKernel):
    r"""Computes an approximation of the kernel using Random Fourier Features.

    All stationary kernels are equivalent to the Fourier transform of a probability
    distribution. We call the corresponding distribution the spectral density. Using
    a finite number of basis functions, we can compute the spectral density using a
    Monte-Carlo approximation. This is done by sampling from the spectral density and
    computing the Fourier transform of the samples. The kernel is then approximated by
    the inner product of the Fourier transform of the samples with the Fourier
    transform of the data.

    The key reference for this implementation is the following papers:
    - 'Random Features for Large-Scale Kernel Machines' by Rahimi and Recht (2008).
    - 'On the Error of Random Fourier Features' by Sutherland and Schneider (2015).
    """

    compute_engine: BasisFunctionComputation

    def __init__(
        self,
        kernel0: StationaryKernel,
        kernel1: StationaryKernel,
        num_basis_fns: int = 50,
        frequencies: tp.Union[Float[Array, "M D"], None] = None,
        compute_engine: BasisFunctionComputation = BasisFunctionComputation(),
        key: KeyArray = jr.PRNGKey(0),
    ):
        r"""Initialise the RFF kernel.

        Args:
            base_kernel (StationaryKernel): The base kernel to be approximated.
            num_basis_fns (int): The number of basis functions to use in the approximation.
            frequencies (Float[Array, "M D"] | None): The frequencies to use in the approximation.
                If None, the frequencies are sampled from the spectral density of the base
                kernel.
            compute_engine (BasisFunctionComputation): The computation engine to use for
                the basis function computation.
            key (KeyArray): The random key to use for sampling the frequencies.
        """
        self._check_valid_base_kernel(kernel0)
        self._check_valid_base_kernel(kernel1)
        self.kernel0 = kernel0
        self.kernel1 = kernel1
        self.num_basis_fns = num_basis_fns
        self.frequencies = frequencies
        self.compute_engine = compute_engine

        if self.frequencies is None:
            n_dims = self.kernel0.n_dims
            if n_dims is None:
                raise ValueError(
                    "Expected the number of dimensions to be specified for the base kernel. "
                    "Please specify the n_dims argument for the base kernel."
                )

            self.frequencies0 = Static(
                self.kernel0.spectral_density.sample(
                    seed=key, sample_shape=(self.num_basis_fns, n_dims)
                )
            )
            self.frequencies1 = Static(
                self.kernel1.spectral_density.sample(
                    seed=key, sample_shape=(self.num_basis_fns, n_dims)
                )
            )
        self.name = f"{self.kernel0.name} (RFF)"

    def __call__(self, x: Float[Array, "D 1"], y: Float[Array, "D 1"]) -> None:
        """Superfluous for RFFs."""
        raise RuntimeError("RFFs do not have a kernel function.")

    @staticmethod
    def _check_valid_base_kernel(kernel: AbstractKernel):
        r"""Verify that the base kernel is valid for RFF approximation.

        Args:
            kernel (AbstractKernel): The kernel to be checked.
        """
        if not isinstance(kernel, StationaryKernel):
            raise TypeError("RFF can only be applied to stationary kernels.")

        # check that the kernel has a spectral density
        _ = kernel.spectral_density

    def compute_features(self, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            x: A $N \times D$ array of inputs.

        Returns:
            Float[Array, "N L"]: A $N \times L$ array of features where $L = 2M$.
        """
        z = jnp.array(x[:, -1], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2)
        k1_switch = z

        part_1 = jnp.expand_dims(k0_switch, axis=-1) * self.compute_features_otherfile(self.frequencies0, self.kernel0, x[:, :2]) * jnp.sqrt(self.kernel0.variance.value / self.num_basis_fns)
        part_2 = jnp.expand_dims(k1_switch, axis=-1) * self.compute_features_otherfile(self.frequencies1, self.kernel1, x[:, :2]) * jnp.sqrt(self.kernel1.variance.value / self.num_basis_fns)
        return part_1 + part_2
        # TODO unsure the above is the most correct
        # return self.compute_engine.compute_features(self, x)

    def compute_features_otherfile(self, kernel_frequencies, kernel: K, x: Float[Array, "N D"]) -> Float[Array, "N L"]:
        r"""Compute the features for the inputs.

        Args:
            kernel: the kernel function.
            x: the inputs to the kernel function of shape `(N, D)`.

        Returns:
            A matrix of shape $N \times L$ representing the random fourier features where $L = 2M$.
        """
        frequencies = kernel_frequencies.value
        scaling_factor = kernel.lengthscale.value
        z = jnp.matmul(x, (frequencies / scaling_factor).T)
        z = jnp.concatenate([jnp.cos(z), jnp.sin(z)], axis=-1)
        return z