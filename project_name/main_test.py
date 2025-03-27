import gpjax
from fontTools.ttLib.tables.S__i_l_f import pass_attrs_info

# from project_name.models.gpfs_gp import MultiGpfsGp
from project_name.acq.acquisition import MultiBaxAcqFunction
from project_name.acq.acqoptimize import AcqOptimizer
from project_name.alg.algorithms import Algorithm
from project_name.util.misc_util import dict_to_namespace
from project_name.util.domain_util import unif_random_sample_domain, project_to_domain
import optax

import jax.numpy as jnp
import jax.random as jrandom
from jax import config
from typing import (
    List,
    Tuple,
)
import jax

# Enable Float64 for more stable matrix inversions.
from jax import config, hessian
from dataclasses import dataclass, field
import jax.numpy as jnp
import jax.random as jr
from jaxopt import ScipyBoundedMinimize
from jaxtyping import (
    Float,
    Int,
    install_import_hook,
)
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import optax as ox
import tensorflow_probability.substrates.jax as tfp
import sys
from gpjax.parameters import Static
from gpjax.typing import (
    Array,
    FunctionalSample,
    ScalarFloat,
)
import neatplot
from typing import NamedTuple
import numpy as np
from gpjax.kernels.computations import DenseKernelComputation
import pandas as pd
import time
from gpjax.kernels.stationary.utils import (
    build_student_t_distribution,
    euclidean_distance,
)
import tensorflow_probability.substrates.jax.distributions as tfd

config.update("jax_enable_x64", True)

key = jrandom.PRNGKey(42)

# Global variables
DEFAULT_F_IS_DIFF = True
LONG_PATH = True

cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]


class ExPath(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray


class NStep(Algorithm):
    """
    An algorithm that takes n steps through a state space (and touches n+1 states).
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "NStep")
        self.params.n = getattr(params, "n", 10)
        self.params.f_is_diff = getattr(params, "f_is_diff", DEFAULT_F_IS_DIFF)
        self.params.init_x = getattr(params, "init_x", [0.0, 0.0])
        self.params.project_to_domain = getattr(params, "project_to_domain", True)
        self.params.domain = getattr(params, "domain", [[0.0, 10.0], [0.0, 10.0]])

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        len_path = len(self.exe_path.x)
        if len_path == 0:
            next_x = self.params.init_x
        elif len_path >= self.params.n + 1:
            next_x = None
        else:
            if self.params.f_is_diff:
                zip_path_end = zip(self.exe_path.x[-1], self.exe_path.y[-1])
                next_x = [xi + yi for xi, yi in zip_path_end]
            else:
                next_x = self.exe_path.y[-1]

            if self.params.project_to_domain:
                # Optionally, project to domain
                next_x = project_to_domain(next_x, self.params.domain)

        return next_x

    def get_output(self):
        """Return output based on self.exe_path."""
        return self.exe_path


def step_northwest(x_list, step_size=0.5, f_is_diff=DEFAULT_F_IS_DIFF):
    """Return x_list with a small positive value added to each element."""
    if f_is_diff:
        diffs_list = [step_size for x in x_list]
        return diffs_list
    else:
        x_list_new = [x + step_size for x in x_list]
        return x_list_new


def plot_path_2d(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if true_path:
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=15, alpha=0.3)
    else:
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.5)


def plot_path_2d_jax(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path]
    y_plot = [xi[1] for xi in path]

    if true_path:
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=15)
    else:
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)


# -------------
# Start Script
# -------------
# Set black-box function
f = step_northwest

# Set domain
domain = [[0, 23], [0, 23]] if LONG_PATH else [[0, 10], [0, 10]]

# Set algorithm
algo_class = NStep
n_steps = 40 if LONG_PATH else 15
algo_params = {"n": n_steps, "init_x": [0.5, 0.5], "domain": domain}
algo = algo_class(algo_params)

# Set model
gp_params = {"ls": 8.0, "alpha": 5.0, "sigma": 1e-2, "n_dimx": 2}
multi_gp_params = {"n_dimy": 2, "gp_params": gp_params}
# gp_model_class = MultiGpfsGp

# # Set acqfunction  # TODO reinstate all this stuff
# acqfn_params = {"n_path": 30}
# acqfn_class = MultiBaxAcqFunction
# n_rand_acqopt = 1000
#
# Compute true path
true_algo = algo_class(algo_params)
true_path, _ = algo.run_algorithm_on_f(f)


@dataclass
class HelmholtzKernel(gpjax.kernels.stationary.StationaryKernel):
    # initialise Phi and Psi kernels as any stationary kernel in gpJax
    potential_kernel: gpjax.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpjax.kernels.RBF(active_dims=[0, 1])
    )
    stream_kernel: gpjax.kernels.stationary.StationaryKernel = field(
        default_factory=lambda: gpjax.kernels.RBF(active_dims=[0, 1])
    )
    compute_engine = DenseKernelComputation()

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]
    ) -> Float[Array, "1"]:
        # obtain indices for k_helm, implement in the correct sign between the derivatives
        z = jnp.array(X[2], dtype=int)
        zp = jnp.array(Xp[2], dtype=int)
        sign = (-1) ** (z + zp)

        # convert to array to correctly index, -ve sign due to exchange symmetry (only true for stationary kernels)
        potential_dvtve = -jnp.array(
            hessian(self.potential_kernel)(X, Xp), dtype=jnp.float64
        )[z][zp]
        stream_dvtve = -jnp.array(
            hessian(self.stream_kernel)(X, Xp), dtype=jnp.float64
        )[1 - z][1 - zp]

        return potential_dvtve + sign * stream_dvtve

    @property
    def spectral_density(self) -> tfp.distributions.Normal:  # TODO a dodge workaround for now
        return tfp.distributions.Normal(0.0, 1.0)


class VelocityKernel(gpjax.kernels.stationary.StationaryKernel):  #TODO changed this from abstract kernel
    def __init__(
        self,
        # kernel0: gpjax.kernels.AbstractKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),  # TODO the original with abstract kernels
        # kernel1: gpjax.kernels.AbstractKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
        kernel0: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
        kernel1: gpjax.kernels.stationary.StationaryKernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0),
    ):
        self.kernel0 = kernel0
        self.kernel1 = kernel1
        super().__init__(n_dims=3, compute_engine=DenseKernelComputation())

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[-1], dtype=int)
        zp = jnp.array(Xp[-1], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel0(X, Xp) + k1_switch * self.kernel1(X, Xp)

    # @property
    # def spectral_density(self) -> tfp.distributions.Normal:  # TODO this is kinda dodge and idk if it works really
    #     spectral0 = self.kernel0.spectral_density
    #     spectral1 = self.kernel1.spectral_density
    #
    #     # Equal weighting of the two kernels
    #     return spectral0, spectral1

    @property
    def spectral_density(self) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(0.0, 1.0)


def create_gp_model(output_dims):  # TODO can we vmap this ever?
    """Create multi-output GP models."""
    mean = gpjax.mean_functions.Zero()
    kernel = VelocityKernel() # TODO make this more general
    # kernel = HelmholtzKernel() # TODO make this more general
    # kernel = gpjax.kernels.RBF(active_dims=[0, 1], lengthscale=jnp.array([10.0, 8.0]), variance=25.0)
    prior = gpjax.gps.Prior(mean_function=mean, kernel=kernel)
    return prior

def adjust_dataset(x, y):
    # Change vectors x -> X = (x,z), and vectors y -> Y = (y,z) via the artificial z label
    def label_position(data):  # 2,20
        # introduce alternating z label
        n_points = len(data[0])
        label = jnp.tile(jnp.array([0.0, 1.0]), n_points)
        return jnp.vstack((jnp.repeat(data, repeats=2, axis=1), label)).T

    # change vectors y -> Y by reshaping the velocity measurements
    def stack_velocity(data):  # 2,20
        return data.T.flatten().reshape(-1, 1)

    def dataset_3d(pos, vel):
        return gpjax.Dataset(label_position(pos), stack_velocity(vel))

    # takes in dimension (number of data points, num features)

    return dataset_3d(jnp.swapaxes(x, 0, 1), jnp.swapaxes(y, 0, 1))

def sample_and_optimize_posterior(optimized_posterior, D, key, lower_bound, upper_bound, num_samples=1, num_initial_sample_points=1):
    """Sample from posteriors and optimize."""
    key, _key = jr.split(key)
    initial_sample_points = jr.uniform(_key, shape=(num_initial_sample_points, lower_bound.shape[0]), dtype=jnp.float64,
                                       minval=lower_bound, maxval=upper_bound)  # TODO can we batch this?
    D = adjust_dataset(D.X, D.y)

    def create_path(xs, ys):
        return ExPath(jnp.squeeze(xs, axis=-2), jnp.squeeze(ys, axis=-2))

    def outer_loop(init_x, key):
        key, _key = jrandom.split(key)
        sample_func = posterior.sample_approx(num_samples=1, train_data=D, key=_key, num_features=500)

        def _step_fn(runner_state, _):
            this_x, key = runner_state
            key, _key = jrandom.split(key)
            adj_x = adjust_dataset(this_x, jnp.ones((this_x.shape[0], 2)))
            y_tot_NO = jnp.swapaxes(sample_func(adj_x.X), 0, 1)
            # latent_dist = optimized_posterior.predict(adj_x.X, train_data=D)
            # y_tot_NO = latent_dist.mean().reshape(-1, 2)
            # y_tot_NO = latent_dist.sample(_key, (1,))

            next_x = jnp.clip(this_x + y_tot_NO, jnp.array([domain[0][0], domain[1][0]]),
                              jnp.array([domain[0][1], domain[1][1]]))

            return (next_x, key), (this_x, y_tot_NO)

        key, _key = jrandom.split(key)
        _, (all_xs, all_ys) = jax.lax.scan(_step_fn, (init_x, _key), None, length=40)

        exe_path = create_path(all_xs, all_ys)
        exe_path_new = adjust_dataset(exe_path.x, exe_path.y)

        comb_D = D + gpjax.Dataset(exe_path_new.X, exe_path_new.y)
        # TODO is this the error, when appending data what form should it be in?

        adj_sample_points = adjust_dataset(initial_sample_points, jnp.ones((initial_sample_points.shape[0], 2)))

        latent_dist = posterior.predict(adj_sample_points.X, train_data=comb_D)
        predictive_dist = posterior.likelihood(latent_dist)
        sample_mus = predictive_dist.mean()
        sample_stds = predictive_dist.stddev().reshape(2, -1)

        return all_xs, all_ys, exe_path, sample_mus, sample_stds

    # Create initial state for all samples
    init_x = jnp.array([[0.5, 0.5]])
    key, _key = jrandom.split(key)
    batch_key = jrandom.split(key, num_samples)
    all_xs, all_ys, exe_path, sample_mus, sample_stds = jax.vmap(outer_loop, in_axes=(None, 0))(init_x, batch_key)

    return exe_path, initial_sample_points, sample_mus, sample_stds

def optimise_sample(opt_posterior, D, initial_sample_points, sample_mus, sample_stds):
    # Grab the posterior mus and covariance for each GP
    D = adjust_dataset(D.X, D.y)
    adj_sample_points = adjust_dataset(initial_sample_points, jnp.ones((initial_sample_points.shape[0], 2)))
    latent_dist = opt_posterior.predict(adj_sample_points.X, train_data=D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean() # TODO should this be predictive or should it be just posterior, if latter then do we need the points?
    predictive_std = predictive_dist.stddev().reshape(2, -1)

    # # TODO can we vmap this?
    # def test_vmap(collected_data, exe_path, posterior, initial_sample_points):
    #     data = gpjax.Dataset(X=comb_x, y=comb_y)
    #     latent_dist = posterior.predict(initial_sample_points, train_data=data)
    #     predictive_dist = posterior.likelihood(latent_dist)
    #
    #     sample_mean = predictive_dist.mean()  # TODO should this be predictive or should it be just posterior, if latter then do we need the points?
    #     sample_std = predictive_dist.stddev()
    #
    #     return sample_mean, sample_std
    #
    # sample_mean, sample_std = jax.vmap(test_vmap, in_axes=(None, None, 0, 0, None, None))(D.X, jnp.expand_dims(D.y[:, gp_idx], axis=-1), exe_path_x, np.expand_dims(exe_path_y[:, :, gp_idx], axis=-1), posterior, initial_sample_points)
    # #
    # # data = gpjax.Dataset(X=comb_x, y=jnp.expand_dims(comb_y[:, gp_idx], axis=-1))
    # # latent_dist = posterior.predict(initial_sample_points, train_data=data)
    # # predictive_dist = posterior.likelihood(latent_dist)
    #
    # # sample_mean = predictive_dist.mean()  # TODO should this be predictive or should it be just posterior, if latter then do we need the points?
    # # sample_std = predictive_dist.stddev()


    def acq_exe_normal(predictive, sample):
        def entropy_given_normal_std_list(std_list):
            return jnp.log(std_list) + jnp.log(jnp.sqrt(2 * jnp.pi)) + 0.5  # TODO check if correct std or var
        h_post = jnp.sum(entropy_given_normal_std_list(jnp.array(predictive)), axis=0)
        h_sample = jnp.mean(jnp.sum(entropy_given_normal_std_list(jnp.array(sample)), axis=0), axis=0)

        acq_exe = h_post - h_sample  # TODO add in the sample average

        return acq_exe

    acq_list = acq_exe_normal(predictive_std, sample_stds)
    # TODO the shapes above are flipped maybe can sort this out at some point

    # best_x = jnp.array([initial_sample_points[jnp.argmin(initial_sample_y)]])
    # best_x = jnp.array([initial_sample_points[jnp.argmin(acq_list)]])

    # # We want to maximise the utility function, but the optimiser performs minimisation. Since we're minimising the sample drawn, the sample is actually the negative utility function.
    # negative_utility_fn = lambda x: sample(x)[0][0]
    # lbfgsb = ScipyBoundedMinimize(fun=negative_utility_fn, method="l-bfgs-b")
    # bounds = (lower_bound, upper_bound)
    # x_star = lbfgsb.run(best_x, bounds=bounds).params

    acq_idx = np.argmax(acq_list)
    acq_opt = initial_sample_points[acq_idx]
    acq_val = acq_list[acq_idx]

    return jnp.expand_dims(acq_opt, axis=0)  # x_star

lower_bound = jnp.array([domain[0][0], domain[1][0]])
upper_bound = jnp.array([domain[0][1], domain[1][1]])
n_init_data = 1
bo_iters = 25
num_samples = 20
num_initial_sample_points = 1000

init_x = jnp.array(((4.146202844165692, 0.44793055421536542),))  # jnp.array(unif_random_sample_domain(domain, n_init_data))
init_y = jnp.array(((0.5, 0.5),))  # jnp.array([step_northwest(xi) for xi in init_x])  # TODO hard coded to test

D = gpjax.Dataset(X=init_x, y=init_y)
output_dims = 2  # TODO can change this for later
prior = create_gp_model(output_dims)

data = adjust_dataset(init_x, init_y)

start_time = time.time()
for i in range(bo_iters):
    print("---" * 5 + f" Start iteration i={i} " + "---" * 5)
    # Generate optimised posterior
    key, _key = jrandom.split(key)
    likelihood = gpjax.likelihoods.Gaussian(num_datapoints=data.n, obs_stddev=Static(jnp.array(1e-6)))
    posterior = prior * likelihood
    opt_posterior, _ = gpjax.fit(model=posterior,
                                 objective=lambda p, d: -gpjax.objectives.conjugate_mll(p, d),
                                 train_data=data,
                                 optim=optax.adam(learning_rate=0.01),
                                 num_iters=1000,
                                 safe=True,
                                 key=_key,
                                 verbose=False)

    # Sample from posteriors and find minimizer
    key, _key = jrandom.split(key)
    ind_time = time.time()
    exe_path, initial_sample_points, sample_mus, sample_stds = sample_and_optimize_posterior(opt_posterior, D,
                                                                                             _key, lower_bound,
                                                                                             upper_bound, num_samples,
                                                                                             num_initial_sample_points)
    print(f"{time.time() - ind_time} - Exe path time taken")
    ind_time = time.time()
    x_star = optimise_sample(opt_posterior, D, initial_sample_points, sample_mus, sample_stds)
    print(f"{time.time() - ind_time} - Optimisation time taken")
    y_star = f([x_star[0, 0], x_star[0, 1]])
    print(f"BO Iteration: {i + 1}, Queried Point: {x_star}, Black-Box Function Value:" f" {y_star}")

    adj_stars = adjust_dataset(x_star, jnp.expand_dims(jnp.array(y_star), axis=0))

    data = data + gpjax.Dataset(X=adj_stars.X, y=adj_stars.y)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot true path and posterior path samples
    plot_path_2d(true_path, ax, true_path=True)
    for path in exe_path.x:
        plot_path_2d_jax(path, ax)

    # Plot observations
    x_obs = [xi[0] for xi in D.X]
    y_obs = [xi[1] for xi in D.X]
    ax.scatter(x_obs, y_obs, color="green", s=120)

    ax.scatter(x_star[1][0], x_star[1][1], color="deeppink", s=120, zorder=100)
    ax.set(xlim=(domain[0][0], domain[0][1]), ylim=(domain[1][0], domain[1][1]), xlabel="$x_1$", ylabel="$x_2$")

    save_figure = True
    if save_figure:
        neatplot.save_figure(f"bax_multi_gpjax_new{i}", "png")

print(time.time() - start_time)







