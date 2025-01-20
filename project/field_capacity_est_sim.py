import jax
import jax.numpy as jnp
import jax.random as jr
from jax import config
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)

config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
from gpjax.kernels.computations import DenseKernelComputation

from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#############
# Simulator #
#############

class RegularVariableIrrigationSim:
    def __init__(self, theta0, starts, fc, sigma_theta):
        self.soil_moist = np.dot(theta0, 0 >= starts)
        self.theta0 = theta0
        self.lam = 0.05/(fc * np.log(2))
        self.sigma_theta = sigma_theta
        self.time = 0.0
        self.starts = starts
        self.data = {
            'latent' : {
                't' : [0.0],
                'y' : [self.soil_moist]
            },
            'observed' : {
                't' : [],
                'y' : []
            }
        }
        np.random.seed(42)
    
    def update(self, dt):
        self.soil_moist = np.dot(self.theta0 * np.exp(-self.lam * (self.time - self.starts)/15), self.time >= self.starts)
        self.time += dt
        self.data['latent']['t'].append(self.time)
        self.data['latent']['y'].append(self.soil_moist)

    def observation(self):
        self.data['observed']['t'].append(self.time)
        self.data['observed']['y'].append(self.soil_moist + 0 * np.random.normal(0, self.sigma_theta))

########
# Data #
########

def get_datasets(total_iter, dataset_size, rnd_key, sim=None):
    if not sim:
        df_raw = pd.read_csv('../RainGarden.csv')
        df_f = df_raw.iloc[7000:9880]

        # convert to unix time (and reset by start time) and add zero dimension
        xs = df_f['Time']
        xs = (pd.to_datetime(xs, format='%m/%d/%y %H:%M').astype(int) // 1e9).values
        xs = xs - xs[0]
        ys = df_f[f'wfv_1'].values[:, None]
        xs = xs / (60 * 60 * 24)
    else:
        dt = 0.1
        obs_interval = 15
        t = 15 * 2880
        min_threshold = 0.0

        # run simulation
        obs_timer = 0
        while (sim.time < t) and (sim.soil_moist > min_threshold):
            sim.update(dt)
            obs_timer += dt
            if obs_timer >= obs_interval:
                sim.observation()
                obs_timer = 0
        xs = jnp.array(sim.data['observed']['t'][:-1])
        ys = jnp.array(sim.data['observed']['y'][:-1])[:,None]
        xs = xs / (60 * 24)
    xs = xs.astype(float)
    xs = xs[:, None]
    xs = add_zeros_dim(xs)
    full_data = gpx.Dataset(xs, ys)  
    datasets = []
    for i in range(total_iter):
        # select random data for each dataset
        rnd_key, subkey = jr.split(rnd_key)
        idx = np.array(jr.choice(subkey, np.arange(xs.shape[0]), shape=(dataset_size,), replace=True))
        xs = jnp.array(xs[idx])
        ys = jnp.array(ys[idx])
        datasets.append(gpx.Dataset(xs, ys))

    return full_data, datasets, rnd_key

def add_zeros_dim(xs):
    zeros = np.zeros((xs.shape[0], 1))
    return np.concatenate([xs, zeros], axis=1)

def add_ones_dim(xs):
    ones = np.ones((xs.shape[0], 1))
    return np.concatenate([xs, ones], axis=1)

#######
# Vis #
#######

# Latex document Text width
latex_width = 397.48499
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['font.size'] = 9

def set_size(width=latex_width, height=latex_width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.
    
    Credit to Jack Walton for the function.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """

    fig_width_pt = width
    fig_height_pt = height * fraction
    
    inches_per_pt = 1 / 72.27
    
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_height_pt * inches_per_pt * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def format_data_to_plot(D):
    x = D.X[:, 0]
    y = D.y

    # need to sort the data for plotting
    argsort = np.argsort(x)
    x = x[argsort]
    y = y[argsort]

    return x, y

def format_standard_plot(ax):
    ax.set_xlabel("Elapsed Days")
    ax.set_ylabel("Soil Moisture")
    ax.set_xlim([0, 30])
    ax.set_ylim([0, 60])
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([10, 35, 60])

def format_derivative_plot(ax):
    ax.set_xlabel("Elapsed Days")
    ax.set_ylabel("Soil Moisture Rate")
    ax.set_xlim([0, 30])
    ax.set_ylim([-30, 30])
    ax.set_xticks([0, 15, 30])
    ax.set_yticks([-30, 0, 30])



######
# GP #
######

class StationaryDerivativeKernel(gpx.kernels.AbstractKernel):
    def __init__(
        self,
        kernel: gpx.kernels.AbstractKernel,
        ): 
        self.kernel = kernel
        
        super().__init__(compute_engine=DenseKernelComputation())

    def __call__(
        self, X: Float[Array, "1 D"], Xp: Float[Array, "1 D"]) -> Float[Array, "1"]:
        z = jnp.array(X[1], dtype=int)
        zp = jnp.array(Xp[1], dtype=int)

        k0 = (1 - z)* (1 - zp) * self.kernel(X, Xp)
        k1 = (z * zp) * jnp.array(jax.hessian(self.kernel, argnums=[0,1])(X, Xp), dtype=jnp.float64)[1][0][0][0]
        k2 = (z - zp) * (z - zp) * jnp.array(jax.grad(self.kernel, argnums=[0,1])(X, Xp), dtype=jnp.float64)[zp][0]

        return k0 + k1 + k2

def get_gp(kernel, init_gp_params):
    kernel = StationaryDerivativeKernel(kernel=kernel)
    mean = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=mean, kernel=kernel)

    likelihood = gpx.likelihoods.Gaussian(num_datapoints=1, obs_stddev=np.sqrt(init_gp_params["likelihood"]["variance"]))
    posterior = prior * likelihood

    return prior, posterior
                
def fit_gp(D, posterior, init_gp_mean):
    x = D.X
    y = D.y - init_gp_mean
    train_D = gpx.Dataset(x, y)

    opt_posterior, history = gpx.fit_scipy(
    model=posterior,
    # we use the negative mll as we are minimising
    objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
    train_data=train_D,
    verbose=False
    )
    # print(history)

    return opt_posterior

def get_gp_pred_dist(D_train, x_test, prior, posterior, opt_posterior, init_gp_mean):
    pred_dist_dict = {
        "Prior": {
            "mean": np.nan,
            "std": np.nan
        },
        "Posterior": {
            "mean": np.nan,
            "std": np.nan
        },
        "Opt. Posterior": {
            "mean": np.nan,
            "std": np.nan
        } 
    }

    x = D_train.X
    y = D_train.y - init_gp_mean
    D_train = gpx.Dataset(x, y)

    prior_dist = prior.predict(x_test)
    pred_dist_dict["Prior"]["mean"] = prior_dist.mean() + init_gp_mean
    pred_dist_dict["Prior"]["std"] = prior_dist.stddev()

    post_dist = posterior.predict(x_test, train_data=D_train)
    post_pred_dist = posterior.likelihood(post_dist)
    pred_dist_dict["Posterior"]["mean"] = post_pred_dist.mean() + init_gp_mean
    pred_dist_dict["Posterior"]["std"] = post_pred_dist.stddev()
    
    opt_post_dist = opt_posterior.predict(x_test, train_data=D_train)
    opt_post_pred_dist = opt_posterior.likelihood(opt_post_dist)
    pred_dist_dict["Opt. Posterior"]["mean"] = opt_post_pred_dist.mean() + init_gp_mean
    pred_dist_dict["Opt. Posterior"]["std"] = opt_post_pred_dist.stddev()

    return pred_dist_dict


############
# Sampling #
############

def ready_sample(sample):
    sample = jnp.repeat(sample, 2)[:, None]
    second_row = jnp.tile(jnp.array([0,1]), len(sample) // 2)[:, None]
    sample = jnp.concatenate([sample, second_row], axis=1)
    return sample
    
def rejection_sampling(datasets, gp, init_gp_mean, a, b, T, total_tries, samples_per_try, var_n, refit, rnd_key):
    samples = []
    for i in range(total_tries):
        curr_samples = []

        if refit:
            posterior = fit_gp(datasets[i], gp, init_gp_mean)
        else:
            posterior = gp
        while len(curr_samples) < var_n:
            D_train = gpx.Dataset(datasets[i].X, datasets[i].y - init_gp_mean)

            rnd_key, subkey = jr.split(rnd_key)
            t = jr.uniform(subkey, (samples_per_try,), minval=0, maxval=T)
            t = ready_sample(t)

            rnd_key, subkey = jr.split(rnd_key)
            u = jr.uniform(subkey, (samples_per_try,), minval=0, maxval=1)

            joint_dist = posterior.predict(t, train_data=D_train)
            joint_pred_dist = posterior.likelihood(joint_dist)
            joint_pred_mean = joint_pred_dist.mean()
            joint_pred_std = joint_pred_dist.stddev()

            f_pred_mean = joint_pred_mean[::2] + init_gp_mean
            fd_pred_mean = joint_pred_mean[1::2]
            fd_pred_std = joint_pred_std[1::2]

            y = jax.scipy.stats.norm.cdf(b, loc=fd_pred_mean, scale=fd_pred_std) - \
                jax.scipy.stats.norm.cdf(a, loc=fd_pred_mean, scale=fd_pred_std)
            
            indices = np.where(u < y)[0]
            curr_samples.extend(f_pred_mean[indices].tolist())

            if len(curr_samples) >= var_n:
                break
        samples.append(curr_samples)
        print(f"{i+1:2}/{total_tries} Accepted Samples: {len(curr_samples)}")
        
    return samples, rnd_key


def sample_mean_variance(samples, var_n):
    means = np.zeros(var_n)
    vars = np.zeros(var_n)
    for i in range(var_n):
        means[i] = np.mean([np.mean(s[:i+1]) for s in samples])
        vars[i] = np.var([np.mean(s[:i+1]) for s in samples])
    return means, vars


########
# Main #
########

def main():
    print("Starting Script")

    GROUND_TRUTH = 30.151352963023253
    MIN15_PER_DAY = 4 * 24
    A = -0.1 * MIN15_PER_DAY
    B = -0.05 * MIN15_PER_DAY
    T = 30
    DATASET_SIZE = 100
    TOTAL_ITER = 10
    SAMPLES_PER_ITER = 1000
    REFIT = False
    VAR_N = 100
    INIT_GP_PARAMS = {
        "kernel": {
            "lengthscale": 10.0, 
            "variance": 1.0
        },
        "likelihood": {
            "variance": 1.0
        },
        "mean" :{
            "constant": 0.0
        }
    }
    theta0 = np.array([60])
    starts = np.array([0])
    sigma_theta = 1.0
    rnd_key = jr.key(42)


    print("Loading datasets")

    # run simulation
    sim = RegularVariableIrrigationSim(theta0, starts, GROUND_TRUTH, sigma_theta)
    full_data, datasets, rnd_key = get_datasets(TOTAL_ITER, DATASET_SIZE, rnd_key, sim)

    print("Plotting data")
    cols = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=0.25))
    x, y = format_data_to_plot(full_data)
    plt.plot(x, y, color=cols[0])
    format_standard_plot(ax)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    plt.savefig("data.pdf", bbox_inches='tight')


    kernel = gpx.kernels.RBF(
        active_dims=[0],
        lengthscale=INIT_GP_PARAMS["kernel"]["lengthscale"],
        variance=INIT_GP_PARAMS["kernel"]["variance"]
    )  
    prior, posterior = get_gp(kernel, INIT_GP_PARAMS)
    opt_posterior = fit_gp(datasets[0], posterior, INIT_GP_PARAMS['mean']['constant'])
    opt_params = {
        "kernel": {
            "lengthscale": float(opt_posterior.prior.kernel.kernel.lengthscale.value.take(0)),
            "variance": float(opt_posterior.prior.kernel.kernel.variance.value.take(0))
        },
        "likelihood": {
            "variance": float(opt_posterior.likelihood.obs_stddev)
        }
    }
    print(f"Initial GP Parameters: {INIT_GP_PARAMS}")
    print(f"Optimised GP Parameters: {opt_params}")


    print(f"Plotting GP")    
    x_test = np.linspace(0, 30, 100)

    x_test_gp = add_zeros_dim(x_test[:, None])
    pred_dist_dict = get_gp_pred_dist(datasets[0], x_test_gp, 
                                      prior, posterior, opt_posterior, INIT_GP_PARAMS['mean']['constant'])
    
    x_test_gp = add_ones_dim(x_test[:, None])
    rate_pred_dist_dict = get_gp_pred_dist(datasets[0], x_test_gp,
                                           prior, posterior, opt_posterior, 0.0)
    names = ['Prior', 'Posterior', 'Opt. Posterior']
    for i in range(3):
        fig, axs = plt.subplots(1, 2, figsize=set_size(subplots=(1, 2), fraction=0.75))
        mean = pred_dist_dict[names[i]]["mean"]
        std = pred_dist_dict[names[i]]["std"]
        x_train, y_train = format_data_to_plot(datasets[0])
        format_standard_plot(axs[0])
        axs[0].plot(x_test, mean, label='$\mu$', color=cols[1], linewidth=1)
        axs[0].fill_between(x_test, mean - 2 * std, mean + 2 * std, alpha=0.2, color=cols[1], label='$\mu \pm 2\sigma$')
        axs[0].scatter(x_train, y_train, color=cols[0], label='Samples', marker='.', s=10)
        # axs[0].legend()

        mean = rate_pred_dist_dict[names[i]]["mean"]
        std = rate_pred_dist_dict[names[i]]["std"]
        format_derivative_plot(axs[1])
        axs[1].plot(x_test, mean, label='$\mu$', color=cols[1], linewidth=1.0)
        axs[1].fill_between(x_test, mean - 2 * std, mean + 2 * std, alpha=0.2, color=cols[1], label='$\mu \pm 2\sigma$')
        axs[1].hlines(A, 0, 30, color=cols[2], linestyle='--', label='Lower bound')
        axs[1].hlines(B, 0, 30, color=cols[2], linestyle='--', label='Upper bound')
        # axs[1].legend()

        plt.tight_layout()
        plt.savefig(f"gp_{names[i].replace(' ', '_').lower()}.pdf", bbox_inches='tight')



    print("Rejection Sampling")
    
    samples, rnd_key = rejection_sampling(datasets, 
                                          posterior, INIT_GP_PARAMS['mean']['constant'],
                                          A, B, T, TOTAL_ITER, SAMPLES_PER_ITER, VAR_N, REFIT,
                                          rnd_key)
    
    x, y
    total_samples = np.concatenate(samples)
    total_fc = np.mean(total_samples)
    means, vars = sample_mean_variance(samples, VAR_N)
    print(f"Number of Samples: {len(total_samples)}")
    print(f"Estimated Field Capacity: {total_fc}")
    print(f"{len(vars)}-Sample Variance of Field Capacity: {vars[-1]}")

    # plotting variance and mean field capacity over number of samples
    fig, ax = plt.subplots(1, 1, figsize=set_size(subplots=(1, 1), fraction=0.25))
    plt.plot(1 + np.arange(VAR_N), means)
    ax.fill_between(1 + np.arange(VAR_N), means - 2 * np.sqrt(vars), means + 2 * np.sqrt(vars), alpha=0.2)
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Mean Field Capacity")
    ax.set_xlim([1, VAR_N])
    ax.set_ylim([25, 45])
    ax.hlines(GROUND_TRUTH, 1, VAR_N, color=cols[2], linestyle='--', label='GT')
    plt.savefig("mc_field_capacity.pdf", bbox_inches='tight')

    print("Finished Script")


if __name__ == "__main__":
    main()