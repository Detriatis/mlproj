import jax.numpy as jnp 
import gpjax as gpx 
import jax.random as jr 
import numpy as np 
import jax 
from scipy.stats import truncnorm, norm 
from gphandler import GaussianProcessHandler

class RejectionSampler:
    """
    A class to perform rejection sampling with optional refitting and compute sample statistics
    using Gaussian processes.

    Attributes
    ----------
    gp : gpx.gps.Posterior
        The Gaussian process model used for predictions.
    lowerbound : float
        Lower bound for the rejection sampling region.
    upperbound : float
        Upper bound for the rejection sampling region.
    T : float
        Maximum value for the time range to sample from.
    var_n : int
        Number of accepted samples to collect per dataset.
    refit : bool
        Whether to refit the GP for each dataset during sampling.
    rnd_key : jax.random.PRNGKey
        Random number generator key.
    total_tries : int
        Number of datasets to perform sampling on.
    samples_per_try : int
        Number of samples to attempt per dataset.


    Methods
    -------
    ready_sample(sample)
        Prepares the sample input by expanding and adding auxiliary dimensions.
    perform_sampling(datasets, total_tries, samples_per_try)
        Executes rejection sampling for the given datasets.
    compute_sample_statistics(samples)
        Computes the mean and variance of the collected samples.
    """

    def __init__(self, gp: GaussianProcessHandler, lowerbound, upperbound, T, var_n, refit, rnd_key, samples_per_try, total_tries):
        self.gp = gp
        self.init_gp_mean = gp.init_gp_mean
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.T = T
        self.var_n = var_n
        self.refit = refit
        self.rnd_key = rnd_key
        self.total_tries = total_tries
        self.samples_per_try = samples_per_try

    @staticmethod
    def ready_sample(sample):
        """
        Prepares the sample input by expanding and adding auxiliary dimensions.

        Parameters
        ----------
        sample : jax.numpy.ndarray
            Array of sample time points.

        Returns
        -------
        jax.numpy.ndarray
            Modified sample with auxiliary dimensions for predictions.
        """
        
        sample = jnp.repeat(sample, 2)[:, None]
        second_row = jnp.tile(jnp.array([0, 1]), len(sample) // 2)[:, None]
        
        return jnp.concatenate([sample, second_row], axis=1)

    def perform_sampling(self, datasets, opt_posterior=True):
        """
        Executes rejection sampling for the given datasets.

        Parameters
        ----------
        datasets : list of gpx.Dataset
            List of training datasets for sampling.

        Returns
        -------
        list 
            The samples 
        """
        samples = []

        for i in range(self.total_tries):
            curr_samples = []
            
            posterior = self.gp.posterior 
            if opt_posterior: 
                print('Using optimised posterior for sampling')
                posterior = self.gp.opt_posterior
            
            if self.refit:
                print('Refitting Gaussian for each dataset')
                posterior = self.gp.fit_gp(datasets[i])
            tries = 0 
            while len(curr_samples) < self.var_n:
                tries += 1 
                D_train = gpx.Dataset(datasets[i].X, datasets[i].y - self.init_gp_mean)
                self.rnd_key, subkey = jr.split(self.rnd_key)
                t = jr.uniform(subkey, (self.samples_per_try,), minval=0, maxval=self.T)
                t = self.ready_sample(t)

                self.rnd_key, subkey = jr.split(self.rnd_key)
                u = jr.uniform(subkey, (self.samples_per_try,), minval=0, maxval=1)

                joint_dist = posterior.predict(t, train_data=D_train)
                joint_pred_dist = posterior.likelihood(joint_dist)
                joint_pred_mean = joint_pred_dist.mean()
                joint_pred_std = joint_pred_dist.stddev()

                f_pred_mean = joint_pred_mean[::2] + self.init_gp_mean
                fd_pred_mean = joint_pred_mean[1::2]
                fd_pred_std = joint_pred_std[1::2]
                y = jax.scipy.stats.norm.cdf(self.upperbound, loc=fd_pred_mean, scale=fd_pred_std) - \
                    jax.scipy.stats.norm.cdf(self.lowerbound, loc=fd_pred_mean, scale=fd_pred_std)

                indices = np.where(u < y)[0]
                curr_samples.extend(f_pred_mean[indices].tolist())
                if tries % 100 == 0: 
                    print(f'Try number: {tries}', f'Current Samps: {len(curr_samples)}') 
                    break 
                if len(curr_samples) >= self.var_n:
                    break
            
            samples.append(curr_samples)
            print(f"{i+1:2}/{self.total_tries} Accepted Samples: {len(curr_samples)}")

        return samples
    

    def perform_sampling_with_functions(self, datasets, opt_posterior=True):
        """
        Executes rejection sampling for the given datasets.

        Parameters
        ----------
        datasets : list of gpx.Dataset
            List of training datasets for sampling.

        Returns
        -------
        list 
            The samples 
        """
        samples = []

        for i in range(self.total_tries):
            curr_samples = []

            # Determine the posterior
            posterior = self.gp.posterior
            if opt_posterior:
                posterior = self.gp.opt_posterior

            # Refit the GP if required
            if self.refit:
                posterior = self.gp.fit_gp(datasets[i])

            # Prepare training data
            D_train = gpx.Dataset(datasets[i].X, datasets[i].y - self.init_gp_mean)

            # Sampling directly from the posterior
            self.rnd_key, subkey = jr.split(self.rnd_key)
            t = jr.uniform(subkey, (self.samples_per_try,), minval=0, maxval=self.T)
            t = self.ready_sample(t)

            # Predict joint distribution for test points
            joint_dist = posterior.predict(t, train_data=D_train)
            joint_pred_dist = posterior.likelihood(joint_dist)

            # Sample functions from posterior
            self.rnd_key, subkey = jr.split(self.rnd_key)
            samps = joint_pred_dist.sample(seed=subkey, sample_shape=(10,))
            fc = samps[:, ::2]
            slopes = samps[:, 1::2]

            # Filter based on slope bounds
            valid_indices = (slopes > self.lowerbound) & (slopes < self.upperbound) 
            # Collect valid samples
            curr_samples.extend(fc[valid_indices].tolist())

            # Logging
            print(f"{i+1:2}/{self.total_tries} Accepted Samples: {len(curr_samples)}, Current Mean Estimate: {np.mean(curr_samples)}")
            samples.append(curr_samples)

        return samples
                
            
    def compute_sample_statistics(self, samples, var_n):
        """
        Computes the mean and variance of the collected samples.

        Parameters
        ----------
        samples : list of lists
            Nested list where each sublist contains samples collected for a dataset.

        Returns
        -------
        tuple
            Arrays of sample means and variances.
        """
        means = np.zeros(var_n)
        vars = np.zeros(var_n)
        for i in range(var_n):
            means[i] = np.mean([np.mean(s[:i+1]) for s in samples])
            vars[i] = np.var([np.mean(s[:i+1]) for s in samples])

        self.means = means 
        self.vars = vars 
        total_samples = np.concatenate(samples)
        self.fc = np.mean(total_samples)

        print(f"Estimated Field Capacity: {self.fc}")
        print(f"{len(vars)}-Sample Variance of Field Capacity: {vars[-1]}")
        
        return means, vars, self.fc