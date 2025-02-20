import gpjax as gpx 
import numpy as np

class GaussianProcessHandler:
    """
    A class to handle Gaussian Process (GP) modeling, including setting up the GP, 
    fitting the posterior, and obtaining predictions and optimized parameters.

    Attributes
    ----------
    kernel : gpx.kernels.Kernel
        The kernel function used in the GP.
    init_gp_params : dict
        Initial parameters for the GP, including kernel and likelihood settings.
    prior : gpx.gps.Prior
        The prior distribution of the GP.
    posterior : gpx.gps.Posterior
        The posterior distribution of the GP after combining prior and likelihood.

    Methods
    -------
    initialize_gp()
        Sets up the GP with the given kernel and initial parameters.
    fit_gp(D)
        Fits the posterior GP to the training data.
    get_prediction_distribution(D_train, x_test, init_gp_mean)
        Computes predictive distributions for prior, posterior, and optimized posterior.
    extract_optimized_parameters()
        Extracts the optimized parameters from the posterior distribution.
    """

    def __init__(self, kernel = None, mean_function = None):
        """
        Initialize the GaussianProcessHandler.

        Parameters
        ----------
        kernel : gpx.kernels.Kernel
            The kernel function used in the GP.
        mean_function : gpx.mean_functions
            The mean function used in the GP.
        """
        
        self.mean_function = mean_function
        if isinstance(self.mean_function, gpx.mean_functions.Constant):
            self.init_gp_mean = mean_function.constant 
        else: 
            self.init_gp_mean = 0.0 
        self.kernel = kernel
        self.prior = None 
        self.posterior = None 
        self.initialize_gp()

    def initialize_gp(self):
        """
        Sets up the GP with the given kernel and initial parameters.

        Creates a stationary derivative kernel, zero mean function, and combines them with
        a Gaussian likelihood to form the prior and posterior distributions.
        """

        mean_function = self.mean_function
        self.prior = gpx.gps.Prior(mean_function=mean_function, kernel=self.kernel)

        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=100,
            obs_stddev=np.sqrt(1.0)
        )
        self.posterior = self.prior * likelihood

    def fit_gp(self, D: gpx.Dataset):
        """
        Fits the posterior GP to the training data.

        Parameters
        ----------
        D : gpx.Dataset
            The training dataset containing input-output pairs.

        Returns
        -------
        opt_posterior : gpx.gps.Posterior
            The optimized posterior distribution.
        """
        x = D.X
        y = D.y - self.init_gp_mean
        train_D = gpx.Dataset(x, y)

        opt_posterior, _ = gpx.fit_scipy(
            model=self.posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=train_D,
            verbose=False,
        )   
    
        self.opt_posterior = opt_posterior
        self.opt_params = self.extract_optimized_parameters(opt_posterior)
        print(self.opt_params)
        
        return opt_posterior

    def get_prediction_distribution(self, D_train: gpx.Dataset, x_test: np.ndarray, init_gp_mean: float = 0):
        """
        Computes predictive distributions for prior, posterior, and optimized posterior.

        Parameters
        ----------
        D_train : gpx.Dataset
            Training dataset containing input-output pairs.
        x_test : numpy.ndarray
            Test inputs for which predictions are required.
        init_gp_mean : float
            Initial GP mean to adjust predictions.

        Returns
        -------
        pred_dist_dict : dict
            A dictionary containing means and standard deviations for prior, posterior,
            and optimized posterior predictive distributions.
        """
        pred_dist_dict = {
            "Prior": {"mean": np.nan, "std": np.nan},
            "Posterior": {"mean": np.nan, "std": np.nan},
            "Opt. Posterior": {"mean": np.nan, "std": np.nan},
        }

        x = D_train.X
        y = D_train.y - self.init_gp_mean
        train_D = gpx.Dataset(x, y)

        prior_dist = self.prior.predict(x_test)
        pred_dist_dict["Prior"]["mean"] = prior_dist.mean() + self.init_gp_mean
        pred_dist_dict["Prior"]["std"] = prior_dist.stddev()

        post_dist = self.posterior.predict(x_test, train_data=train_D)
        post_pred_dist = self.posterior.likelihood(post_dist)
        pred_dist_dict["Posterior"]["mean"] = post_pred_dist.mean() + self.init_gp_mean
        pred_dist_dict["Posterior"]["std"] = post_pred_dist.stddev()

        post_dist = self.opt_posterior.predict(x_test, train_data=train_D)
        post_pred_dist = self.opt_posterior.likelihood(post_dist)
        pred_dist_dict["Opt. Posterior"]["mean"] = post_pred_dist.mean() + self.init_gp_mean
        pred_dist_dict["Opt. Posterior"]["std"] = post_pred_dist.stddev()
        
        return pred_dist_dict

    def extract_optimized_parameters(self, posterior=None):
        """
        Extracts the optimized parameters from the posterior distribution.

        Returns
        -------
        opt_params : dict
            A dictionary containing optimized kernel and likelihood parameters.
        """
        params = ['lengthscale', 'variance', 'period']
        
        if posterior is None:
            posterior = self.opt_posterior

        def get_kernels(prior):
            if hasattr(prior, "kernel"):
                return prior.kernel, True
            elif hasattr(prior, "kernels"):
                return prior.kernels, False
            return prior.kernel, False

        kernels = posterior.prior
        flag = True
        while flag:
            kernels, flag = get_kernels(kernels)
            if isinstance(kernels, gpx.kernels.RBF):
                kernels = [kernels]
                break

        opt_params = {}
        for kernel in kernels:
            name = kernel.name
            opt_params[name] = {}
            for param in params:
                attr = getattr(kernel, param, None)
                if attr is not None:
                    opt_params[name][param] = attr.value

        opt_params["likelihood"] = {
            "variance": float(posterior.likelihood.obs_stddev)
        }
        
        return opt_params
