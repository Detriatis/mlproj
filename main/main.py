from simulator import RegularVariableIrrigationSim
from data_preperation import DatasetGenerator
from kernels import StationaryDerivativeKernel
from gphandler import GaussianProcessHandler
from rejectionsampler import RejectionSampler
from utils import plot_gp_predictions, set_size, plot_rejection_sampling
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np 
import gpjax as gpx 
from gpjax.kernels import RBF, Periodic, Polynomial, SumKernel, ProductKernel
import distrax 
import jax.numpy as jnp 



def main(): 
    MIN15_PER_DAY = 4 * 24
    LOWERBOUND = -0.1 * MIN15_PER_DAY
    UPPERBOUND = -0.05 * MIN15_PER_DAY
    
    DATASET_PARAMETERS = {
        "total_iter": 10, 
        "dataset_size": 100,
        "rnd_key": 42, 
    }        
    dataset_generator = DatasetGenerator(**DATASET_PARAMETERS)


    SIM = False
    if SIM:
        SIM_PARAMETERS = {
            "fc": 30.151352963023253,  
            "theta0": np.array([60]),
            "starts": np.array([0]),
            "sigma_theta": 1.0,
        }

        OBSERVATION_PARAMETERS = {
            "dt": 0.1, 
            "obs_interval": 15, 
            "t": 15 * 2880,
            "min_threshold": 0.0,
            "obs_timer": 0, 
        }

        ## Initialise the sim
        sim = RegularVariableIrrigationSim(**SIM_PARAMETERS)
        DATASET_PARAMETERS['sim'] = sim 

        dataset_generator.generate_simulation_data(sim, **OBSERVATION_PARAMETERS)
    
    else: 
        
        REALDATA_PARAMETERS = {
            "file_path": '../data/RainGarden.csv', 
            "time_col": 'Time', 
            "value_col": 1, 
            "start_row": 7000, 
            "end_row": 9880
        }

        dataset_generator.load_real_data(**REALDATA_PARAMETERS)
    
    full_data, datasets, rnd_key = dataset_generator.create_datasets()
    training_data = datasets[0]
    X_test, X_test_derivative = dataset_generator.testing_datasets(training_data.X, size=1000)
    
    ###### Kernel Definition and Mean Function
    rbf_kernel = RBF(active_dims=[0], 
                     lengthscale=5.0, 
                     variance=1.0)
    polynomial_kernel = Polynomial(active_dims=[0], degree=2)
    periodic_kernel = Periodic(active_dims=[0], lengthscale=5.0, variance=1.0, period=5.0)
    
    kernel = ProductKernel(kernels=[rbf_kernel, periodic_kernel, polynomial_kernel])
    mean_function = gpx.mean_functions.Zero()
    
    ####### GP Work 
    GP = GaussianProcessHandler(StationaryDerivativeKernel(kernel), mean_function)
    GP.fit_gp(training_data)
    pred_dist_dict = GP.get_prediction_distribution(training_data, X_test)    
    rate_pred_dist_dict = GP.get_prediction_distribution(training_data, X_test_derivative)    


    ###### Plotting the data
    PLOTTING_PARAMETERS = {
        "pred_dist_dict": pred_dist_dict, 
        "rate_pred_dist_dict": rate_pred_dist_dict, 
        "datasets": datasets, 
        "X_test_plot": X_test[:, 0], 
        "lowerbound": LOWERBOUND, 
        "upperbound": UPPERBOUND, 
        "cols": mpl.rcParams["axes.prop_cycle"].by_key()["color"], 
        "set_size": set_size, 
        "output_dir":"../figures/", 
        "subplots":(1, 2), 
        "fraction":0.75
    }
    plot_gp_predictions(**PLOTTING_PARAMETERS)


    ##### Rejection Sampling
    print('Starting Rejection Sampling')
    REJECTION_SAMPLING_PARAMETERS = {
        "gp": GP,
        "lowerbound": LOWERBOUND,
        "upperbound": UPPERBOUND,
        "T": 30,
        "var_n": 100,
        "refit": False,
        "rnd_key": rnd_key,
        "total_tries": DATASET_PARAMETERS["total_iter"],
        "samples_per_try": 1000,
    }

    RS = RejectionSampler(**REJECTION_SAMPLING_PARAMETERS)
    samples = RS.perform_sampling_with_functions(datasets, opt_posterior=False)
    means, vars, fc = RS.compute_sample_statistics(samples, REJECTION_SAMPLING_PARAMETERS['var_n'])

    
    
    #### Plotting Rejection Sampling
    print('Plotting Rejection Sampling')
    RPLOTTING_PARAMETERS = {
        "var_n": REJECTION_SAMPLING_PARAMETERS['var_n'],
        "cols": PLOTTING_PARAMETERS['cols'],
        "output_dir": '../figures/', 
        "ground_truth": SIM_PARAMETERS['fc'] if SIM else 30
    }

    plot_rejection_sampling(means = means, vars = vars, **RPLOTTING_PARAMETERS)

if __name__ == '__main__':
    main()