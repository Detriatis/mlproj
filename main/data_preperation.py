import numpy as np 
import pandas as pd 
import gpjax as gpx 
import jax.numpy as jnp 
import jax.random as jr
from simulator import RegularVariableIrrigationSim
from jax import config 
config.update("jax_enable_x64", True)

class DatasetGenerator:
    """
    A class for generating datasets for modeling or simulation purposes. Supports loading real-world data
    or using a simulation to generate data.

    Attributes
    ----------
    total_iter : int
        Total number of datasets to generate.
    dataset_size : int
        Number of data points in each dataset.
    rnd_key : numpy.ndarray
        Random key used for dataset generation.
    sim : object, optional
        A simulation object that generates data if real-world data is not used.

    Methods
    -------
    load_real_data(file_path, time_col, value_col, start_row, end_row)
        Load and preprocess real-world data from a CSV file.
    generate_simulation_data()
        Run the simulation to generate observed data.
    add_dim()
        Add a column of fill_value to the input array.
    create_datasets()
        Generate datasets by randomly sampling from the loaded or simulated data.
    """

    def __init__(self, total_iter, dataset_size, rnd_key: int = 42):
        """
        Initialize the DatasetGenerator.

        Parameters
        ----------
        total_iter : int
            Total number of datasets to generate.
        dataset_size : int
            Number of data points in each dataset.
        rnd_key : numpy.ndarray
            Random key used for dataset generation.
        """
        self.total_iter = total_iter
        self.dataset_size = dataset_size
        self.rnd_key = jr.key(rnd_key)
        self.xs = None
        self.ys = None


    
    def load_real_data(self, file_path='../data/RainGarden.csv', time_col='Time', value_col=1, start_row=7000, end_row=9880):
        """
        Load and preprocess real-world data from a CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        time_col : str
            Column name for timestamps.
        value_col : str
            Column name for observed values.
        start_row : int
            Starting row index for data extraction.
        end_row : int
            Ending row index for data extraction.
        """
        df_raw = pd.read_csv(file_path)
        df_f = df_raw.iloc[start_row:end_row]
        # Convert time to Unix time and normalize by days
        xs = pd.to_datetime(df_f[time_col], format='%m/%d/%y %H:%M').astype(int) // 1e9
        xs = xs.values - xs.values[0]
        xs = xs / (60 * 60 * 24)
        ys = df_f.iloc[:,value_col].values[:, None]

        self.xs = self.add_dim(xs[:, None], 0)
        self.ys = ys

    def generate_simulation_data(self, sim: RegularVariableIrrigationSim, dt, obs_interval, t, min_threshold, obs_timer):
        """
        Run the simulation to generate observed data.

        Assumes that the simulation object has `update` and `observation` methods.
        """
        if not sim:
            raise ValueError("Simulation object is not provided.")

        xs, ys = sim.run_simulation(dt, obs_interval, t, min_threshold)

        self.xs = self.add_dim(xs.reshape(-1, 1), 0)
        self.ys = ys
        return xs, ys 

    
    def add_dim(self, xs, fill_value):
        """
        Add a column of fill_value to the input array.

        Parameters
        ----------
        xs : numpy.ndarray
            Input array.

        Returns
        -------
        numpy.ndarray
            Array with an additional column of fill_value.
        """
        full = np.full_like(xs, fill_value)
        return np.concatenate([xs, full], axis=1)

    def create_datasets(self):
        """
        Generate datasets by randomly sampling from the loaded or simulated data.

        Returns
        -------
        tuple
            A tuple containing the full dataset, a list of sampled datasets, and the updated random key.
        """
        if self.xs is None or self.ys is None:
            raise ValueError("Data is not loaded or simulated. Call load_real_data or generate_simulation_data first.")

        xs = jnp.array(self.xs, dtype=jnp.float64)
        ys = jnp.array(self.ys, dtype=jnp.float64)

        full_data = gpx.Dataset(xs, ys)
        datasets = []

        for _ in range(self.total_iter):
            self.rnd_key, subkey = jr.split(self.rnd_key)
            idx = np.array(jr.choice(subkey, np.arange(xs.shape[0]), shape=(self.dataset_size,), replace=True))
            sampled_xs = xs[idx]
            sampled_ys = ys[idx]
            datasets.append(gpx.Dataset(sampled_xs, sampled_ys))

        return full_data, datasets, self.rnd_key
    
    def testing_datasets(self, X, size):
        """
        Generate the testing data considering both normal X and derivative X
        Uniformly sampled over X input domain, with a given size
        Returns
        -------
        tuple
            A tuple containing the X_test (column 1 is 0s) and the derivative assigned values of X_test (column 1 is 1s) 
        """
        X_test = np.linspace(min(X[:,0]), max(X[:,0]), size).reshape(-1, 1)
        X_test_derivative = self.add_dim(X_test, 1)
        X_test = self.add_dim(X_test, 0)

        return X_test, X_test_derivative
