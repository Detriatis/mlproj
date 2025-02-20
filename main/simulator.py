import numpy as np 
import jax.numpy as jnp 

class RegularVariableIrrigationSim:

    """
    Simulates a repeating exponential distribution model for soil moisture under variable irrigation conditions.

    Attributes
    ----------
    theta0 : float
        Initial soil moisture level.
    starts : float
        Time at which irrigation starts.
    fc : float
        Field capacity of the soil, related to how much water it can hold.
    sigma_theta : float
        Standard deviation of noise added to observations.
    soil_moist : float
        Current soil moisture level.
    lam : float
        Decay constant derived from field capacity and a logarithmic relationship.
    time : float
        Current time in the simulation.
    data : dict
        Stores latent and observed data over time, including timestamps and moisture levels.
    xs : numpy.ndarray or None
        Observed timestamps (normalized) after running the simulation.
    ys : numpy.ndarray or None
        Observed soil moisture values after running the simulation.
    """
        
    def __init__(self, theta0, starts, fc, sigma_theta, seed=42):
        

        self.seed = seed 
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
        self.xs = None
        self.ys = None 

        np.random.seed(self.seed)
    
    def update(self, dt):

        """
        Update the soil moisture and time in the simulation.

        Parameters
        ----------
        dt : float
            Time increment for the simulation step.
        """
        
        self.soil_moist = np.dot(self.theta0 * np.exp(-self.lam * (self.time - self.starts)/15), self.time >= self.starts)
        self.time += dt
        self.data['latent']['t'].append(self.time)
        self.data['latent']['y'].append(self.soil_moist)

    def observation(self):
        """
        Record an observation of the current soil moisture level. 
        Adds observational noise to the soil moisture and updates the observed data dictionary.
        """
        
        self.data['observed']['t'].append(self.time)
        self.data['observed']['y'].append(self.soil_moist + 0 * np.random.normal(0, self.sigma_theta))

    def run_simulation(self, dt=0.1, obs_interval=15, t=15 * 2880, min_threshold=0.0, obs_timer=0): 
        """
        Run the simulation over a specified time period.

        Parameters
        ----------
        dt : float
            Time increment for each simulation step.
        obs_interval : float
            Time interval at which observations are recorded.
        t : float
            Total simulation time.
        min_threshold : float, optional
            Minimum soil moisture threshold to stop the simulation, by default 0.0.

        Notes
        -----
        Normalized timestamps and soil moisture levels are stored in `xs` and `ys`.
        """

        # run simulation
        obs_timer = 0
        while (self.time < t) and (self.soil_moist > min_threshold):
            self.update(dt)
            obs_timer += dt
            if obs_timer >= obs_interval:
                self.observation()
                obs_timer = 0
        xs = jnp.array(self.data['observed']['t'][:-1])
        self.ys = jnp.array(self.data['observed']['y'][:-1])[:,None]
        self.xs = xs / (60 * 24)
        
        return self.xs, self.ys 