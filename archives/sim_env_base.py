'''-----------------------------------------------------------------------------
   MetaNet Traffic Simulation Environment:
   This environment simulates a traffic network using the MetaNet framework,
   allowing for reinforcement learning agents to control speed limits and
   observe the resulting traffic dynamics.

   This file defines the base class, initializing the common parts of all roadways
   & their methods. The specifics of a roadway are to be defined in subclasses. 

   It has been written with reference to the example files of the following framework repo:
   https://github.com/FilippoAiraldi/sym-metanet
--------------------------------------------------------------------------------'''

import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp, LinkWithVsl
import sym_metanet

class BaseMetaNetEnv:
    def __init__(self, reward_scale=0.01):
        # Simulation parameters
        self.T = 10 / 3600                        # Time step duration (10 seconds in hours)
        self.Tfin = 2.5                           # Total simulation time in hours
        self.timesteps = int(self.Tfin / self.T)  # Total number of steps to simulate = 900
        self.reward_scale = reward_scale          # Scaling factor for reward calculation
        self.time = 0

        # Two VSL segments → two speed limits (Initial)
        # self.current_action = np.array([120.0, 120.0])  # shape (2,)
        # self.prev_action = np.array([120.0, 120.0])

        # Road parameters
        # self.L = 1  # Link length
        # self.lanes = 2  # Number of lanes

        # Synthetic parameters
        self.free_flow_speed = 120   # Max speed (used to normalize)
        self.jam_density = 180       # Max density at which flow stops
        self.critical_density = 33.5 # Density where flow is maximized

        # self.STATE_DIM = 2 + 6 + 6  # 2 speed limits + 6 speeds + 6 densities
        self.ACTION_RANGE = [60, 120]

        # Demand profile over time (vehicles entering network); inflow at points O1 & O2
        self.demands = self.create_demands()

        self.build_network()
        self.reset()

    def create_demands(self):       
        # Will be replaced with real patterns in realistic subclasses 
        # Create time-varying demand arrays using interpolation
        time = np.arange(0, self.Tfin, self.T)

        d1 = 2000 + 1500 * np.sin(2*np.pi*time/0.5)**2  # Mainstream; sinusoidal inflow at O1
        d2 = 500 + 300 * np.sin(2*np.pi*(time-0.25)/0.5)  # On-ramp; sinusoidal inflow at O2
        return np.stack((d1, d2), axis=1)

    def reset(self):
        self.time = 0
        self.prev_action = np.array([120.0] * self.vsl_count)  # Initial speed limits for VSL segments
        self.current_action = np.array([120.0] * self.vsl_count)  

        self.rho = cs.DM([22] * self.n_segments).reshape((-1, 1))
        self.v = cs.DM([80] * self.n_segments).reshape((-1, 1))
        self.w = cs.DM([0] * self.n_origins).reshape((-1, 1))

        return self._build_state()

    def _build_state(self):
        # Convert and flatten all inputs
        current = np.asarray(self.current_action / 120).flatten()
        prev = np.asarray(self.prev_action / 120).flatten()
        speeds = np.asarray(self.v).flatten() / self.free_flow_speed
        densities = np.asarray(self.rho).flatten() / self.jam_density

        # Verify shapes
        assert current.shape == (self.vsl_count,), f"Bad current action shape: {current.shape}"
        assert prev.shape == (self.vsl_count,), f"Bad prev action shape: {prev.shape}"
        assert speeds.shape == (self.n_segments,), f"Bad speeds shape: {speeds.shape}"
        assert densities.shape == (self.n_segments,), f"Bad densities shape: {densities.shape}"

        state = np.concatenate([current, prev, speeds, densities]).astype(np.float32)
        assert state.shape[0] == self.STATE_DIM, f"State shape mismatch: expected {self.STATE_DIM}, got {state.shape[0]}"
        return state
  

    def _compute_reward(self):
        # Encourage lower total time spent (TTS) on road + in ramp queues
        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)  # Arbitrary large queue limit

        # Calculate veh-hours for highway and on-ramp queues, with numerical safeguards
        highway = np.sum(rho * self.L * self.lanes) * self.T
        queue = np.sum(w) * self.T

        # Combined and scaled
        total_hours = highway + queue
        reward = -total_hours       # Negative TTS as reward
        # NaN check and fallback
        if np.isnan(reward):
            print(f"NaN detected! rho: {rho}, w: {w}")
            return -10  # Fallback reward
        
        return float(reward)
