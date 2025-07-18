'''-----------------------------------------------------------------------------
   MetaNet Traffic Simulation Environment:
   This environment simulates a traffic network using the MetaNet framework,
   allowing for reinforcement learning agents to control speed limits and
   observe the resulting traffic dynamics.

   It has been written with reference to the example files of the following framework repo:
   https://github.com/FilippoAiraldi/sym-metanet
--------------------------------------------------------------------------------'''

import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp, LinkWithVsl
import sym_metanet

np.random.seed(42)

# MetaNetEnv: A traffic simulation environment using the MetaNet framework
class MetaNetEnv:
    def __init__(self, reward_scale=0.01):
        # Simulation parameters
        self.T = 10 / 3600                        # Time step duration (10 seconds in hours)
        self.Tfin = 2.5                           # Total simulation time in hours
        self.timesteps = int(self.Tfin / self.T)  # Total number of steps to simulate = 900
        self.reward_scale = reward_scale          # Scaling factor for reward calculation
        self.time = 0

        # Two VSL segments → two speed limits (Initial)
        self.current_action = np.array([120.0, 120.0])  # shape (2,)
        self.prev_action = np.array([120.0, 120.0])
        self.vsl_count = 2  # Number of VSL segments
        self.n_segments = 6  # Total number of segments in the network

        # Road parameters
        self.L = 1.5  # Link length
        self.lanes = 2  # Number of lanes

        # Synthetic parameters
        self.free_flow_speed = 120   # Max speed (used to normalize)
        self.jam_density = 180       # Max density at which flow stops
        self.critical_density = 33.5 # Density where flow is maximized

        self.STATE_DIM = self.vsl_count*2 + self.n_segments*2  # 2 speed limits + 2 previous speed limits + 6 speeds + 6 densities
        self.ACTION_RANGE = [60, 120]

        # Demand profile over time (vehicles entering network); inflow at points O1 & O2
        self.demands = self.create_demands()

        # Build the traffic network and compile the simulation function
        self.build_network()

        # Initialize state variables
        self.reset()

    def create_demands(self):
        # Simulates realistic dual-peak demand across simulation period
        
        # Create time-varying demand arrays using interpolation
        time = np.arange(0, self.Tfin, self.T)

        # d1 = np.interp(time, (2.0, 2.25), (3500, 1000))  # Mainstream origin demand
        # d2 = np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500))  # On-ramp demand
        
        d1 = 2000 + 1500 * np.sin(2*np.pi*time/0.5)**2  # Mainstream; sinusoidal inflow at O1
        d2 = 500 + 300 * np.sin(2*np.pi*(time-0.25)/0.5)  # On-ramp; sinusoidal inflow at O2
        return np.stack((d1, d2), axis=1)

    def build_network(self):
        # Road and model parameters; Network constants from METANET literature
        # L = 1.5               # Link length (km)
        lanes = 2           # Number of lanes per link
        rho_max = 180       # Maximum density (veh/km/lane)
        rho_crit = 33.5     # Critical density (veh/km/lane)
        v_free = 102        # Free-flow speed (km/h)
        a = 1.867           # Model parameter (driver sensitivity)
        C = (4000, 2000)    # Max Capacity for origins (veh/h); O1, O2

        # MetaNet calibration parameters
        tau = 18 / 3600     # Relaxation time (h); Driver reaction delay
        kappa = 40          # Anticipation factor for downstream conditions
        eta = 60            # Merging priority (on-ramp vs mainline)
        delta = 0.0122      # Smoothing parameter

        # Define network structure using nodes and links
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")

        O1 = MainstreamOrigin[cs.SX](name="O1")
        O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
        D1 = Destination[cs.SX](name="D1")

        # Define links with segment counts
        # L1 = Link[cs.SX](4, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
        L1 = LinkWithVsl[cs.SX](3, lanes, self.L, rho_max, rho_crit, v_free, a,
                       segments_with_vsl={1, 2}, alpha=0.1, name="L1")
        L2 = Link[cs.SX](2, lanes, self.L, rho_max, rho_crit, v_free, a, name="L2")

        O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
        R2 = Link[cs.SX](1, 2, 0.1, rho_max, rho_crit, v_free, a, name="L3")

        # Define network layout (paths from origins to destinations)
        # self.net = (
        #     Network(name="A1")
        #     .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        #     .add_origin(O2, N2)
        # )
        self.net = (
            Network("Test")
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_path(origin=O2, path=(N3, R2))
            .add_origin(O1, N1)
        )

        # Compile the symbolic MetaNet model into a CasADi function
        sym_metanet.engines.use("casadi", sym_type="SX")
        self.net.is_valid(raises=True)
        # self.net.step(T=self.T, tau=tau, eta=eta, kappa=kappa, delta=delta)
        self.net.step(T=self.T, tau=tau, eta=eta,
              kappa=kappa, delta=delta,
              init_conditions={O1: {"v_ctrl": v_free}})

        # self.F = sym_metanet.engine.to_function(net=self.net, T=self.T)
        self.F = sym_metanet.engine.to_function(
            net=self.net,
            more_out=True,
            compact=2,  # Use compact mode to match (x, u, d) input format
            T=self.T,
        )


    def reset(self):
        # Reset simulation time
        self.time = 0
        self.current_action = np.array([80.0] * self.vsl_count)
        self.action = np.array([80.0] * self.vsl_count)

        # Initial conditions for density (rho), speed (v), and on-ramp queue (w)
        # self.rho = cs.DM([22, 22, 22.5, 24, 30, 32]).T  # Transpose for column vector
        # self.v = cs.DM([80, 80, 78, 72.5, 66, 62]).T
        # self.w = cs.DM([0, 0]).T

        self.rho = cs.DM([22, 22, 22.5, 24, 30])
        self.v = cs.DM([80, 80, 78, 72.5, 66])
        self.w = cs.DM([0, 0])

        return self._build_state()

    # def step(self, action):
    #     # Convert agent's action (speed limit) into CasADi DM format
    #     # v_ctrl = cs.DM([float(action)])

    #     # Validate input
    #     if isinstance(action, np.ndarray):
    #         action = float(action.item())

    #     self.prev_action = self.current_action
    #     self.current_action = action

    #     # Inputs for MetaNet: demand, control, etc.
    #     r = cs.DM.ones(1, 1)                # Ramp metering rate (currently fixed)
    #     d = cs.DM(self.demands[self.time])  # Retrieve current time step demand
    #     v_ctrl_O1 = cs.DM([[float(action)]])    # Format action (speed limit)

    #     # Split state; highway into L1 and L2 segments
    #     rho_L1 = self.rho[:4]
    #     rho_L2 = self.rho[4:]
    #     v_L1   = self.v[:4]
    #     v_L2   = self.v[4:]
    #     w_O1   = self.w[0:1]
    #     w_O2   = self.w[1:2]
    #     r_O2 = cs.DM([[1.0]])
    #     d_O1 = cs.DM([[self.demands[self.time, 0]]])
    #     d_O2 = cs.DM([[self.demands[self.time, 1]]])

    #     # Call the MetaNet model
    #     result = self.F(rho_L1, v_L1, rho_L2, v_L2, w_O1, w_O2, v_ctrl_O1, r_O2, d_O1, d_O2)

    #     # Update internal state
    #     self.rho = cs.vertcat(result[0], result[2])  # rho_L1 + rho_L2
    #     self.v   = cs.vertcat(result[1], result[3])  # v_L1 + v_L2
    #     self.w   = cs.vertcat(result[4], result[5])  # w_O1 + w_O2

    #     # Compute new state and reward
    #     next_state = self._build_state()
    #     reward = self._compute_reward()

    #     # Advance simulation time
    #     self.time += 1
    #     done = self.time >= self.timesteps

    #     return next_state, reward, done, {}
    
    def step(self, action):
        # Expecting a list/array with 2 values (for segments 2 and 3)
        if isinstance(action, (float, int)):
            action = np.array([action, action])
        elif isinstance(action, list):
            action = np.array(action)
        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"

        self.prev_action = self.current_action
        self.current_action = action

        v_ctrl_O1 = cs.DM(action.reshape(-1, 1))  # (2, 1)
        r_O2 = cs.DM([[1.0]])
        u = cs.vertcat(v_ctrl_O1, r_O2)  # Now shape (3, 1)

        # Format inputs for MetaNet function
        # v_ctrl = cs.DM([[float(action[0])], [float(action[1])]])  # if `action` is array-like
        # u = cs.vertcat(v_ctrl, cs.DM([[1.0]]))
        # print('Rho shape:', np.array(self.rho).shape, 'V shape:', np.array(self.v).shape, 'W shape:', np.array(self.w).shape)
        x = cs.vertcat(self.rho, self.v, self.w)                        # State: densities, speeds, queues
        # u = cs.vertcat(cs.DM([float(action)]), cs.DM([1.0]))           # Control: speed limit + metering rate
        d = cs.DM(self.demands[self.time])                             # Demand: from O1 and O2
        # print("Expected u shape:", self.F.size1_in(1)) 
        # Run MetaNet step
        x_next, _ = self.F(x, u, d)  # Discard q_all (second output) for now

        # Split next state
        self.rho = x_next[0:5]
        self.v   = x_next[5:10]
        self.w   = x_next[10:12]

        # Construct new state, compute reward
        next_state = self._build_state()
        reward = self._compute_reward()

        self.time += 1
        done = self.time >= self.timesteps

        return next_state, reward, done, {}


    def _build_state(self):
        try:
            # Convert and flatten all inputs
            # current = np.asarray(self.current_action/120).flatten()
            # prev = np.asarray(self.prev_action/120).flatten()
            current = np.asarray(self.current_action / 120).flatten()
            prev = np.asarray(self.prev_action / 120).flatten()
            speeds = np.asarray(self.v).flatten() / self.free_flow_speed
            densities = np.asarray(self.rho).flatten() / self.jam_density
            
            # Verify shapes
            assert current.shape == (2,), f"Bad current action shape: {current.shape}"
            assert prev.shape == (2,), f"Bad prev action shape: {prev.shape}"
            assert speeds.shape == (self.n_segments,), f"Bad speeds shape: {speeds.shape}"
            assert densities.shape == (self.n_segments,), f"Bad densities shape: {densities.shape}"

            return np.concatenate([current, prev, speeds, densities]).astype(np.float32)
            # return np.concatenate([current, speeds, densities]).astype(np.float32)

                    
        except Exception as e:
            print(f"State construction failed: {e}")
            print(f"Types - Current: {type(self.current_action)}, "
                f"Prev: {type(self.prev_action)}, "
                f"V: {type(self.v)}, R: {type(self.rho)}")
            print(f"Shapes - V: {np.array(self.v).shape}, "
                f"R: {np.array(self.rho).shape}")
            # raise

    def _compute_reward(self, baseline=False):
        # Corrected stabilized reward calculation
        # rho_np = np.array(self.rho).flatten()
        # w_np = np.array(self.w).flatten()
        
        # epsilon = 1e-6  # Small constant to prevent division by zero
        
        # highway_veh_hours = np.sum(np.clip(rho_np, 0, None) * self.L * self.lanes + epsilon) * self.T
        # queue_veh_hours = np.sum(np.clip(w_np, 0, None) + epsilon) * self.T
        
        # vehicle_hours = highway_veh_hours + queue_veh_hours
        # return -np.tanh(vehicle_hours * self.reward_scale)

        # Encourage lower total time spent (TTS) on road + in ramp queues
        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)  # Arbitrary large queue limit

        # Calculate veh-hours for highway and on-ramp queues, with numerical safeguards
        highway = np.sum(rho * self.L * self.lanes) * self.T
        queue = np.sum(w) * self.T

        # Combined and scaled
        total_hours = highway + queue
        # reward = -np.tanh(total_hours * self.reward_scale)
        # reward = -total_hours * self.reward_scale  # Scale reward
        # reward = total_hours * self.reward_scale  
        reward = -total_hours
        # reward = total_hours
        # reward = (-total_hours + 2)          
        # reward = (-total_hours + 1)  
        # reward = (-total_hours + 1) * 10 
        # reward = (-total_hours + 0.5)         
        # reward = (-total_hours + 0.5) * 10           # Negative TTS as reward
        # reward = 1/total_hours
        # Encourage higher VSLs when traffic is not congested
        
        # vsl_term = np.mean(self.current_action[0]) / self.free_flow_speed  # Normalize to [0,1]
        # reward += vsl_term * 1

        # NaN check and fallback
        if np.isnan(reward):
            print(f"NaN detected! rho: {rho}, w: {w}")
            return -10  # Fallback reward
        
        return float(reward)