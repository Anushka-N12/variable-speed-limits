import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp
import sym_metanet

# MetaNetEnv: A traffic simulation environment using the MetaNet framework
class MetaNetEnv:
    def __init__(self, reward_scale=0.01):
        # Simulation parameters
        self.T = 10 / 3600  # Time step duration (10 seconds in hours)
        self.Tfin = 2.5     # Total simulation time in hours
        self.time = 0
        self.timesteps = int(self.Tfin / self.T)  # Total number of steps to simulate
        self.reward_scale = reward_scale  

        self.current_action = 120
        self.prev_action = 120
        self.L = 1  # Link length
        self.lanes = 2  # Number of lanes

        # Synthetic parameters
        self.free_flow_speed = 120
        self.jam_density = 180
        self.critical_density = 33.5

        # For clarity
        self.STATE_DIM = 2 + 6 + 6  # 2 speed limits + 6 speeds + 6 densities
        self.ACTION_RANGE = [60, 120]

        # Demand profile over time (vehicles entering network)
        self.demands = self.create_demands()

        # Build the traffic network and compile the simulation function
        self.build_network()

        # Initialize state variables
        self.reset()

    def create_demands(self):
        # Create time-varying demand arrays using interpolation
        time = np.arange(0, self.Tfin, self.T)

        # d1 = np.interp(time, (2.0, 2.25), (3500, 1000))  # Mainstream origin demand
        # d2 = np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500))  # On-ramp demand
        
        # More realistic dual-peak pattern
        d1 = 2000 + 1500 * np.sin(2*np.pi*time/0.5)**2  # Mainstream
        d2 = 500 + 300 * np.sin(2*np.pi*(time-0.25)/0.5)  # On-ramp

        return np.stack((d1, d2), axis=1)

    def _generate_demand(self, t):
    # Sinusoidal demand pattern
        peak_hour = 0.25 + 0.2 * np.sin(2*np.pi*t/self.episode_length)
        return {
            'mainstream': 3000 * peak_hour,
            'on_ramp': 800 * (1 - peak_hour)
        }

    def build_network(self):
        # Road and model parameters
        L = 1               # Link length (km)
        lanes = 2           # Number of lanes per link
        rho_max = 180       # Maximum density (veh/km/lane)
        rho_crit = 33.5     # Critical density (veh/km/lane)
        v_free = 102        # Free-flow speed (km/h)
        a = 1.867           # Model parameter (driver sensitivity)
        C = (4000, 2000)    # Capacity for origins (veh/h)

        # MetaNet model parameters
        tau = 18 / 3600     # Relaxation time (h)
        kappa = 40          # Anticipation factor
        eta = 60            # Merging priority
        delta = 0.0122      # Smoothing parameter

        # Define network structure using nodes and links
        N1 = Node(name="N1")
        N2 = Node(name="N2")
        N3 = Node(name="N3")

        O1 = MainstreamOrigin[cs.SX](name="O1")
        O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
        D1 = Destination[cs.SX](name="D1")

        L1 = Link[cs.SX](4, lanes, L, rho_max, rho_crit, v_free, a, name="L1")
        L2 = Link[cs.SX](2, lanes, L, rho_max, rho_crit, v_free, a, name="L2")

        # Define network layout (paths from origins to destinations)
        self.net = (
            Network(name="A1")
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_origin(O2, N2)
        )

        # Compile the symbolic MetaNet model to a CasADi function
        sym_metanet.engines.use("casadi", sym_type="SX")
        
        self.net.is_valid(raises=True)
        self.net.step(T=self.T, tau=tau, eta=eta, kappa=kappa, delta=delta)

        self.F = sym_metanet.engine.to_function(net=self.net, T=self.T)

    def reset(self):
        # Reset simulation time
        self.time = 0
        self.current_action = 120
        self.action = 120
        # Initial conditions for density (rho), speed (v), and on-ramp queue (w)
        self.rho = cs.DM([22, 22, 22.5, 24, 30, 32]).T  # Transpose for column vector
        self.v = cs.DM([80, 80, 78, 72.5, 66, 62]).T
        self.w = cs.DM([0, 0]).T

        # Return the initial state observation
        return self._build_state()

    def step(self, action):
        # Convert agent's action (speed limit) into CasADi DM format
        # v_ctrl = cs.DM([float(action)])

        # Validate input
        if isinstance(action, np.ndarray):
            action = float(action.item())

        self.prev_action = self.current_action
        self.current_action = action

        # Ramp metering rate (currently fixed)
        r = cs.DM.ones(1, 1)

        # Retrieve current time step demand
        d = cs.DM(self.demands[self.time])

        # Split state
        rho_L1 = self.rho[:4]
        rho_L2 = self.rho[4:]
        v_L1   = self.v[:4]
        v_L2   = self.v[4:]
        w_O1   = self.w[0:1]
        w_O2   = self.w[1:2]

        # Format action (speed limit)
        v_ctrl_O1 = cs.DM([[float(action)]])
        r_O2 = cs.DM([[1.0]])
        d_O1 = cs.DM([[self.demands[self.time, 0]]])
        d_O2 = cs.DM([[self.demands[self.time, 1]]])

        # Call the MetaNet model
        result = self.F(rho_L1, v_L1, rho_L2, v_L2, w_O1, w_O2, v_ctrl_O1, r_O2, d_O1, d_O2)

        # Update internal state
        self.rho = cs.vertcat(result[0], result[2])  # rho_L1 + rho_L2
        self.v   = cs.vertcat(result[1], result[3])  # v_L1 + v_L2
        self.w   = cs.vertcat(result[4], result[5])  # w_O1 + w_O2

        # Compute new state and reward
        next_state = self._build_state()
        reward = self._compute_reward()

        # Advance simulation time
        self.time += 1
        done = self.time >= self.timesteps

        # Return next state, reward, and done flag
        return next_state, reward, done, {}

    def _build_state(self):
        try:
            # Convert and flatten all inputs
            current = np.asarray(self.current_action/120).flatten()
            prev = np.asarray(self.prev_action/120).flatten()
            speeds = np.asarray(self.v).flatten() / self.free_flow_speed
            densities = np.asarray(self.rho).flatten() / self.jam_density
            
            # Verify shapes
            assert current.shape == (1,), f"Bad current action shape: {current.shape}"
            assert prev.shape == (1,), f"Bad prev action shape: {prev.shape}"
            assert speeds.shape == (6,), f"Bad speeds shape: {speeds.shape}"
            assert densities.shape == (6,), f"Bad densities shape: {densities.shape}"
            
            return np.concatenate([current, prev, speeds, densities]).astype(np.float32)
        
        except Exception as e:
            print(f"State construction failed: {e}")
            print(f"Types - Current: {type(self.current_action)}, "
                f"Prev: {type(self.prev_action)}, "
                f"V: {type(self.v)}, R: {type(self.rho)}")
            print(f"Shapes - V: {np.array(self.v).shape}, "
                f"R: {np.array(self.rho).shape}")
            # raise

    def _compute_reward(self):
        # Corrected stabilized reward calculation
        # rho_np = np.array(self.rho).flatten()
        # w_np = np.array(self.w).flatten()
        
        # epsilon = 1e-6  # Small constant to prevent division by zero
        
        # highway_veh_hours = np.sum(np.clip(rho_np, 0, None) * self.L * self.lanes + epsilon) * self.T
        # queue_veh_hours = np.sum(np.clip(w_np, 0, None) + epsilon) * self.T
        
        # vehicle_hours = highway_veh_hours + queue_veh_hours
        # return -np.tanh(vehicle_hours * self.reward_scale)

        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)  # Arbitrary large queue limit

        # Calculate components with numerical safeguards
        highway = np.sum(rho * self.L * self.lanes) * self.T
        queue = np.sum(w) * self.T

        # Combined and scaled
        total_hours = highway + queue
        reward = -np.tanh(total_hours * self.reward_scale)
        # NaN check and fallback
        if np.isnan(reward):
            print(f"NaN detected! rho: {rho}, w: {w}")
            return -10  # Fallback reward
        
        return float(reward)
