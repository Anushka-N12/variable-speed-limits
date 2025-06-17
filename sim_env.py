import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp
import sym_metanet

# MetaNetEnv: A traffic simulation environment using the MetaNet framework
class MetaNetEnv:
    def __init__(self):
        # Simulation parameters
        self.T = 10 / 3600  # Time step duration (10 seconds in hours)
        self.Tfin = 2.5     # Total simulation time in hours
        self.time = 0
        self.timesteps = int(self.Tfin / self.T)  # Total number of steps to simulate

        # Demand profile over time (vehicles entering network)
        self.demands = self.create_demands()

        # Build the traffic network and compile the simulation function
        self.build_network()

        # Initialize state variables
        self.reset()

    def create_demands(self):
        # Create time-varying demand arrays using interpolation
        time = np.arange(0, self.Tfin, self.T)
        d1 = np.interp(time, (2.0, 2.25), (3500, 1000))  # Mainstream origin demand
        d2 = np.interp(time, (0.0, 0.15, 0.35, 0.5), (500, 1500, 1500, 500))  # On-ramp demand
        return np.stack((d1, d2), axis=1)

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

        # Function F simulates one time step
        self.F = sym_metanet.engine.to_function(net=self.net, T=self.T)

    def reset(self):
        # Reset simulation time
        self.time = 0

        # Initial conditions for density (rho), speed (v), and on-ramp queue (w)
        self.rho = cs.DM([22, 22, 22.5, 24, 30, 32])
        self.v = cs.DM([80, 80, 78, 72.5, 66, 62])
        self.w = cs.DM([0, 0])

        # Return the initial state observation
        return self._build_state()

    def step(self, action):
        # Convert agent's action (speed limit) into CasADi DM format
        v_ctrl = cs.DM([float(action)])

        # Ramp metering rate (currently fixed)
        r = cs.DM.ones(1, 1)

        # Retrieve current time step demand
        d = cs.DM(self.demands[self.time])

        # Simulate one time step
        self.rho, self.v, self.w, q, q_o = self.F(self.rho, self.v, self.w, v_ctrl, r, d)

        # Compute new state and reward
        next_state = self._build_state()
        reward = self._compute_reward()

        # Advance simulation time
        self.time += 1
        done = self.time >= self.timesteps

        # Return next state, reward, and done flag
        return next_state, reward, done, {}

    def _build_state(self):
        # Create a simple normalized state vector (e.g., speed normalized to [0,1])
        return np.array([float(self.v[0]) / 120])  # Assuming v_free = 120 km/h

    def _compute_reward(self):
        # Basic reward: encourage higher speed and discourage high density
        speed = float(self.v[0])
        density = float(self.rho[0])
        reward = (speed / 120) - (density / 180)  # Normalize both components
        return reward
