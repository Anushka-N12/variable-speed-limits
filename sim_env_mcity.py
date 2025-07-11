import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp, LinkWithVsl
import sym_metanet

class MCityRoadEnv:
    def __init__(self, reward_scale=0.01):
        self.T = 10 / 3600
        self.Tfin = 2.5
        self.timesteps = int(self.Tfin / self.T)
        self.reward_scale = reward_scale
        self.time = 0

        self.current_action = 120.0
        self.prev_action = 120.0
        self.vsl_count = 3
        self.n_segments = 3  # Total number of segments in the network

        self.L = 1.5  # Each segment ~1.5km
        self.lanes = 8
        self.free_flow_speed = 120
        self.jam_density = 180
        self.critical_density = 33.5
        self.STATE_DIM = 1 + 6 + 6
        self.ACTION_RANGE = [60, 120]

        self.demands = self.create_demands()
        self.build_network()
        self.reset()

    def create_demands(self):
        time = np.arange(0, self.Tfin, self.T)
        d_main = 2500 + 1000 * np.sin(2 * np.pi * time / 0.5)**2
        d_ramps = 300 + 200 * np.sin(2 * np.pi * (time - 0.25) / 0.5)
        return np.stack([d_main, d_ramps, d_ramps, d_ramps], axis=1)  # O1, O2, O3, O4

    def build_network(self):
        L = self.L
        lanes = self.lanes
        rho_max = self.jam_density
        rho_crit = self.critical_density
        v_free = self.free_flow_speed
        a = 1.867
        C = (4000, 1500)

        tau = 18 / 3600
        kappa = 40
        eta = 60
        delta = 0.0122

        # Nodes
        N0 = Node(name="N0")  # Entry node (connected directly to origin)
        N1 = Node(name="N1")  # Entry point of first link
        N2 = Node(name="N2")  # Midpoint of first segment
        N3 = Node(name="N3")  # Midpoint of second segment
        N4 = Node(name="N4")  # Exit nodes

        # Origins and Destination
        O1 = MainstreamOrigin[cs.SX](name="O1")   # Mainline start
        O2 = MeteredOnRamp[cs.SX](C[1], name="O2")  # Incoming ramp end of segment 1
        O3 = MeteredOnRamp[cs.SX](C[1], name="O3")  # Incoming ramp end of segment 2
        O4 = MeteredOnRamp[cs.SX](C[1], name="O4")  # Optional ramp mid-seg 2 if needed
        D1 = Destination[cs.SX](name="D1")

        # Main highway: 3 segments, 1.5 km each (so 4.5 km total), 8 lanes
        L1 = LinkWithVsl[cs.SX](
            3,              # number of segments
            8,              # lanes
            1.5,            # length (km)
            180,            # rho_max
            33.5,           # rho_crit
            102,            # free flow speed
            1.867,          # a (driver sensitivity)
            segments_with_vsl={0, 1, 2},
            alpha=0.1,
            name="L1"
        )
        L2 = Link[cs.SX](1, 8, 0.01, 180, 33.5, 102, 1.867, name="L2")

        # On-ramps: 1 segment each, 2 lanes, short (0.1 km)
        R2 = Link[cs.SX](
            1, 2, 0.1,
            180, 33.5, 102, 1.867,
            name="R2"
        )
        R3 = Link[cs.SX](
            1, 2, 0.1,
            180, 33.5, 102, 1.867,
            name="R3"
        )
        R4 = Link[cs.SX](
            1, 2, 0.1,
            180, 33.5, 102, 1.867,
            name="R4"
        )

        # Define the network
        self.net = (
            Network("MCity")
            .add_path(origin=O1, path=(N0, L1, N1, L2, N4), destination=D1)
            .add_path(origin=O2, path=(N2, R2, N1))  # On-ramp at end of seg 1
            .add_path(origin=O3, path=(N3, R3, N1))  # On-ramp at end of seg 2 (same node for now)
            .add_path(origin=O4, path=(N3, R4, N1))  # Another possible on-ramp
            .add_origin(O1, N0)  # Connect O1 to N0
        )


        engines.use("casadi", sym_type="SX")
        self.net.is_valid(raises=True)
        self.net.step(
            T=self.T, tau=tau, eta=eta, kappa=kappa, delta=delta,
            controls={"L1": {"v_ctrl": True}}
            # , init_conditions={O1: {"v_ctrl": self.free_flow_speed}}
        )

        self.F = sym_metanet.engine.to_function(self.net, more_out=True, compact=2, T=self.T)

    def reset(self):
        self.time = 0
        self.prev_action = self.current_action = np.array([120.0] * self.vsl_count)

        self.rho = cs.DM([22] * 4)  # 3 segs in L1 + 1 in L2
        self.v = cs.DM([80] * 4)
        self.w = cs.DM([0] * 3)

        return self._build_state()

    def step(self, action):
        action = np.asarray(action).flatten()
        assert action.shape == (self.vsl_count,), f"Expected action of shape ({self.vsl_count},), got {action.shape}"

        self.prev_action = self.current_action
        self.current_action = action

        v_ctrl = cs.DM(action.reshape(-1, 1))     # shape: (3, 1)
        r_fixed = cs.DM([[1.0]] * 4)              # 4 on-ramps
        u = cs.vertcat(v_ctrl, r_fixed)           # shape: (7, 1)

        print('Rho shape:', np.array(self.rho).shape,
              'V shape:', np.array(self.v).shape,
              'W shape:', np.array(self.w).shape)
        x = cs.vertcat(self.rho, self.v, self.w)
        d = cs.DM(self.demands[self.time])
        x_next, _ = self.F(x, u, d)

        self.rho = x_next[0:4]
        self.v   = x_next[4:8]
        self.w   = x_next[8:11]

        next_state = self._build_state()
        reward = self._compute_reward()

        self.time += 1
        done = self.time >= self.timesteps
        return next_state, reward, done, {}

    def _build_state(self):
        action_scaled = np.array([self.current_action / 120]).flatten()
        speeds = np.asarray(self.v).flatten() / self.free_flow_speed
        densities = np.asarray(self.rho).flatten() / self.jam_density

        assert speeds.shape == (4,)
        assert densities.shape == (4,)

        # assert action_scaled.shape == (self.vsl_count,), f"Bad action shape: {action_scaled.shape}"
        # assert speeds.shape == (self.n_segments,), f"Bad speeds shape: {speeds.shape}"
        # assert densities.shape == (self.n_segments,), f"Bad densities shape: {densities.shape}"

        return np.concatenate([action_scaled, speeds, densities]).astype(np.float32)

    def _compute_reward(self):
        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)
        highway = np.sum(rho * self.L * self.lanes) * self.T
        queue = np.sum(w) * self.T
        total = highway + queue
        return -float(total)
