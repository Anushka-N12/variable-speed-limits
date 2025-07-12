import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp, LinkWithVsl
import sym_metanet

class DeiraEnv:
    def __init__(self, reward_scale=0.01):
        self.T = 10 / 3600
        self.Tfin = 2.5
        self.timesteps = int(self.Tfin / self.T)
        self.reward_scale = reward_scale
        self.time = 0

        self.vsl_count = 6  # VSL applied only to L2 (2nd segment, 6 subsegments)
        self.current_action = np.array([120.0] * self.vsl_count)
        self.prev_action = np.array([120.0] * self.vsl_count)

        self.lanes = 8
        self.L1_len = 2.8
        self.L2_len = 5.6
        self.free_flow_speed = 120
        self.jam_density = 180
        self.critical_density = 33.5

        self.STATE_DIM = self.vsl_count + 12 + self.vsl_count  # action + ρ + v
        self.ACTION_RANGE = [60, 120]

        self.demands = self.create_demands()
        self.build_network()
        self.reset()

    def create_demands(self):
        time = np.arange(0, self.Tfin, self.T)
        d_main = 2500 + 1000 * np.sin(2 * np.pi * time / 0.5)**2
        d_ramps = 300 + 200 * np.sin(2 * np.pi * (time - 0.25) / 0.5)
        return np.stack([d_main] + [d_ramps] * 8, axis=1)  # O1 + 8 on-ramps

    def build_network(self):
        # Constants
        L_ramp = 0.1
        lanes_ramp = 2
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
        N = [Node(name=f"N{i}") for i in range(20)]  # create many to handle all junctions

        # Origins and Destination
        O1 = MainstreamOrigin[cs.SX](name="O1")  # mainline start
        ramps = [MeteredOnRamp[cs.SX](C[1], name=f"O{i+2}") for i in range(8)]  # 8 ramps
        D1 = Destination[cs.SX](name="D1")

        # Mainline links
        L1 = Link[cs.SX](3, self.lanes, self.L1_len / 3, rho_max, rho_crit, v_free, a, name="L1")  # seg1 = 2.8 km
        L2 = LinkWithVsl[cs.SX](6, self.lanes, self.L2_len / 6, rho_max, rho_crit, v_free, a,
                                segments_with_vsl={0, 1, 2, 3, 4, 5}, alpha=0.1, name="L2")  # seg2 = 5.6 km

        # Ramp links (0.1 km, 2 lanes)
        R_links = [Link[cs.SX](1, lanes_ramp, L_ramp, rho_max, rho_crit, v_free, a, name=f"R{i+2}") for i in range(8)]

        # Network layout
        self.net = (
            Network("Deira")
            .add_path(origin=O1, path=(N[0], L1, N[3], L2, N[10]), destination=D1)
            .add_origin(O1, N[0])
        )

        # Add all on-ramps
        self.net.add_path(origin=ramps[0], path=(N[1], R_links[0], N[3]))   # in @ seg1 @2.1
        self.net.add_path(origin=ramps[1], path=(N[2], R_links[1], N[3]))   # in @ seg1 @2.3
        self.net.add_path(origin=ramps[2], path=(N[4], R_links[2], N[5]))   # in @ seg2 @1.0
        self.net.add_path(origin=ramps[3], path=(N[6], R_links[3], N[6]))   # in @ seg2 @2.1
        self.net.add_path(origin=ramps[4], path=(N[7], R_links[4], N[7]))   # in @ seg2 @2.4
        self.net.add_path(origin=ramps[5], path=(N[8], R_links[5], N[8]))   # in @ seg2 @4.4
        self.net.add_path(origin=ramps[6], path=(N[9], R_links[6], N[9]))   # in @ seg2 @4.8
        self.net.add_path(origin=ramps[7], path=(N[11], R_links[7], N[11])) # in @ seg2 @5.2

        # Compile model
        engines.use("casadi", sym_type="SX")
        self.net.is_valid(raises=True)
        self.net.step(
            T=self.T, tau=tau, eta=eta, kappa=kappa, delta=delta,
            controls={"L2": {"v_ctrl": True}},
        )

        self.F = sym_metanet.engine.to_function(self.net, more_out=True, compact=2, T=self.T)

    def reset(self):
        self.time = 0
        self.current_action = self.prev_action = np.array([120.0] * self.vsl_count)

        self.rho = cs.DM([22] * 9)  # 3 from L1, 6 from L2
        self.v = cs.DM([80] * 9)
        self.w = cs.DM([0] * 8)     # 8 on-ramps

        return self._build_state()

    def step(self, action):
        action = np.asarray(action).flatten()
        assert action.shape == (self.vsl_count,), f"Expected action shape ({self.vsl_count},), got {action.shape}"

        self.prev_action = self.current_action
        self.current_action = action

        v_ctrl = cs.DM(action.reshape(-1, 1))       # (6, 1)
        r_fixed = cs.DM([[1.0]] * 8)                 # ramp meter rates
        u = cs.vertcat(v_ctrl, r_fixed)             # (14, 1)

        x = cs.vertcat(self.rho, self.v, self.w)    # (9+9+8 = 26)
        d = cs.DM(self.demands[self.time])
        x_next, _ = self.F(x, u, d)

        self.rho = x_next[0:9]
        self.v = x_next[9:18]
        self.w = x_next[18:26]

        next_state = self._build_state()
        reward = self._compute_reward()
        self.time += 1
        done = self.time >= self.timesteps
        return next_state, reward, done, {}

    def _build_state(self):
        action_scaled = np.array(self.current_action / 120).flatten()
        speeds = np.asarray(self.v[3:]).flatten() / self.free_flow_speed   # only L2
        densities = np.asarray(self.rho[3:]).flatten() / self.jam_density  # only L2

        return np.concatenate([action_scaled, speeds, densities]).astype(np.float32)

    def _compute_reward(self):
        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)
        total = (np.sum(rho * self.L2_len * self.lanes / 6) + np.sum(w)) * self.T
        return -float(total)
