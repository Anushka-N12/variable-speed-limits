import numpy as np
import casadi as cs
from sym_metanet import engines, Network, Node, Link, Destination, MainstreamOrigin, MeteredOnRamp, LinkWithVsl
import sym_metanet

class MohdRoadEnv:
    def __init__(self, reward_scale=0.01):
        self.T = 10 / 3600
        self.Tfin = 2.5
        self.timesteps = int(self.Tfin / self.T)
        self.reward_scale = reward_scale
        self.time = 0

        self.current_action = np.array([120.0, 120.0, 120.0])
        self.prev_action = np.array([120.0, 120.0, 120.0])
        self.vsl_count = 3

        self.L = 1.4
        self.lanes = 8
        self.free_flow_speed = 120
        self.jam_density = 180
        self.critical_density = 33.5

        self.STATE_DIM = 6 + 18 + 18
        self.ACTION_RANGE = [60, 120]

        self.demands = self.create_demands()
        self.build_network()
        self.reset()

    def create_demands(self):
        time = np.arange(0, self.Tfin, self.T)
        d1 = 5000 + 3000 * np.sin(2 * np.pi * time / 0.5) ** 2
        ramps = [500 + 300 * np.sin(2 * np.pi * (time - 0.1 * i) / 0.5) for i in range(4)]
        return np.stack((d1, *ramps), axis=1)

    def build_network(self):
        L = self.L
        lanes = self.lanes
        rho_max = self.jam_density
        rho_crit = self.critical_density
        v_free = self.free_flow_speed
        a = 1.867
        C = (6000, 2000)

        tau = 18 / 3600
        kappa = 40
        eta = 60
        delta = 0.0122

        nodes = {f"N{i}": Node(name=f"N{i}") for i in range(10)}

        O1 = MainstreamOrigin[cs.SX](name="O1")
        D1 = Destination[cs.SX](name="D1")
        O2 = MeteredOnRamp[cs.SX](C[1], name="O2")
        O3 = MeteredOnRamp[cs.SX](C[1], name="O3")
        O4 = MeteredOnRamp[cs.SX](C[1], name="O4")
        O5 = MeteredOnRamp[cs.SX](C[1], name="O5")

        L1 = LinkWithVsl[cs.SX](3, lanes, L, rho_max, rho_crit, v_free, a, segments_with_vsl={0}, alpha=0.1, name="L1")
        L2 = LinkWithVsl[cs.SX](3, lanes, L, rho_max, rho_crit, v_free, a, segments_with_vsl={1}, alpha=0.1, name="L2")
        L3 = LinkWithVsl[cs.SX](3, lanes, L, rho_max, rho_crit, v_free, a, segments_with_vsl={2}, alpha=0.1, name="L3")

        ramps = {
            "R_out1": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_out1"),
            "R_in1": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_in1"),
            "R_out2a": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_out2a"),
            "R_in2": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_in2"),
            "R_out2b": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_out2b"),
            "R_out2c": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_out2c"),
            "R_out3": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_out3"),
            "R_in3a": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_in3a"),
            "R_in3b": Link[cs.SX](1, 1, 0.2, rho_max, rho_crit, v_free, a, name="R_in3b")
        }

        self.net = Network("Mohd")
        self.net.add_path(origin=O1, path=(nodes["N0"], L1, nodes["N1"], L2, nodes["N2"], L3, nodes["N3"]), destination=D1)
        self.net.add_origin(O2, nodes["N1"]).add_path(path=(nodes["N1"], ramps["R_in1"], nodes["N1"]))
        self.net.add_origin(O3, nodes["N1"]).add_path(path=(nodes["N1"], ramps["R_in2"], nodes["N1"]))
        self.net.add_origin(O4, nodes["N2"]).add_path(path=(nodes["N2"], ramps["R_in3a"], nodes["N2"]))
        self.net.add_origin(O5, nodes["N2"]).add_path(path=(nodes["N2"], ramps["R_in3b"], nodes["N2"]))

        self.net.add_path(origin=O2, path=(nodes["N1"], ramps["R_in1"], nodes["N1"]), destination=D1)
        self.net.add_path(origin=O3, path=(nodes["N1"], ramps["R_in2"], nodes["N1"]), destination=D1)
        self.net.add_path(origin=O4, path=(nodes["N2"], ramps["R_in3a"], nodes["N2"]), destination=D1)
        self.net.add_path(origin=O5, path=(nodes["N2"], ramps["R_in3b"], nodes["N2"]), destination=D1)

        engines.use("casadi", sym_type="SX")
        self.net.is_valid(raises=True)
        self.net.step(T=self.T, tau=tau, eta=eta, kappa=kappa, delta=delta, init_conditions={O1: {"v_ctrl": v_free}})
        self.F = sym_metanet.engine.to_function(net=self.net, more_out=True, compact=2, T=self.T)

    def reset(self):
        self.time = 0
        self.current_action = np.array([120.0, 120.0, 120.0])
        self.prev_action = np.array([120.0, 120.0, 120.0])
        self.rho = cs.DM([22] * 18)
        self.v = cs.DM([80] * 18)
        self.w = cs.DM([0] * 5)
        return self._build_state()

    def step(self, action):
        action = np.array(action)
        assert action.shape == (3,), f"Expected action shape (3,), got {action.shape}"
        self.prev_action = self.current_action
        self.current_action = action

        v_ctrl = cs.DM(action.reshape(-1, 1))
        r_ons = cs.DM([[1.0]] * 5)
        u = cs.vertcat(v_ctrl, r_ons)
        x = cs.vertcat(self.rho, self.v, self.w)
        d = cs.DM(self.demands[self.time])

        x_next, _ = self.F(x, u, d)

        self.rho = x_next[0:18]
        self.v = x_next[18:36]
        self.w = x_next[36:41]

        next_state = self._build_state()
        reward = self._compute_reward()
        self.time += 1
        done = self.time >= self.timesteps

        return next_state, reward, done, {}

    def _build_state(self):
        current = np.asarray(self.current_action / 120).flatten()
        prev = np.asarray(self.prev_action / 120).flatten()
        speeds = np.asarray(self.v).flatten() / self.free_flow_speed
        densities = np.asarray(self.rho).flatten() / self.jam_density
        return np.concatenate([current, prev, speeds, densities]).astype(np.float32)

    def _compute_reward(self):
        rho = np.clip(np.array(self.rho).flatten(), 0, self.jam_density)
        w = np.clip(np.array(self.w).flatten(), 0, 1000)
        highway = np.sum(rho * self.L * self.lanes) * self.T
        queue = np.sum(w) * self.T
        total_hours = highway + queue
        reward = -total_hours
        if np.isnan(reward):
            return -10
        return float(reward)
