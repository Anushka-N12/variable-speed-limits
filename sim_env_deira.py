'''-----------------------------------------------------------------------------------------------
This file defines the DeiraEnv class, which simulates a traffic network in the Deira environment.
It is designed to  model part of Sheikh Rashid Road near Deira, Dubai, UAE.
Starting from the intersection with Al Rebat Street, it extends to the intersection with Al Maktoum Road.
This stretch is known for its high traffic volume during peak hours, and real data has been collected. 

It is a large 8-lane road with 2 segments considered, the first segment is 3km long, and second is 5km long.
Uniform lengths & segments were not possible due to data API limits. 

Segment 1:
outgoing ramp - 1.7km
incoming ramp - 2.2km
incoming ramp - 2.3km
division - 2.5km  (could be modelled as a larger outgoing ramp)

Segment 2:
incoming ramp - 1km
outgoing ramp - 1.2km
incoming ramp - 2.1km
outgoing ramp - 2.7km
incoming ramp - 3.5km
incoming ramp - 3.5km
division - 3.9km  (could be modelled as a larger outgoing ramp)
join - 4km  (could be modelled as a larger incoming ramp)
outgoing ramp - 4.4km
incoming ramp - 4.8km
---------------------------------------------------------------------------------------------------'''

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
        self.lanes_ramp = 2
        self.L_ramp = 0.1

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
        return np.stack([d_main] + [d_ramps] * 7, axis=1)  # O1 + 7 on-ramps

    def build_network(self):
        # Constants
        L_ramp = 0.1
        lanes_ramp = 2
        rho_max = 180
        rho_crit = 40
        v_free = 120
        a = 1.5
        tau = 18
        eta = 0.01
        kappa = 40
        delta = 0.05
        C = [1800]     # Ramp capacity

        # Nodes
        N = [Node(name=f"N{i}") for i in range(20)]

        # Origins and Destination
        O1 = MainstreamOrigin[cs.SX](name="O1")
        ramps = [MeteredOnRamp[cs.SX](1800, name=f"O{i+2}") for i in range(7)]
        D1 = Destination[cs.SX](name="D1")

        D1a = Destination[cs.SX](name="D1a")
        D1b = Destination[cs.SX](name="D1b")
        D1c = Destination[cs.SX](name="D1c")

        # Mainline links
        L1 = Link[cs.SX](3, self.lanes, self.L1_len / 3, rho_max, rho_crit, v_free, a, name="L1")
        L2 = LinkWithVsl[cs.SX](6, self.lanes, self.L2_len / 6, rho_max, rho_crit, v_free, a,
                                segments_with_vsl={0,1,2,3,4,5}, alpha=0.1, name="L2")

        # Ramp links (all unique names)
        R_links = [Link[cs.SX](1, self.lanes_ramp, self.L_ramp, rho_max, rho_crit, v_free, a, name=f"RampLink{i}") for i in range(10)]

        # Network layout
        self.net = (
            Network("Deira")
            .add_path(origin=O1, path=(N[0], L1, N[3], L2, N[10]), destination=D1)
            .add_origin(O1, N[0])
        )

        # Segment 1 ramps
        self.net.add_path(destination=D1a, path=(N[1], R_links[0], N[2]))    # out @1.7km
        self.net.add_path(origin=ramps[0], path=(N[4], R_links[1], N[3]))   # in @2.1km
        self.net.add_path(origin=ramps[1], path=(N[5], R_links[2], N[3]))   # in @2.3km

        # Segment 2 ramps
        self.net.add_path(origin=ramps[2], path=(N[6], R_links[3], N[7]))   # in @1.0km
        self.net.add_path(path=(N[7], R_links[4], N[8]))    # out @1.2km

        self.net.add_path(origin=ramps[3], path=(N[9], R_links[5], N[11]))  # in @2.1km
        self.net.add_path(path=(N[11], R_links[6], N[12]))  # out @2.7km

        self.net.add_path(origin=ramps[4], path=(N[13], R_links[7], N[14])) # in @3.5km
        self.net.add_path(origin=ramps[5], path=(N[15], R_links[8], N[14])) # in @3.5km

        self.net.add_path(path=(N[14], R_links[9], N[16]))  # out @4.4km
        # self.net.add_path(origin=ramps[6], path=(N[17], R_links[9], N[18])) # in @4.8km

        # Create a separate link for 4.8km in-ramp
        extra_ramp = Link[cs.SX](1, self.lanes_ramp, self.L_ramp,
                                rho_max, rho_crit, v_free, a, name="RampLink10")
        self.net.add_link(extra_ramp, N[17], N[18])  # Connect to N[18] (4.8km in-ramp)
        self.net.add_path(origin=ramps[6], path=(N[17], extra_ramp, N[18]))  

        # self.net.add_destination(D1, N[10])
        # self.net.add_destination(D1a, N[1])

        # Compile
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
