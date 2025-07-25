from sim_env_base import BaseMetaNetEnv
from sym_metanet import *
import sym_metanet
import casadi as cs
import numpy as np

class TwoLinkEnv(BaseMetaNetEnv):
    def __init__(self, free_flow_speed=120):
        self.vsl_count = 4  # Number of VSL segments
        self.n_segments = 6  # Total number of segments in the network
        
        self.STATE_DIM = self.vsl_count + self.vsl_count + self.n_segments + self.n_segments
        self.lanes = 2
        self.L = 1

        self.free_flow_speed = free_flow_speed
        self.prev_action = np.array([120.0] * self.vsl_count)  # Initial speed limits for VSL segments
        self.current_action = np.array([120.0] * self.vsl_count)  
        super().__init__(reward_scale=0.01)

    def build_network(self):
        self.n_segments = 6
        self.n_origins = 2

        # Road and model parameters; Network constants from METANET literature
        a = 1.867           # Model parameter (driver sensitivity)
        C = (4000, 2000)    # Max Capacity for origins (veh/h); O1, O2

        # MetaNet calibration parameters
        tau = 18 / 3600     # Relaxation time (h); Driver reaction delay
        kappa = 40          # Anticipation factor for downstream conditions
        eta = 60            # Merging priority (on-ramp vs mainline)
        delta = 0.0122      # Smoothing parameter

        # Build nodes, links, origins/destinations
        N1, N2, N3 = Node("N1"), Node("N2"), Node("N3")
        O1 = MainstreamOrigin[cs.SX]("O1")
        O2 = MeteredOnRamp[cs.SX](2000, name="O2")
        D1 = Destination[cs.SX]("D1")

        L1 = LinkWithVsl[cs.SX](
            4,                              # Number of segments
            self.lanes,
            self.L,
            self.jam_density,
            self.critical_density,
            self.free_flow_speed,
            1.867,                          # Driver sensitivity
            segments_with_vsl={0, 1, 2, 3}, # Apply VSL to all segments (or pick subset)
            alpha=0.1,                      # Relaxation coefficient for VSL response
            name="L1"
        )
        
        L2 = Link[cs.SX](2, self.lanes, self.L, self.jam_density, self.critical_density, self.free_flow_speed, 1.867, name="L2")

        self.net = (
            Network("A1")
            .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
            .add_origin(O2, N2)
        )

        engines.use("casadi", sym_type="SX")
        self.net.is_valid(raises=True)
        self.net.step(T=self.T, tau=18/3600, eta=60, kappa=40, delta=0.0122, 
                      controls={"L1": {"v_ctrl": True}}, init_conditions={O1: {"v_ctrl": self.free_flow_speed}})
        self.F = sym_metanet.engine.to_function(self.net, more_out=True, compact=2, T=self.T)


    def step(self, action):
        if isinstance(action, (float, int)):
            action = np.array([action] * self.vsl_count)  
        elif isinstance(action, list):
            action = np.array(action)
        assert action.shape == (4,), f"Expected action shape (4,), got {action.shape}"

        self.prev_action = self.current_action
        self.current_action = action
        # print("Shape of current_action going into v_cntrl, expected to be 4:", self.current_action.shape)
        v_ctrl = cs.DM(np.array(self.current_action).reshape(-1, 1))  # shape (n_vsl_segments, 1)
        # print("Shape of v_ctrl, Should be (4, 1):", v_ctrl.shape)   
        d = cs.DM(self.demands[self.time])
        r = cs.DM([[1.0]])

        # x = cs.vertcat(self.rho, self.v, self.w)
        # rho = self.rho if self.rho.shape[1] == 1 else self.rho.reshape((-1, 1))
        # v = self.v if self.v.shape[1] == 1 else self.v.reshape((-1, 1))
        # w = self.w if self.w.shape[1] == 1 else self.w.reshape((-1, 1))
        # x = cs.vertcat(
        #     self.rho if self.rho.numel() > 1 else self.rho.reshape((-1, 1)),
        #     self.v if self.v.numel() > 1 else self.v.reshape((-1, 1)),
        #     self.w if self.w.numel() > 1 else self.w.reshape((-1, 1)),
        # )

        # u = cs.vertcat(v_ctrl, r)
        # result = self.F(rho_L1, v_L1, rho_L2, v_L2, w_O1, w_O2, v_ctrl, r, d_O1, d_O2)
        # v_ctrl = cs.DM([action] * 4).reshape((-1, 1))  # 4-by-1
        # r = cs.DM([[1.0]])                             # 1-by-1
        u = cs.vertcat(v_ctrl, r)                      # 5-by-1
        # print("Expected u shape:", self.F.size1_in(1))  # Should print 5

        x = cs.vertcat(self.rho, self.v, self.w)       # (14, 1)
        # d = cs.DM(self.demands[self.time]).reshape((-1, 1))  # (2, 1)
        # print("Shapes → x:", x.shape, "u:", u.shape, "d:", d.shape)    # Sanity check
        x_next, _ = self.F(x, u, d)
        # print("F expects control input size:", self.F.size1_in(1))  # should print 5 if vsl_count=4

        # self.rho = cs.vertcat(result[0], result[2])
        # self.v = cs.vertcat(result[1], result[3])
        # self.w = cs.vertcat(result[4], result[5])
        self.rho = x_next[0:6]
        self.v   = x_next[6:12]
        self.w   = x_next[12:14]

        # assert self.rho.shape == (self.n_segments,), f"Bad speeds shape: {self.rho.shape}"
        # assert self.v.shape == (self.n_segments,), f"Bad densities shape: {self.v.shape}"
        # assert self.w.shape == (self.n_segments,), f"Bad densities shape: {self.w.shape}"

        self.time += 1
        done = self.time >= self.timesteps
        return self._build_state(), self._compute_reward(), done, {}
