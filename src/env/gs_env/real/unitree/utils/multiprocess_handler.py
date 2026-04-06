"""Multiprocess DDS handler — isolates DDS threads in a subprocess to eliminate GIL contention.

On Jetson NX, the DDS callback thread (~1000Hz) holds the GIL and starves the main control loop.
This module runs the entire DDS handler in a child process, communicating via shared memory arrays.
The main process gets its own GIL, achieving full 50Hz control with no contention.

Usage: pass multiprocess=True to UnitreeLeggedEnv (or --multiprocess to g1_teleop.py).
"""

import ctypes
import multiprocessing as mp
import time

import numpy as np

from gs_env.sim.robots.config.schema import HumanoidRobotArgs

# Flag indices in the shared flags array
_F_MSG_RECEIVED = 0
_F_ESTOP_REQ = 1  # main -> child: request emergency stop
_F_ESTOP_STATUS = 2  # child -> main: emergency stop active
_F_START_CMD = 3  # main -> child: call handler.start()
_F_STARTED = 4  # child -> main: start() completed
_F_START_BTN = 5  # Start button state
_F_L2 = 6
_F_R2 = 7


def _child_main(
    cfg: HumanoidRobotArgs,
    state_arr: mp.Array,
    cmd_arr: mp.Array,
    flags_arr: mp.Array,
    num_dof: int,
    num_full_dof: int,
) -> None:
    """Child process entry point — owns all DDS threads."""
    from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler

    handler = LowStateCmdHandler(cfg)
    handler.init()

    # Numpy views into shared memory
    flags = np.frombuffer(flags_arr.get_obj(), dtype=np.int32)
    state = np.frombuffer(state_arr.get_obj(), dtype=np.float64)
    cmd = np.frombuffer(cmd_arr.get_obj(), dtype=np.float64)

    # Signal that DDS is up and first message received
    flags[_F_MSG_RECEIVED] = 1

    # Wait for main process to signal start()
    while not flags[_F_START_CMD]:
        # Pump buttons even before start so main can read Start button
        flags[_F_START_BTN] = handler.Start
        time.sleep(0.01)

    handler.start()

    # Wire handler's target arrays directly to shared memory (zero-copy commands).
    # handler.start() sets target_pos = reset_dof_pos, so copy that first.
    cmd[:num_dof] = handler.target_pos
    cmd[num_dof : 2 * num_dof] = handler.target_vel
    handler.target_pos = cmd[:num_dof]
    handler.target_vel = cmd[num_dof : 2 * num_dof]

    flags[_F_STARTED] = 1

    # Offsets into state array
    o_q = 0
    o_av = 4
    o_jp = 7
    o_jv = 7 + num_dof
    o_fj = 7 + 2 * num_dof
    o_t = 7 + 2 * num_dof + num_full_dof

    # Sync loop: copy handler state -> shared memory at 500Hz
    while True:
        state[o_q : o_q + 4] = handler.quat
        state[o_av : o_av + 3] = handler.ang_vel
        state[o_jp : o_jp + num_dof] = handler.joint_pos
        state[o_jv : o_jv + num_dof] = handler.joint_vel
        state[o_fj : o_fj + num_full_dof] = handler.full_joint_pos
        state[o_t : o_t + num_dof] = handler.torque

        flags[_F_ESTOP_STATUS] = 1 if handler._emergency_stop else 0
        flags[_F_START_BTN] = handler.Start
        flags[_F_L2] = handler.L2
        flags[_F_R2] = handler.R2

        if flags[_F_ESTOP_REQ]:
            handler.emergency_stop()

        time.sleep(0.002)  # 500Hz


class MultiprocessCmdHandler:
    """Drop-in proxy for LowStateCmdHandler that isolates DDS in a child process.

    Provides the same interface as LowStateCmdHandler so UnitreeLeggedEnv works unchanged.
    State is read from shared memory; commands are written to shared memory and read
    directly by the child's LowCmdWrite thread (zero-copy via numpy views).
    """

    def __init__(self, cfg: HumanoidRobotArgs, freq: int = 1000) -> None:
        self.cfg = cfg
        self.dof_names = cfg.dof_names
        self.num_dof = len(self.dof_names)
        self.num_full_dof = 29 if "g1" in cfg.morph_args.file else 12

        n = self.num_dof
        f = self.num_full_dof
        state_size = 4 + 3 + n + n + f + n  # quat + ang_vel + jpos + jvel + fjpos + torque

        self._state_arr = mp.Array(ctypes.c_double, state_size)
        self._cmd_arr = mp.Array(ctypes.c_double, n + n)  # target_pos + target_vel
        self._flags_arr = mp.Array(ctypes.c_int32, 16)

        # Zero-copy numpy views into shared memory
        self._sv = np.frombuffer(self._state_arr.get_obj(), dtype=np.float64)
        self._cv = np.frombuffer(self._cmd_arr.get_obj(), dtype=np.float64)
        self._fv = np.frombuffer(self._flags_arr.get_obj(), dtype=np.int32)

        # State offsets
        self._o_q = 0
        self._o_av = 4
        self._o_jp = 7
        self._o_jv = 7 + n
        self._o_fj = 7 + 2 * n
        self._o_t = 7 + 2 * n + f

        # Static config (same computation as LowStateCmdHandler.__init__)
        self.default_dof_pos = np.array([cfg.default_dof_pos[name] for name in self.dof_names])
        kp_groups = cfg.dof_kp
        self.kp = [
            kp_groups[self._group_from_name(name, kp_groups.keys())] for name in self.dof_names
        ]
        kd_groups = cfg.dof_kd
        self.kd = [
            kd_groups[self._group_from_name(name, kd_groups.keys())] for name in self.dof_names
        ]

        self._process: mp.Process | None = None
        self.msg_received = False
        self._emergency_stop_flag = False

    @staticmethod
    def _group_from_name(joint_name: str, groups: dict[str, float]) -> str:
        for g in sorted(groups, key=len, reverse=True):
            if joint_name.endswith(g + "_joint") or joint_name.endswith(g):
                return g
        raise ValueError(f"No group found for joint: {joint_name}")

    def init(self) -> None:
        self._process = mp.Process(
            target=_child_main,
            args=(
                self.cfg,
                self._state_arr,
                self._cmd_arr,
                self._flags_arr,
                self.num_dof,
                self.num_full_dof,
            ),
            daemon=True,
        )
        self._process.start()
        while not self._fv[_F_MSG_RECEIVED]:
            print("Waiting for Low State Message (multiprocess)...")
            time.sleep(0.5)
        self.msg_received = True
        print("Low State Message Received (multiprocess)!!!")

    def start(self) -> None:
        self._fv[_F_START_CMD] = 1
        while not self._fv[_F_STARTED]:
            time.sleep(0.01)
        print("Handler started (multiprocess)")

    # --- State properties (read from shared memory) ---

    @property
    def quat(self) -> np.ndarray:
        return self._sv[self._o_q : self._o_q + 4].copy()

    @property
    def ang_vel(self) -> np.ndarray:
        return self._sv[self._o_av : self._o_av + 3].copy()

    @property
    def joint_pos(self) -> np.ndarray:
        return self._sv[self._o_jp : self._o_jp + self.num_dof].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self._sv[self._o_jv : self._o_jv + self.num_dof].copy()

    @property
    def full_joint_pos(self) -> np.ndarray:
        return self._sv[self._o_fj : self._o_fj + self.num_full_dof].copy()

    @property
    def torque(self) -> np.ndarray:
        return self._sv[self._o_t : self._o_t + self.num_dof].copy()

    # --- Button properties ---

    @property
    def Start(self) -> int:
        return int(self._fv[_F_START_BTN])

    @property
    def L2(self) -> int:
        return int(self._fv[_F_L2])

    @property
    def R2(self) -> int:
        return int(self._fv[_F_R2])

    # --- Command properties (write to shared memory -> child reads directly) ---

    @property
    def target_pos(self) -> np.ndarray:
        return self._cv[: self.num_dof]

    @target_pos.setter
    def target_pos(self, val: np.ndarray) -> None:
        self._cv[: self.num_dof] = val

    @property
    def target_vel(self) -> np.ndarray:
        return self._cv[self.num_dof : 2 * self.num_dof]

    @target_vel.setter
    def target_vel(self, val: np.ndarray) -> None:
        self._cv[self.num_dof : 2 * self.num_dof] = val

    # --- Control ---

    def emergency_stop(self) -> None:
        self._emergency_stop_flag = True
        self._fv[_F_ESTOP_REQ] = 1

    @property
    def is_emergency_stop(self) -> bool:
        return self._emergency_stop_flag or bool(self._fv[_F_ESTOP_STATUS])
