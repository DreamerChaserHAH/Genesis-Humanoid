"""Pico 4 Ultra + XRoboToolkit publisher for ExtremControl teleop.

Bridges XRoboToolkit-PC-Service-Pybind tracking data to ExtremControl's
Redis-based teleop pipeline, matching the SteamVR publisher's output format.

Hardware: Pico 4 Ultra + 2 controllers + 2 PICO Swift trackers (ankles)

Coordinate systems:
  - XRoboToolkit (documented): right-handed, X-right, Y-up, Z-in (toward user)
    Same as SteamVR. Origin at headset position when app starts.
  - Body joints: "Same world coordinate system as headset" (per docs)
  - Robot frame: X-forward, Y-left, Z-up
  - Transform (same as SteamVR): x_r = -z, y_r = -x, z_r = y

Usage:
    redis-server
    python deploy/pico_publisher.py
    # Stand with arms forward, press A+B on right controller to calibrate
    python deploy/g1_teleop.py --exp_name extremcontrol
"""

import argparse
import json
import time

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_euler,
    quat_mul,
    quat_to_euler,
    rotmat_to_quat,
)
from gs_env.common.utils.motion_utils import G1Retargeter

# Vertical offset from headset to pelvis in Y-up space (meters).
HEAD_TO_PELVIS_Y_OFFSET = 0.45

# Body joint indices (SMPL 24-joint)
BODY_HEAD = 15
BODY_LEFT_ANKLE = 7
BODY_RIGHT_ANKLE = 8


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


class PicoReceiver:
    """Receives XRoboToolkit tracking data and outputs 6-link poses.

    Matches SteamVRReceiver output format exactly:
      - 6 links in G1Retargeter.LINK_ORDER
      - Z-up world frame
      - Quaternions in (w,x,y,z) format

    Data sources:
      - Headset → torso (index 4)
      - Controllers → hands (indices 2, 3)
      - Pelvis → estimated from headset with vertical offset (index 5)
      - Ankles → body joints if available, else motion trackers, else zeros (indices 0, 1)
    """

    def __init__(self, left_ankle_sn: str = "", right_ankle_sn: str = "") -> None:
        import xrobotoolkit_sdk as xrt

        self._xrt = xrt
        xrt.init()

        self._left_ankle_sn = left_ankle_sn
        self._right_ankle_sn = right_ankle_sn
        self._left_ankle_idx: int | None = None
        self._right_ankle_idx: int | None = None
        self._use_body_tracking = False

        # --- Coordinate transforms (IDENTICAL to SteamVRReceiver) ---
        # XRoboToolkit/SteamVR Y-up → Robot Z-up
        # x_r = -z_o, y_r = -x_o, z_r = y_o
        A = torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        self.global_rot = rotmat_to_quat(A).view(1, 4)
        self.global_rot_inv = rotmat_to_quat(A.T).view(1, 4)

        # Local rotation for pelvis tracker
        A = torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        self.base_local_rot = rotmat_to_quat(A)

        # Local rotation for foot trackers
        A = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self.foot_local_rot = rotmat_to_quat(A)

        # Tracker-to-body-center shift
        self.x_shift = torch.tensor([-0.10, 0.0, 0.0], dtype=torch.float32)

        # Fixed offset to align body-joint space to headset space (computed once)
        self._body_to_headset_offset: torch.Tensor | None = None

        # Z offset to bring floor to Z=0 (headset origin is at head height)
        self._z_floor_offset: float | None = None

        self._frame_id: int = 0

    def start(self) -> None:
        """Wait for connection and detect tracking sources."""
        print("[pico] Waiting for XRoboToolkit connection...")
        for _ in range(100):
            pose = self._xrt.get_headset_pose()
            if pose is not None and any(v != 0.0 for v in pose[:3]):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Could not connect to XRoboToolkit PC Service")
        print("[pico] Connected.")

        # Check body tracking
        print("[pico] Checking for body tracking...")
        for _ in range(30):
            if self._xrt.is_body_data_available():
                break
            time.sleep(0.2)

        if self._xrt.is_body_data_available():
            body = self._xrt.get_body_joints_pose()
            l_ankle = body[BODY_LEFT_ANKLE]
            r_ankle = body[BODY_RIGHT_ANKLE]
            head = body[BODY_HEAD]
            print(f"[pico] Body tracking available!")
            print(f"[pico]   Head(15) raw:     {[f'{v:.3f}' for v in head]}")
            print(f"[pico]   L ankle(7) raw:   {[f'{v:.3f}' for v in l_ankle]}")
            print(f"[pico]   R ankle(8) raw:   {[f'{v:.3f}' for v in r_ankle]}")
            if any(v != 0.0 for v in l_ankle[:3]):
                self._use_body_tracking = True
                print("[pico] → Using BODY JOINTS for ankle positions")
            else:
                print("[pico] → Body data zeros, falling back to motion trackers")

        # Fall back to motion trackers
        if not self._use_body_tracking:
            print("[pico] Checking for motion trackers...")
            for _ in range(50):
                if self._xrt.num_motion_data_available() >= 2:
                    break
                time.sleep(0.2)
            n = self._xrt.num_motion_data_available()
            if n >= 2:
                serials = self._xrt.get_motion_tracker_serial_numbers()
                print(f"[pico] Found {n} tracker(s): {serials}")
                self._resolve_ankle_indices(serials)
            else:
                print(f"[pico] WARNING: No ankle source. num_trackers={n}")

    def _resolve_ankle_indices(self, serials: list[str]) -> None:
        for i, sn in enumerate(serials):
            if sn == self._left_ankle_sn:
                self._left_ankle_idx = i
            elif sn == self._right_ankle_sn:
                self._right_ankle_idx = i
        if self._left_ankle_idx is None and self._right_ankle_idx is None and len(serials) >= 2:
            self._left_ankle_idx = 0
            self._right_ankle_idx = 1
            print(f"[pico] Auto-assigned: idx 0 → left, idx 1 → right")

    def shutdown(self) -> None:
        self._xrt.close()

    def _parse_pose(self, raw: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse [x,y,z,qx,qy,qz,qw] → pos(3,), quat(4,) in wxyz order."""
        pos = torch.tensor(raw[:3], dtype=torch.float32)
        quat = torch.tensor([raw[6], raw[3], raw[4], raw[5]], dtype=torch.float32)
        return pos, quat

    def _correct_body_pos(self, body_pos: torch.Tensor) -> torch.Tensor:
        """Convert body-joint position to headset coordinate space.

        Body joints use Y-down, headset uses Y-up. Negate Y, then add fixed offset.
        """
        corrected = body_pos.clone()
        corrected[1] = -corrected[1]  # Y-down → Y-up
        return corrected + self._body_to_headset_offset

    def _get_ankle_poses_body(
        self, h_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get ankle poses from body joints in headset coordinate space.

        On first call, computes a fixed offset between body-joint head and the
        actual headset position. This offset is then applied to all body joint
        positions, so ankles track independently of headset movement.
        """
        body = self._xrt.get_body_joints_pose()
        body_head_pos = torch.tensor(body[BODY_HEAD][:3], dtype=torch.float32)
        lf_raw = torch.tensor(body[BODY_LEFT_ANKLE][:3], dtype=torch.float32)
        rf_raw = torch.tensor(body[BODY_RIGHT_ANKLE][:3], dtype=torch.float32)

        # Compute fixed alignment offset once
        if self._body_to_headset_offset is None:
            corrected_head = body_head_pos.clone()
            corrected_head[1] = -corrected_head[1]  # Y-down → Y-up
            self._body_to_headset_offset = h_pos - corrected_head
            print(f"[pico] Body-to-headset offset: {self._body_to_headset_offset.tolist()}")

        # Convert ankle positions to headset space
        lf_pos = self._correct_body_pos(lf_raw)
        rf_pos = self._correct_body_pos(rf_raw)

        # Identity quats for feet (body joint orientations are SMPL-convention)
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return lf_pos, ident.clone(), rf_pos, ident.clone()

    def _get_ankle_poses_tracker(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get ankle poses from motion tracker API."""
        lf_pos, lf_q = torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0])
        rf_pos, rf_q = torch.zeros(3), torch.tensor([1.0, 0.0, 0.0, 0.0])
        poses = self._xrt.get_motion_tracker_pose()
        if self._left_ankle_idx is not None and self._left_ankle_idx < len(poses):
            lf_pos, lf_q = self._parse_pose(list(poses[self._left_ankle_idx]))
        if self._right_ankle_idx is not None and self._right_ankle_idx < len(poses):
            rf_pos, rf_q = self._parse_pose(list(poses[self._right_ankle_idx]))
        return lf_pos, lf_q, rf_pos, rf_q

    def get_links(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get 6 tracked link poses in G1Retargeter order, Z-up robot frame."""
        self._frame_id += 1

        # --- Raw poses in XRoboToolkit space (Y-up) ---
        h_pos, h_quat = self._parse_pose(list(self._xrt.get_headset_pose()))
        l_pos, l_quat = self._parse_pose(list(self._xrt.get_left_controller_pose()))
        r_pos, r_quat = self._parse_pose(list(self._xrt.get_right_controller_pose()))

        # Pelvis: headset position shifted down in Y
        pelvis_pos = h_pos.clone()
        pelvis_pos[1] -= HEAD_TO_PELVIS_Y_OFFSET  # Y is UP → subtract to go down
        pelvis_quat = h_quat.clone()

        # Ankles
        if self._use_body_tracking:
            lf_pos, lf_quat, rf_pos, rf_quat = self._get_ankle_poses_body(h_pos)
        else:
            lf_pos, lf_quat, rf_pos, rf_quat = self._get_ankle_poses_tracker()

        # Stack: [left_foot, right_foot, left_hand, right_hand, torso(head), pelvis]
        tracked_pos = torch.stack([lf_pos, rf_pos, l_pos, r_pos, h_pos, pelvis_pos])
        tracked_quat = torch.stack([lf_quat, rf_quat, l_quat, r_quat, h_quat, pelvis_quat])

        # --- Y-up → Z-up transform (same as SteamVRReceiver) ---
        global_rot = self.global_rot.repeat(6, 1)
        global_rot_inv = self.global_rot_inv.repeat(6, 1)
        tracked_pos = quat_apply(global_rot, tracked_pos)
        tracked_quat = quat_mul(quat_mul(global_rot, tracked_quat), global_rot_inv)

        # Shift floor to Z=0. Headset origin is at head height, so pelvis Z is
        # negative. The retargeter needs positive pelvis Z for correct scaling.
        if self._z_floor_offset is None:
            pelvis_z = tracked_pos[5, 2].item()  # pelvis Z (negative)
            # Estimate person height: head ~0.45m above pelvis → head Z ≈ pelvis_z + 0.45
            # Floor is ~(head_z + person_height_above_head) below head, but simpler:
            # we want pelvis_z + offset ≈ 0.85 (typical human pelvis height)
            self._z_floor_offset = 0.85 - pelvis_z
            print(f"[pico] Floor Z offset: pelvis_z={pelvis_z:.3f}, offset={self._z_floor_offset:.3f}")
        tracked_pos[:, 2] += self._z_floor_offset

        # Local rotations for feet and pelvis (same as SteamVRReceiver)
        tracked_quat[[0, 1]] = quat_mul(tracked_quat[[0, 1]], self.foot_local_rot)
        tracked_quat[5] = quat_mul(tracked_quat[5], self.base_local_rot)

        # Tracker-to-body-center shift (same as SteamVRReceiver)
        shift_idxs = [0, 1, 4, 5]
        tracked_pos[shift_idxs] = self._manual_shift_x(
            tracked_pos[shift_idxs], tracked_quat[shift_idxs]
        )

        return tracked_pos, tracked_quat, self._frame_id

    def _manual_shift_x(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        euler = quat_to_euler(quat)
        euler[:, 0] = 0.0
        euler[:, 1] = 0.0
        quat_xy = quat_from_euler(euler)
        return pos + quat_apply(quat_xy, self.x_shift.view(1, 3).repeat(euler.shape[0], 1))

    def get_button_a(self) -> bool:
        return bool(self._xrt.get_A_button())

    def get_button_b(self) -> bool:
        return bool(self._xrt.get_B_button())

    def get_button_x(self) -> bool:
        return bool(self._xrt.get_X_button())


class PicoRedisPublisher:
    """Receives Pico tracking, retargets to G1, publishes to Redis."""

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        freq_hz: float,
        save: bool,
        save_dir: str,
        left_ankle_sn: str,
        right_ankle_sn: str,
    ) -> None:
        self._redis = redis.from_url(redis_url)
        self.key_prefix = key_prefix
        self.freq_hz = freq_hz
        self.save = save
        self.save_dir = save_dir

        self.receiver = PicoReceiver(
            left_ankle_sn=left_ankle_sn,
            right_ankle_sn=right_ankle_sn,
        )
        self.retargeter = G1Retargeter()
        self.retargeter.estimate_torso_quat = True

        self._frame_count: int = 0
        self._paused = False
        self._pause_toggle_held = False  # debounce for A+X
        self.save_data: dict = {
            "fps": int(self.freq_hz),
            "pos": [],
            "quat": [],
            "frame_id": [],
            "foot_contact": [],
        }

        self._foot_contact_height_thresh = 0.04
        self._foot_contact_velocity_thresh = 0.5
        self._foot_initial_height: torch.Tensor | None = None
        self._foot_indices = [0, 1]
        self._foot_last_pos = torch.zeros((2, 3), dtype=torch.float32)

    def publish(self, key: str, value: torch.Tensor, frame_id: int) -> None:
        self._redis.set(f"{self.key_prefix}:motion:{key}", json.dumps(_to_list(value)))
        self._redis.set(f"{self.key_prefix}:timestamp:{key}", frame_id)

    def _get_foot_contact(self, all_link_pos: torch.Tensor) -> torch.Tensor:
        foot_pos = all_link_pos[self._foot_indices, :]
        if self._foot_initial_height is None:
            self._foot_initial_height = foot_pos[:, 2].clone()
            self._foot_last_pos = foot_pos.clone()
        foot_height = foot_pos[:, 2]
        foot_not_contact_height = (
            (foot_height - self._foot_initial_height) / self._foot_contact_height_thresh
        ).clamp(0.0, 1.0)
        foot_velocity = (foot_pos - self._foot_last_pos) * self.freq_hz
        self._foot_last_pos = foot_pos.clone()
        foot_not_contact_velocity = (
            torch.norm(foot_velocity[..., :2], dim=-1) / self._foot_contact_velocity_thresh
        ).clamp(0.0, 1.0)
        foot_contact = 1 - (foot_not_contact_height + foot_not_contact_velocity).clamp(0.0, 1.0)
        return foot_contact

    def close(self) -> None:
        self.receiver.shutdown()

    def run(self) -> None:
        print("=" * 80)
        print("[pico_publisher] Started")
        print(f"  Redis prefix: {self.key_prefix}")
        print(f"  Frequency: {self.freq_hz} Hz")
        print("=" * 80)

        self.receiver.start()
        self.receiver.get_links()
        print("[pico_publisher] Receiving data.")
        print("[pico_publisher] Stand with arms forward → press A+B to calibrate")

        try:
            next_publish_time = time.time() + 1.0 / self.freq_hz
            while True:
                tracked_pos, tracked_quat, frame_id = self.receiver.get_links()
                foot_contact = self._get_foot_contact(tracked_pos)

                if self.save:
                    self.save_data["pos"].append(tracked_pos.detach().cpu().clone())
                    self.save_data["quat"].append(tracked_quat.detach().cpu().clone())
                    self.save_data["foot_contact"].append(foot_contact.detach().cpu().clone())
                    self.save_data["frame_id"].append(frame_id)

                if not self.retargeter.calibrated:
                    if self._frame_count % 60 == 0:
                        xrt = self.receiver._xrt
                        h_raw = list(xrt.get_headset_pose())
                        print(f"[debug] --- Frame {self._frame_count} ---")
                        print(f"[debug] headset raw: {[f'{v:.3f}' for v in h_raw]}")
                        if xrt.is_body_data_available():
                            body = xrt.get_body_joints_pose()
                            print(f"[debug] body head(15):    {[f'{v:.3f}' for v in body[15]]}")
                            print(f"[debug] body L ankle(7):  {[f'{v:.3f}' for v in body[7]]}")
                            print(f"[debug] body R ankle(8):  {[f'{v:.3f}' for v in body[8]]}")
                        print(f"[debug] tracked_pos (robot Z-up):\n{tracked_pos}")
                        print(f"[debug] pelvis Z = {tracked_pos[5, 2]:.3f}")
                    self._frame_count += 1
                    if self.receiver.get_button_a() and self.receiver.get_button_b():
                        self.retargeter.calibrate(
                            tracked_pos=tracked_pos,
                            tracked_quat=tracked_quat,
                        )
                        print("[pico_publisher] Calibrated!")
                    continue

                # A+X toggle: pause/resume publishing
                ax_pressed = self.receiver.get_button_a() and self.receiver.get_button_x()
                if ax_pressed and not self._pause_toggle_held:
                    self._paused = not self._paused
                    state = "PAUSED" if self._paused else "RESUMED"
                    print(f"[pico_publisher] {state} (A+X)")
                self._pause_toggle_held = ax_pressed

                if self._paused:
                    continue

                retargeted = self.retargeter.step(
                    tracked_pos=tracked_pos,
                    tracked_quat=tracked_quat,
                    frame_id=frame_id,
                )
                retargeted["foot_contact"] = foot_contact
                for k, v in retargeted.items():
                    self.publish(k, v, frame_id)

                if time.time() >= next_publish_time:
                    next_publish_time = time.time() + 1.0 / self.freq_hz
                    continue
                time.sleep(max(0.0, next_publish_time - time.time()))
                next_publish_time += 1.0 / self.freq_hz
        except KeyboardInterrupt:
            print("\n[pico_publisher] Stopped.")
        finally:
            if self.save:
                import os
                import pickle

                os.makedirs(self.save_dir, exist_ok=True)
                fname = f"pico_{self.save_data['frame_id'][0]}_{self.save_data['frame_id'][-1]}"
                self.save_data["pos"] = torch.stack(self.save_data["pos"], dim=0)
                self.save_data["quat"] = torch.stack(self.save_data["quat"], dim=0)
                self.save_data["foot_contact"] = torch.stack(
                    self.save_data["foot_contact"], dim=0
                )
                self.save_data["frame_id"] = torch.tensor(
                    self.save_data["frame_id"], dtype=torch.int64
                )
                path = os.path.join(self.save_dir, fname + ".pkl")
                with open(path, "wb") as f:
                    pickle.dump(self.save_data, f)
                print(f"Saved to {path}")
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pico 4 Ultra → Redis for ExtremControl")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0")
    parser.add_argument("--key", type=str, default="motion:ref:latest")
    parser.add_argument("--freq", type=float, default=60.0)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="assets/pico")
    parser.add_argument("--left_ankle_sn", type=str, default="")
    parser.add_argument("--right_ankle_sn", type=str, default="")
    args = parser.parse_args()

    PicoRedisPublisher(
        redis_url=args.redis_url,
        key_prefix=args.key,
        freq_hz=args.freq,
        save=args.save,
        save_dir=args.save_dir,
        left_ankle_sn=args.left_ankle_sn,
        right_ankle_sn=args.right_ankle_sn,
    ).run()
