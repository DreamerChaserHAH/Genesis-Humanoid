"""DDS-to-UDP bridge for the G1 robot. Runs on the G1's NX computer.

Bridges DDS low-level messages over UDP so that g1_teleop.py can run on
a remote PC connected via WiFi instead of requiring direct ethernet to the MCU.

NX side (this script):
  - Subscribes to rt/lowstate via DDS (from MCU)
  - Sends lowstate data to PC via UDP
  - Receives lowcmd data from PC via UDP
  - Publishes to rt/lowcmd via DDS (to MCU)

Usage on G1 NX:
    python dds_bridge_nx.py --pc_ip 10.42.0.1 --iface eth0

The --iface should be the NX's internal ethernet interface to the MCU.
"""

import argparse
import socket
import struct
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowState_hg
from unitree_sdk2py.utils.crc import CRC

# Ports
LOWSTATE_PORT = 9501  # NX → PC
LOWCMD_PORT = 9502    # PC → NX

# Number of motors on G1
NUM_MOTORS = 29

# Packet format for lowstate (NX → PC):
#   quaternion:      4f  (16 bytes)
#   gyroscope:       3f  (12 bytes)
#   motor_q:        29f  (116 bytes)
#   motor_dq:       29f  (116 bytes)
#   motor_tau:      29f  (116 bytes)
#   motor_reserve0: 29I  (116 bytes)
#   wireless_remote: 40s (40 bytes)
#   mode_machine:    B   (1 byte)
# Total: 533 bytes
LOWSTATE_FMT = f"<4f3f{NUM_MOTORS}f{NUM_MOTORS}f{NUM_MOTORS}f{NUM_MOTORS}I40sB"
LOWSTATE_SIZE = struct.calcsize(LOWSTATE_FMT)

# Packet format for lowcmd (PC → NX):
#   mode_pr:      B     (1 byte)
#   mode_machine: B     (1 byte)
#   motor_mode:  29B    (29 bytes)
#   motor_q:     29f    (116 bytes)
#   motor_dq:    29f    (116 bytes)
#   motor_kp:    29f    (116 bytes)
#   motor_kd:    29f    (116 bytes)
#   motor_tau:   29f    (116 bytes)
# Total: 611 bytes
LOWCMD_FMT = f"<2B{NUM_MOTORS}B{NUM_MOTORS}f{NUM_MOTORS}f{NUM_MOTORS}f{NUM_MOTORS}f{NUM_MOTORS}f"
LOWCMD_SIZE = struct.calcsize(LOWCMD_FMT)


def main(pc_ip: str, iface: str, domain_id: int = 0) -> None:
    # Initialize DDS
    ChannelFactoryInitialize(domain_id, iface)

    crc = CRC()

    # --- LowState: DDS → UDP ---
    udp_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_count = 0
    last_print = time.time()

    def lowstate_callback(msg: LowState_hg) -> None:
        nonlocal state_count, last_print

        # Pack fields
        quat = msg.imu_state.quaternion
        gyro = msg.imu_state.gyroscope
        motor_q = [msg.motor_state[i].q for i in range(NUM_MOTORS)]
        motor_dq = [msg.motor_state[i].dq for i in range(NUM_MOTORS)]
        motor_tau = [msg.motor_state[i].tau_est for i in range(NUM_MOTORS)]
        motor_reserve = [msg.motor_state[i].reserve[0] for i in range(NUM_MOTORS)]
        wireless = bytes(msg.wireless_remote[:40])
        mode_machine = msg.mode_machine

        packet = struct.pack(
            LOWSTATE_FMT,
            *quat, *gyro, *motor_q, *motor_dq, *motor_tau, *motor_reserve,
            wireless, mode_machine,
        )
        udp_send_sock.sendto(packet, (pc_ip, LOWSTATE_PORT))

        state_count += 1
        now = time.time()
        if now - last_print >= 5.0:
            print(f"[bridge] Forwarded {state_count} lowstate packets in last 5s "
                  f"({state_count / 5.0:.0f} Hz)")
            state_count = 0
            last_print = now

    sub = ChannelSubscriber("rt/lowstate", LowState_hg)
    sub.Init(lowstate_callback, 10)
    print(f"[bridge] Subscribed to rt/lowstate via DDS on {iface}")
    print(f"[bridge] Sending lowstate UDP to {pc_ip}:{LOWSTATE_PORT}")

    # --- LowCmd: UDP → DDS ---
    pub = ChannelPublisher("rt/lowcmd", LowCmd_hg)
    pub.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()

    udp_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_recv_sock.bind(("0.0.0.0", LOWCMD_PORT))
    print(f"[bridge] Listening for lowcmd UDP on port {LOWCMD_PORT}")

    cmd_count = 0
    cmd_last_print = time.time()

    def recv_loop() -> None:
        nonlocal cmd_count, cmd_last_print
        while True:
            data, _ = udp_recv_sock.recvfrom(2048)
            if len(data) != LOWCMD_SIZE:
                continue

            values = struct.unpack(LOWCMD_FMT, data)
            idx = 0
            low_cmd.mode_pr = values[idx]; idx += 1
            low_cmd.mode_machine = values[idx]; idx += 1

            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].mode = values[idx]; idx += 1
            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].q = values[idx]; idx += 1
            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].dq = values[idx]; idx += 1
            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].kp = values[idx]; idx += 1
            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].kd = values[idx]; idx += 1
            for i in range(NUM_MOTORS):
                low_cmd.motor_cmd[i].tau = values[idx]; idx += 1

            low_cmd.crc = crc.Crc(low_cmd)
            pub.Write(low_cmd)

            cmd_count += 1
            now = time.time()
            if now - cmd_last_print >= 5.0:
                print(f"[bridge] Forwarded {cmd_count} lowcmd packets in last 5s "
                      f"({cmd_count / 5.0:.0f} Hz)")
                cmd_count = 0
                cmd_last_print = now

    recv_thread = threading.Thread(target=recv_loop, daemon=True)
    recv_thread.start()

    print("[bridge] DDS ↔ UDP bridge running. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[bridge] Shutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G1 DDS-to-UDP bridge (runs on NX)")
    parser.add_argument("--pc_ip", required=True, help="PC's IP address on the WiFi network")
    parser.add_argument("--iface", default="eth0", help="NX internal network interface to MCU")
    parser.add_argument("--domain_id", type=int, default=0, help="DDS domain ID")
    args = parser.parse_args()
    main(args.pc_ip, args.iface, args.domain_id)
