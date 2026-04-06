# G1 Deployment Guide

## Prerequisites

- Unitree G1 in development mode, standing
- Redis server running on PC
- Trained policy checkpoint in `deploy/logs/<exp_name>/`
- Unitree remote controller paired

## Option A: Direct Ethernet

Connect PC directly to G1 via ethernet cable. Update the network interface in
`src/env/gs_env/real/unitree/utils/low_state_handler.py` line 175 to match your
PC's ethernet interface (e.g., `eno1`, `enx2c16dbaafd43`).

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Pico publisher (if using Pico teleop)
python deploy/pico_publisher.py

# Terminal 3: Deploy
python deploy/g1_teleop.py --exp_name extremcontrol --sim False --action_scale 1.0
```

## Option B: WiFi via UDP Bridge

Use when ethernet is not available. Requires a WiFi network where PC and G1 can
communicate directly (no client isolation). The easiest way is to create a hotspot
from the PC.

### 1. Create WiFi hotspot on PC

```bash
nmcli device wifi hotspot ifname <WIFI_INTERFACE> ssid G1-Teleop password "teleop123"
```

Find your WiFi interface with `nmcli device status`.

### 2. Connect G1 to the hotspot

SSH into G1 (via ethernet or existing connection):

```bash
ssh unitree@<G1_IP>
nmcli device wifi list
nmcli device wifi connect G1-Teleop password "teleop123"
```

Note the G1's new IP on the hotspot network (check with `ip addr show`).
The PC hotspot IP is typically `10.42.0.1`.

### 3. Copy and run the DDS bridge on G1 NX

```bash
# From PC
scp deploy/dds_bridge_nx.py unitree@<G1_HOTSPOT_IP>:~/
```

On the G1 NX:

```bash
python dds_bridge_nx.py --pc_ip 10.42.0.1 --iface enP8p1s0
```

- `--pc_ip`: Your PC's IP on the hotspot network
- `--iface`: G1 NX's internal ethernet interface to the MCU (`enP8p1s0`)

You should see `[bridge] Subscribed to rt/lowstate via DDS` and lowstate packets
being forwarded.

### 4. Run teleop on PC

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Pico publisher
python deploy/pico_publisher.py

# Terminal 3: Deploy with bridge
python deploy/g1_teleop.py --exp_name extremcontrol --sim False --action_scale 1.0 --bridge_ip <G1_HOTSPOT_IP>
```

The `--bridge_ip` flag switches from DDS to UDP bridge mode.

## Startup Sequence

1. Power on G1, use remote to stand up (L2 + A), switch to development mode
2. Start redis-server
3. Start pico_publisher.py, calibrate (stand with arms forward, press A+B)
4. Start g1_teleop.py
5. Press **Start** on Unitree remote to begin policy execution
6. Use **A+X** on Pico controller to pause/resume teleop input

## Safety

- **L2 + R2** on Unitree remote: emergency stop
- **A+X** on Pico controller: pause/resume publishing
- Start with `--action_scale 0.1` for initial testing, increase to `1.0` once verified
- The policy ramps action magnitude from 0 to full over the first 50 steps (~1 second)
