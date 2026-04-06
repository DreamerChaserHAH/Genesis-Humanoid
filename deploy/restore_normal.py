from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
import time

ChannelFactoryInitialize(0, "enP8p1s0")
msc = MotionSwitcherClient()
msc.SetTimeout(5.0)
msc.Init()

status, result = msc.CheckMode()
print(f"Current mode: {result}")

# Select normal mode (restores built-in controller)
msc.SelectMode("normal")
time.sleep(2)

status, result = msc.CheckMode()
print(f"After SelectMode: {result}")
