import pyzlc

from franka_control_client.franka_robot.panda_arm import (
    ControlMode,
    RemotePandaArm,
)

if __name__ == "__main__":
    pyzlc.init(
        "policy_inference",
        "192.168.1.1",
        group_name="DroidGroup",
        group_port=7730,
    )
    robot = RemotePandaArm("FrankaPanda")
    robot.connect()
    robot.set_franka_arm_control_mode(ControlMode.IDLE)
