import pyzlc
from scipy.spatial.transform import Rotation as R
import math

from franka_control_client.franka_robot.panda_arm import (
    ControlMode,
    RemotePandaArm,
)


if __name__ == "__main__":
    pyzlc.init(
        "MujocoRobotClient",
        "192.168.1.1",
        group_port=7730,
        group_name="DroidGroup",
        log_level=pyzlc.LogLevel.DEBUG,
    )
    robot = RemotePandaArm("FrankaPanda")
    robot.connect()
    count = 100
    pos = robot.current_ee_position
    rot = R.from_quat(robot.current_ee_rotation).as_euler("xyz", degrees=True)
    print(f"Current EE position: {pos}, rotation (Euler angles): {rot}")
    robot.set_franka_arm_control_mode(ControlMode.CartesianImpedance)
    for i in range(count * 10):
        desired_pos = [pos[0], pos[1] - 0.1 * math.sin(i / count * 2 * math.pi), pos[2] - 0.1 * math.cos(i / count * 2 * math.pi) + 0.1]
        robot.send_cartesian_pose_command(
            pos=desired_pos, rot=[0, 0, 0]
        )
        pyzlc.sleep(0.05)
