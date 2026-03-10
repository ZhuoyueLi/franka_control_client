import pyzlc

from franka_control_client.franka_robot.panda_arm import (
    ControlMode,
    RemotePandaArm,
)

if __name__ == "__main__":
    pyzlc.init(
        "MujocoRobotClient",
        "127.0.0.1",
        group_name="MujocoRobotGroup",
        log_level=pyzlc.LogLevel.DEBUG,
    )
    robot = RemotePandaArm("MujocoRobot")
    robot.connect()
    robot.set_franka_arm_control_mode(ControlMode.CartesianImpedance)
    for i in range(10):
        robot.send_cartesian_pose_command(
            pos=[0.5, 0.0, 0.5 + 0.01 * i], rot=[0.0, 90.0, 0.0]
        )
        pyzlc.sleep(1)
