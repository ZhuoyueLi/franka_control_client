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
    robot.set_franka_arm_control_mode(ControlMode.HybridJointImpedance)
    state = robot.get_franka_arm_state()
    for i in range(10):
        target_joint_positions = state["q"]
        target_joint_positions[3] += 0.05
        robot.send_joint_position_command(
            joint_positions=target_joint_positions
        )
        pyzlc.sleep(1)
