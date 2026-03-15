import threading
import time
from typing import List, Optional

import numpy as np
import scipy
from franka_control_client.core.remote_device import RemoteDevice
from simpub import init_xr_node_manager, MetaQuest3
import pyzlc
from scipy.spatial.transform import Rotation as R

from franka_control_client.franka_robot.panda_arm import (
    ControlMode,
    RemotePandaArm,
)
from franka_control_client.franka_robot.panda_robotiq import RemoteRobotiqGripper

def print_state(state: Optional[dict]) -> None:
    if state is None:
        print("No state data available.")
        return
    print("Current Franka Arm State: ===============================")
    print(f"EE Position: {state['EE_pos']}")
    print(f"EE Rotation (quaternion): {state['EE_quat']}")
    print(f"Joint Angles: {state['q']}")

class MetaQuest3PandaController(RemoteDevice):
    def __init__(self, device_name: str, panda: RemotePandaArm) -> None:
        self.mq3 = MetaQuest3(device_name)
        self.gripper = RemoteRobotiqGripper("FrankaPanda")
        self.panda = panda
        self.mq3.register_trigger_press_event(
            "hand_trigger", "right", self.start_control
        )
        self.mq3.register_trigger_release_event(
            "hand_trigger", "right", self.stop_control
        )
        self.on_control = False
        self.base_ee_position: Optional[List[float]] = None
        self.base_ee_rotation: Optional[List[float]] = None
        self.base_mq3_position: Optional[List[float]] = None
        self.base_mq3_rotation: Optional[List[float]] = None

    def start_control(self) -> None:
        arm_state = self.panda.current_state
        if arm_state is None:
            raise ValueError("No arm state data received from the robot.")
        controller_data = self.mq3.get_controller_data()
        if controller_data is None:
            raise ValueError("No controller data received from Meta Quest 3.")
        right_data = controller_data["right"]
        self.base_ee_position, self.base_ee_rotation = (
            arm_state["EE_pos"],
            arm_state["EE_quat"],
        )
        self.base_mq3_position, self.base_mq3_rotation = (
            right_data["pos"],
            right_data["rot"],
        )
        self.on_control = True

    def stop_control(self) -> None:
        self.on_control = False
        self.base_ee_position = None
        self.base_ee_rotation = None
        self.base_mq3_position = None
        self.base_mq3_rotation = None

    def connect(self) -> None:
        pass

    def update(self) -> None:
        # arm_state = self.panda.current_state
        # arm_state = self.panda.current_state  # to ensure we have the latest state
        # print(f"Current arm state: {arm_state['q']}")
        if not self.on_control:
            return
        controller_data = self.mq3.get_controller_data()
        if controller_data is None or self.base_mq3_position is None:
            return
        right_data = controller_data["right"]
        mq3_position, mq3_rotation = right_data["pos"], right_data["rot"]
        delta_position = np.array(mq3_position) - np.array(
            self.base_mq3_position
        )
        desired_position = np.array(self.base_ee_position) + delta_position
        # print(f"Control: Base MQ3 position: {self.base_mq3_position}, mq3_position: {mq3_position} delta_position: {delta_position}, desired_position: {desired_position}")
        r_mq3 = R.from_quat(mq3_rotation)
        r_mq3_base = R.from_quat(self.base_mq3_rotation)

        # MQ3 relative rotation
        delta_rot = r_mq3 * r_mq3_base.inv()

        r_ee_base = R.from_quat(self.base_ee_rotation)

        # apply delta rotation to EE
        desired_rot = delta_rot * r_ee_base
        self.panda.send_cartesian_pose_command(
            pos=desired_position.tolist(), rot=desired_rot.as_quat().tolist()
        )
        self.gripper.send_grasp_command(
            position=right_data["index_trigger"],
            speed=0.7,
            force=0.3,
            blocking=False,
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
    init_xr_node_manager("MQ3", "192.168.0.117")
    mq3 = MetaQuest3PandaController(
        "IRL-MQ3-2", robot
    )  # You can change the name by using simpubweb
    robot.connect()
    # pyzlc.register_subscriber_handler("FrankaPanda/franka_arm_state", print_state)
    robot.set_franka_arm_control_mode(ControlMode.CartesianImpedance)
    try:
        while True:
            mq3.update()
            pyzlc.sleep(0.01)
    except KeyboardInterrupt:
        pass
