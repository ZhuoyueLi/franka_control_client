from __future__ import annotations

import threading
import time
import traceback
from typing import Union

import numpy as np
import torch
import pyzlc

from franka_control_client.control_pair.policy_panda_control_pair import PolicyPandaControlPair

from ..franka_robot.panda_robotiq import PandaRobotiq

from .mq3_panda_control_pair import MQ3PandaControlPair

from ..vr.meta_quest3 import MQ3Controller

from .control_pair import ControlPair
from ..franka_robot.panda_arm import ControlMode, RemotePandaArm
from ..franka_robot.panda_gripper import RemotePandaGripper
from ..robotiq_gripper.robotiq_gripper import RemoteRobotiqGripper
from .cartesian_policy_panda_control_pair import (
    CartesianPolicyPandaControlPair,
)

DEFAULT_CONTROL_HZ: float = 1000
GRIPPER_DEADBAND: float = 1e-3
GRIPPER_SPEED = 0.7
GRIPPER_FORCE = 0.3
ACTION_LOG_INTERVAL_S: float = 0.5
GRIPPER_TOGGLE_WARN_WINDOW_S: float = 3.0
GRIPPER_TOGGLE_WARN_COUNT: int = 6
DEFAULT_POSITION = (0.0, 0.0, 0.0, -2.15, 0.0, 2.15, 0.0)

# Calculate velocity limits using the standard approach from training
VELOCITY_LIMITS = np.array([[-4 * np.pi / 2, 4 * np.pi / 2]] * 7).T / 32
VELOCITY_LIMITS_NORM = np.linalg.norm(VELOCITY_LIMITS)


class PILPandaControlPair(CartesianPolicyPandaControlPair):
    """
    Apply policy actions to a Panda arm with a gripper.

    Action semantics: [q0..q6, gripper] (joint positions + gripper).
    Gripper value is normalized in [0, 1]. It is scaled to device range.
    Includes velocity and acceleration limiting for safety.
    """

    def __init__(
        self,
        panda_arm: RemotePandaArm,
        gripper: Union[RemotePandaGripper, RemoteRobotiqGripper],
        control_hz: float = DEFAULT_CONTROL_HZ,
    ) -> None:
        super().__init__(panda_arm, gripper, control_hz)
        self.policy_pair = self
        self.panda_arm = panda_arm
        self.gripper = gripper
        self.mq3_controller = MQ3Controller(
            "IRL-MQ3-2", "192.168.0.117", panda_arm
        )

        self.interrupt_control_pair = MQ3PandaControlPair(
            self.mq3_controller, PandaRobotiq("MQ3Panda", panda_arm, gripper)
        )

        self.current_control_pair: ControlPair = self.policy_pair
        self.control_pair_lock = threading.Lock()

        self.mq3_controller.mq3.register_trigger_press_event("hand_trigger", "right", self.switch_to_interrupt_control)
        self.mq3_controller.mq3.register_trigger_release_event("hand_trigger", "right", self.switch_to_policy_control)

    def switch_to_policy_control(self) -> None:
        with self.control_pair_lock:
            self.current_control_pair = self.policy_pair
        print("Switched to policy control")

    def switch_to_interrupt_control(self) -> None:
        with self.control_pair_lock:
            self.current_control_pair = self.interrupt_control_pair
        print("Switched to interrupt control")

    def _control_task(self) -> None:
        try:
            self.control_reset()
            while self.is_running:
                start = time.perf_counter()
                with self.control_pair_lock:
                    self.current_control_pair.control_step()
                # end_time = time.perf_counter()
                # print(f"Control step took {end_time - start:.3f} seconds")
                if time.perf_counter() - start < (1.0 / self.control_hz):
                    pyzlc.sleep(
                        (1.0 / self.control_hz) - (time.perf_counter() - start)
                    )
            self.control_end()
        except Exception as e:
            print(f"Control task encountered an error: {e}")
            traceback.print_exc()

    def control_reset(self) -> None:
        self.mq3_controller.reset()
        super().control_reset()
        self.interrupt_control_pair.control_reset()

    def control_end(self) -> None:
        super().control_end()
        self.interrupt_control_pair.control_end()

    def reset_action(self):
        super().reset_action()