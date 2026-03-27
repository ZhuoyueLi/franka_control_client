from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import matplotlib
import numpy as np
import pyzlc
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .policy_inference_manager import PolicyInferenceManager
from .irl_wrapper import (
    IRL_HardwareDataWrapper,
    ImageDataWrapper,
    PandaArmDataWrapper,
    PandaGripperDataWrapper,
    RobotiqGripperDataWrapper,
)
from ..control_pair.policy_panda_control_pair import PolicyPandaControlPair

# Try to import load_dataset_meta helper function
try:
    from utils.eval_utils import load_dataset_meta
except ImportError:
    load_dataset_meta = None


@dataclass
class LeRobotPolicyInferenceConfig:
    checkpoint_path: str
    task: str
    fps: int = 30
    device: str = "cuda"
    policy_dtype: Optional[str] = None
    dataset_path: Optional[str] = None
    action_data_path: Optional[str] = None


class LeRobotPolicyInference(PolicyInferenceManager):
    """
    Policy inference loop that directly loads and evaluates a LeRobot policy.

    Observation format: {"state": [...], "images": {...}, "task": "..."}
    Action semantics: [q0..q6, gripper].
    """

    def __init__(
        self,
        data_collectors: List[IRL_HardwareDataWrapper],
        control_pair: PolicyPandaControlPair,
        cfg: LeRobotPolicyInferenceConfig,
    ) -> None:
        super().__init__(task=cfg.task, fps=cfg.fps)
        self.data_collectors = data_collectors
        self.control_pair = control_pair
        self.cfg = cfg

        self.cameras: List[ImageDataWrapper] = []
        self.arm_wrapper: Optional[PandaArmDataWrapper] = None
        self.gripper_wrapper: Optional[
            IRL_HardwareDataWrapper
        ] = None
        for hw in data_collectors:
            if isinstance(hw, ImageDataWrapper) or hw.hw_type == "camera":
                self.cameras.append(hw)  # type: ignore[arg-type]
            elif isinstance(hw, PandaArmDataWrapper) or hw.hw_type == "follower_arm":
                self.arm_wrapper = hw  # type: ignore[assignment]
            elif isinstance(hw, (PandaGripperDataWrapper, RobotiqGripperDataWrapper)) or hw.hw_type == "follower_gripper":
                self.gripper_wrapper = hw

        if self.arm_wrapper is None:
            raise ValueError("Missing PandaArmDataWrapper for inference.")
        if self.gripper_wrapper is None:
            raise ValueError("Missing gripper wrapper for inference.")

        # Load policy stack directly
        self.train_cfg = self._load_train_cfg()
        pyzlc.info(f"Loaded train config: {self.train_cfg}")
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_stack()

        self._expected_image_shapes = self._get_expected_image_shapes()
        self._expected_state_dim = self._get_expected_state_dim()

        # Auto-hook control start/stop to inference events.
        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)
        self._debug_image_dir = Path("debug/inference_start_images")
        self._debug_action_dir = Path("debug/action_chunk")
        self._loaded_action_chunk = self._load_action_chunk_from_pt_files()
        self._dummy_action_sent = False

    def _load_train_cfg(self) -> TrainPipelineConfig:
        """Load training config from checkpoint."""
        cli_args = [
            f"--policy.pretrained_path={self.cfg.checkpoint_path}",
            f"--policy.device={self.cfg.device}",
            "--dataset.image_transforms.enable=false",
        ]
        if self.cfg.dataset_path:
            cli_args.append(f"--dataset.root={self.cfg.dataset_path}")
        if self.cfg.policy_dtype:
            cli_args.append(f"--policy.dtype={self.cfg.policy_dtype}")

        train_cfg = TrainPipelineConfig.from_pretrained(
            pretrained_name_or_path=self.cfg.checkpoint_path,
            cli_args=cli_args,
        )

        if any("empty_camera" in key for key in train_cfg.policy.input_features):
            train_cfg.policy.input_features = {
                "observation.images.image": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 256, 256)
                ),
                "observation.images.image2": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, 256, 256)
                ),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
            }
            train_cfg.policy.num_views = 2
            train_cfg.policy.empty_camera = 1

        return train_cfg

    def _load_dataset_meta(self) -> Any:
        """Load dataset metadata helper."""
        try:
            return load_dataset_meta(self.train_cfg)
        except Exception as exc:
            pyzlc.info(f"load_dataset_meta helper unavailable, proceeding without ds_meta: {exc}")
            return None

    def _load_policy_stack(self):
        """Load policy, preprocessor, and postprocessor."""
        ds_meta = self._load_dataset_meta()
        pyzlc.info(f"Loaded dataset meta: {ds_meta}")
        device = get_safe_torch_device(self.train_cfg.policy.device, log=True)

        policy = make_policy(
            cfg=self.train_cfg.policy,
            env_cfg=None,
            ds_meta=ds_meta,
            rename_map=getattr(self.train_cfg, "rename_map", None),
        )
        policy.eval()
        policy.to(device)
        pyzlc.info(f"Policy running on {device}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.train_cfg.policy,
            pretrained_path=self.train_cfg.policy.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": device.type}},
        )

        return policy, preprocessor, postprocessor

    def _decode_image(self, img: Any) -> np.ndarray:
        """Decode image from various formats."""
        if isinstance(img, np.ndarray):
            return np.ascontiguousarray(img)
        if isinstance(img, dict) and "rgb_data" in img:
            h = int(img["height"])
            w = int(img["width"])
            c = int(img.get("channels", 3))
            image_array = np.frombuffer(img["rgb_data"], dtype=np.uint8)
            return image_array.reshape((h, w, c)).copy()
        if isinstance(img, list):
            return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        raise ValueError("Unsupported image format in observation.")

    def _get_expected_image_shapes(self) -> dict[str, tuple[int, int, int]]:
        """Get expected image shapes from policy config."""
        shapes: dict[str, tuple[int, int, int]] = {}
        input_feats = getattr(self.train_cfg.policy, "input_features", None)
        if isinstance(input_feats, dict):
            for key, feat in input_feats.items():
                if not str(key).startswith("observation.images."):
                    continue
                try:
                    shape = tuple(feat.shape)
                except Exception:
                    continue
                if len(shape) == 3:
                    shapes[key] = (shape[0], shape[1], shape[2])
        if shapes:
            return shapes

        cfg = getattr(self.policy, "config", None)
        image_feats = getattr(cfg, "image_features", None)
        if isinstance(image_feats, dict):
            for key, feat in image_feats.items():
                try:
                    shape = tuple(feat.shape)
                except Exception:
                    continue
                if len(shape) == 3:
                    shapes[key] = (shape[0], shape[1], shape[2])
        return shapes

    def _get_expected_state_dim(self) -> Optional[int]:
        """Get expected state dimension from policy config."""
        input_feats = getattr(self.train_cfg.policy, "input_features", None)
        if isinstance(input_feats, dict) and "observation.state" in input_feats:
            try:
                shape = input_feats["observation.state"].shape
                if len(shape) >= 1:
                    return int(shape[-1])
            except Exception:
                return None
        return None

    def _resize_image(self, img: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        """Resize image to expected shape."""
        c, h, w = shape
        img = np.ascontiguousarray(img)
        if img.shape[:2] == (h, w):
            return img
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            img_t, size=(h, w), mode="bilinear", align_corners=False
        )
        out = resized.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        if out.shape[2] != c:
            raise ValueError(f"Image channels mismatch after resize: expected {c}, got {out.shape[2]}")
        return out

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Match real_robot_sim image tensor construction."""
        rgb = torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
        return rgb.unsqueeze(0).unsqueeze(0)

    def _resolve_pt_path(self, *candidate_names: str) -> Path:
        """Resolve a .pt file from the configured action-data directory."""
        search_root = (
            Path(self.cfg.action_data_path).expanduser()
            if self.cfg.action_data_path
            else Path.cwd()
        )
        for candidate in candidate_names:
            path = search_root / candidate
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find any of these files in {search_root}: {candidate_names}"
        )

    def _pt_to_numpy(self, value: Any, name: str) -> np.ndarray:
        """Convert torch-saved content to a float32 numpy array."""
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.size == 0:
            raise ValueError(f"{name} is empty.")
        return np.asarray(array, dtype=np.float32)

    def _save_joint_plot(self, joint_pos: np.ndarray, save_path: Path) -> None:
        """Save one plot with all 7 joint trajectories."""
        fig, ax = plt.subplots(figsize=(12, 6))
        for joint_idx in range(joint_pos.shape[1]):
            ax.plot(joint_pos[:, joint_idx], label=f"joint_{joint_idx}")
        ax.set_title("Joint Position Trajectories")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Joint Position")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _save_gripper_plot(self, gripper_command: np.ndarray, save_path: Path) -> None:
        """Save one plot with the gripper command trajectory."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(gripper_command, color="tab:red")
        ax.set_title("Gripper Command Trajectory")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Gripper Command")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _load_action_chunk_from_pt_files(self) -> torch.Tensor:
        """Load prerecorded joint and gripper actions and format them as (B, T, 8)."""
        joint_pos_path = self._resolve_pt_path("joint_pos.pt")
        gripper_command_path = self._resolve_pt_path(
            "gripper_command.pt",
            # "gripper command.pt",
        )

        joint_pos = self._pt_to_numpy(torch.load(joint_pos_path, map_location="cpu"), "joint_pos")
        gripper_command = self._pt_to_numpy(
            torch.load(gripper_command_path, map_location="cpu"),
            "gripper_command",
        )

        if joint_pos.ndim == 1:
            if joint_pos.size % 7 != 0:
                raise ValueError(
                    f"joint_pos must contain a multiple of 7 values, got shape {joint_pos.shape}"
                )
            joint_pos = joint_pos.reshape(-1, 7)
        elif joint_pos.ndim > 2:
            joint_pos = joint_pos.reshape(joint_pos.shape[0], -1)

        if joint_pos.shape[1] != 7:
            raise ValueError(f"joint_pos must have 7 joints per action, got shape {joint_pos.shape}")

        gripper_command = gripper_command.reshape(-1)

        joint_len = int(joint_pos.shape[0])
        gripper_len = int(gripper_command.shape[0])
        pyzlc.info(f"joint_pos length: {joint_len}")
        pyzlc.info(f"gripper_command length: {gripper_len}")
        print(f"joint_pos length: {joint_len}")
        print(f"gripper_command length: {gripper_len}")

        if joint_len != gripper_len:
            aligned_len = min(joint_len, gripper_len)
            pyzlc.error(
                "joint_pos and gripper_command lengths do not match. "
                f"Truncating to {aligned_len} steps."
            )
            joint_pos = joint_pos[:aligned_len]
            gripper_command = gripper_command[:aligned_len]

        self._debug_action_dir.mkdir(parents=True, exist_ok=True)
        self._save_joint_plot(joint_pos, self._debug_action_dir / "joint_pos.png")
        self._save_gripper_plot(
            gripper_command, self._debug_action_dir / "gripper_command.png"
        )

        action_chunk = np.concatenate(
            [joint_pos, gripper_command[:, None]],
            axis=1,
        )
        pyzlc.info(f"Combined action_chunk shape: {action_chunk.shape}")

        return torch.from_numpy(action_chunk).unsqueeze(0)

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation dict from hardware data."""
        state_vec = self._build_state_vector()
        images = self._build_images()
        
        state = np.asarray(state_vec, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if self._expected_state_dim is not None and state.shape[-1] != self._expected_state_dim:
            if state.shape[-1] > self._expected_state_dim:
                state = state[..., : self._expected_state_dim]
            else:
                pad = self._expected_state_dim - state.shape[-1]
                state = np.pad(state, ((0, 0), (0, pad)), mode="constant")
            pyzlc.info(f"Adjusted state dim to {self._expected_state_dim} (now {state.shape[-1]}).")

        observation: Dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
        }

        if not isinstance(images, dict):
            raise ValueError("'images' must be a dict keyed by camera name.")

        right_img = images.get("right_cam") or images.get("zed_right")
        wrist_img = images.get("wrist_cam") or images.get("zed_wrist")
        expected_image_keys = list(self._expected_image_shapes.keys())
        if right_img is not None and wrist_img is not None:
            mapped = {
                "observation.images.image": right_img,
                "observation.images.image2": wrist_img,
            }
        elif not expected_image_keys:
            mapped = {f"observation.images.{k}": v for k, v in images.items()}
        else:
            image_namespaced = {f"observation.images.{k}" for k in images.keys()}
            if set(expected_image_keys).issubset(image_namespaced):
                mapped = {
                    key: images[key.replace("observation.images.", "", 1)]
                    for key in expected_image_keys
                }
            elif len(images) == len(expected_image_keys):
                mapped = dict(zip(expected_image_keys, images.values()))
            elif len(images) == 1 and len(expected_image_keys) == 1:
                mapped = {expected_image_keys[0]: next(iter(images.values()))}
            else:
                raise ValueError(
                    f"Image keys mismatch. Expected {expected_image_keys}, got {list(images.keys())}"
                )

        for obs_key, cam_img in mapped.items():
            # print(f"Processing image for {obs_key} with raw shape {cam_img['height']}x{cam_img['width']}x{cam_img.get('channels', 'unknown')}")
            rgb = self._decode_image(cam_img)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Expected HWC image with 3 channels for {obs_key}, got shape {rgb.shape}")
            observation[obs_key] = self._image_to_tensor(np.ascontiguousarray(rgb))

        observation["task"] = self.task

        return observation

    def _build_state_vector(self) -> np.ndarray:
        arm_state = self.arm_wrapper.capture_step()
        q = None
        if isinstance(arm_state, dict):
            if "q" in arm_state:
                q = np.asarray(arm_state["q"], dtype=np.float32).reshape(-1)
            elif "joint_state" in arm_state:
                q = np.asarray(arm_state["joint_state"], dtype=np.float32).reshape(-1)
        if q is None or q.size != 7:
            raise ValueError("Arm state missing valid joint positions.")

        grip_state = self.gripper_wrapper.capture_step()
        gripper_val = None
        if isinstance(grip_state, dict):
            if "width" in grip_state:
                gripper_val = float(grip_state["width"])
            elif "position" in grip_state:
                gripper_val = float(grip_state["position"])
            elif "gripper" in grip_state:
                gripper_arr = np.asarray(grip_state["gripper"], dtype=np.float32).reshape(-1)
                if gripper_arr.size > 0:
                    gripper_val = float(gripper_arr[0])
        if gripper_val is None:
            raise ValueError("Gripper state missing value.")

        return np.concatenate([q, np.asarray([gripper_val], dtype=np.float32)])

    def _build_images(self) -> Dict[str, Any]:
        images: Dict[str, Any] = {}
        for cam in self.cameras:
            frame = cam.capture_step()
            if frame is None:
                continue
            if isinstance(frame, np.ndarray):
                h, w, c = frame.shape
                images[cam.hw_name] = {
                    "height": int(h),
                    "width": int(w),
                    "channels": int(c),
                    "rgb_data": frame.tobytes(),
                }
            else:
                images[cam.hw_name] = frame
        return images

    def _check_startup_image(self) -> None:
        """Capture and persist one startup image per camera for inspection."""
        images = self._build_images()
        if not images:
            pyzlc.error("Startup image check skipped: no camera frames available.")
            return

        self._debug_image_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_images: list[Path] = []
        for cam_name in sorted(images.keys()):
            try:
                rgb = self._decode_image(images[cam_name])
            except Exception as exc:
                pyzlc.error(f"Startup image check failed for {cam_name}: {exc}")
                continue

            if rgb.ndim != 3 or rgb.shape[2] != 3:
                pyzlc.error(
                    f"Startup image check skipped for {cam_name}: unexpected shape {rgb.shape}"
                )
                continue

            image_path = (self._debug_image_dir / f"{timestamp}_{cam_name}.png").resolve()
            image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(image_path), image_bgr):
                pyzlc.error(f"Failed to save startup image check to {image_path}")
                continue

            saved_images.append(image_path)
            pyzlc.info(
                f"Startup image check saved {cam_name} frame to {image_path} with shape {rgb.shape}"
            )

        if not saved_images:
            pyzlc.error("Startup image check failed: no camera images were saved.")
            return

        pyzlc.info(
            "Inspect startup images to verify the inference camera inputs: "
            + ", ".join(str(path) for path in saved_images)
        )

    def _start_infering(self) -> None:
        # Reset action state for new episode
        self.control_pair.reset_action()
        self._dummy_action_sent = False
        #debug
        # self._check_startup_image()
        # self.last_timestamp = None
        super()._start_infering()

    def _infer_step(self) -> None:
        if self._dummy_action_sent:
            time.sleep(max(0.001, 1.0 / self.fps))
            return

        # if self.last_timestamp is None:
        #     self.last_timestamp = time.perf_counter()
        start_time = time.perf_counter()

        action_chunk = self._loaded_action_chunk
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(1)
        elif action_chunk.ndim != 3:
            raise RuntimeError(
                f"Expected action_chunk to have shape (B, T, D) or (B, D), got {tuple(action_chunk.shape)}"
            )

        post_action_chunk = action_chunk.float().cpu().numpy()
        try:
            self.control_pair.update_action_chunk(post_action_chunk)
            self._dummy_action_sent = True
            pyzlc.info(
                f"Sent dummy action chunk once with shape {post_action_chunk.shape}"
            )
        except Exception as exc:
            pyzlc.error(f"Failed to apply policy action: {exc}")
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        # print(f"Inference step took {elapsed:.4f} seconds.")
       
        sleep_time = max(0.0, (1.0 / self.fps) - elapsed)
        if sleep_time > 0.001:
            time.sleep(sleep_time)
            # print(f"Inference step took {elapsed:.5f} seconds, slept for {sleep_time:.5f} seconds to maintain {self.fps} FPS.")

    def _save_episode(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode saved.")

    def _discard_infering(self) -> None:
        self._stop_infering()
        self._ui_console.log("Episode discarded.")

    def _stop_infering(self) -> None:
        super()._stop_infering()

    def _reset_arm(self) -> None:
        """Reset the robot arm to a safe/home position.
        
        Called only when in WAITING state (control pair is not running).
        """
        self._ui_console.log("Resetting robot arm position...")
        try:
            # Reset the arm to home position
            self.control_pair.go_home()
            time.sleep(3)  # Wait for the arm to reach the home position
            self._ui_console.log("Robot arm reset to home position.")
        except Exception as exc:
            self._ui_console.log(f"Failed to reset arm: {exc}")
