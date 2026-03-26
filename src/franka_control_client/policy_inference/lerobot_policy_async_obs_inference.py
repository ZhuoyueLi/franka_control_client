from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyzlc
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device

from .policy_inference_manager import PolicyInferenceManager
from .async_wrapper import (
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
    observation_history_lengths: Optional[Dict[str, int]] = None
    observation_buffer_capacities: Optional[Dict[str, int]] = None
    pad_history_with_oldest: bool = True


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

        (
            self._expected_image_shapes,
            self._expected_image_history_lengths,
        ) = self._get_expected_image_specs()
        (
            self._expected_state_dim,
            self._expected_state_history_length,
        ) = self._get_expected_state_spec()
        self._state_history_length = self._resolve_state_history_length()
        self._image_history_lengths = self._resolve_image_history_lengths()
        self._configure_observation_buffers()

        # Auto-hook control start/stop to inference events.
        self.register_start_infering_event(self.control_pair.start_control_pair)
        self.register_stop_infering_event(self.control_pair.stop_control_pair)
        self._debug_image_dir = Path("debug/inference_start_images")

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

    def _get_expected_image_specs(
        self,
    ) -> tuple[dict[str, tuple[int, int, int]], dict[str, int]]:
        """Get expected image shapes and history lengths from policy config."""
        shapes: dict[str, tuple[int, int, int]] = {}
        history_lengths: dict[str, int] = {}
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
                    history_lengths[key] = 1
                elif len(shape) == 4:
                    history_lengths[key] = int(shape[0])
                    shapes[key] = (shape[1], shape[2], shape[3])
        if shapes:
            return shapes, history_lengths

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
                    history_lengths[key] = 1
                elif len(shape) == 4:
                    history_lengths[key] = int(shape[0])
                    shapes[key] = (shape[1], shape[2], shape[3])
        return shapes, history_lengths

    def _get_expected_state_spec(self) -> tuple[Optional[int], int]:
        """Get expected state dimension and history length from policy config."""
        input_feats = getattr(self.train_cfg.policy, "input_features", None)
        if isinstance(input_feats, dict) and "observation.state" in input_feats:
            try:
                shape = input_feats["observation.state"].shape
                if len(shape) == 1:
                    return int(shape[-1]), 1
                if len(shape) >= 2:
                    return int(shape[-1]), int(shape[0])
            except Exception:
                return None, 1
        return None, 1

    def _resolve_state_history_length(self) -> int:
        config_lengths = self.cfg.observation_history_lengths or {}
        if "observation.state" in config_lengths:
            return max(1, int(config_lengths["observation.state"]))
        if "state" in config_lengths:
            return max(1, int(config_lengths["state"]))
        return max(1, self._expected_state_history_length)

    def _resolve_image_history_lengths(self) -> Dict[str, int]:
        config_lengths = self.cfg.observation_history_lengths or {}
        if "camera" in config_lengths:
            default_history = max(1, int(config_lengths["camera"]))
        elif self._expected_image_history_lengths:
            default_history = max(1, max(self._expected_image_history_lengths.values()))
        else:
            default_history = 1

        resolved: Dict[str, int] = {}
        for cam in self.cameras:
            resolved[cam.hw_name] = max(
                1, int(config_lengths.get(cam.hw_name, default_history))
            )
        return resolved

    def _configure_observation_buffers(self) -> None:
        configured_capacities = self.cfg.observation_buffer_capacities or {}

        state_capacity = max(
            self._state_history_length,
            int(
                configured_capacities.get(
                    "observation.state",
                    configured_capacities.get("state", 1),
                )
            ),
        )
        self.arm_wrapper.set_buffer_capacity(
            max(self.arm_wrapper.buffer_capacity, state_capacity)
        )
        self.gripper_wrapper.set_buffer_capacity(
            max(self.gripper_wrapper.buffer_capacity, state_capacity)
        )

        if "camera" in configured_capacities:
            default_image_capacity = max(1, int(configured_capacities["camera"]))
        elif self._image_history_lengths:
            default_image_capacity = max(1, max(self._image_history_lengths.values()))
        else:
            default_image_capacity = 1

        for cam in self.cameras:
            requested_capacity = max(
                self._image_history_lengths.get(cam.hw_name, 1),
                int(configured_capacities.get(cam.hw_name, default_image_capacity)),
            )
            cam.set_buffer_capacity(max(cam.buffer_capacity, requested_capacity))

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

    def _images_to_tensor(self, images: List[np.ndarray]) -> torch.Tensor:
        """Match real_robot_sim image tensor construction with optional history."""
        stacked = torch.stack(
            [
                torch.from_numpy(image.copy()).float().permute(2, 0, 1) / 255.0
                for image in images
            ],
            dim=0,
        )
        return stacked.unsqueeze(0)

    def _get_recent_samples(
        self, wrapper: IRL_HardwareDataWrapper, count: int
    ) -> List[Any]:
        if wrapper.buffered_length == 0:
            wrapper.update_buffer()
        return wrapper.get_recent(
            count,
            pad_with_oldest=self.cfg.pad_history_with_oldest,
        )

    def _build_observation(self) -> Dict[str, Any]:
        """Build observation dict from hardware data."""
        state_vec = self._build_state_vector()
        images = self._build_images()
        
        state = np.asarray(state_vec, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        elif state.ndim == 2:
            state = state[None, :, :]
        if self._expected_state_dim is not None and state.shape[-1] != self._expected_state_dim:
            if state.shape[-1] > self._expected_state_dim:
                state = state[..., : self._expected_state_dim]
            else:
                pad = self._expected_state_dim - state.shape[-1]
                if state.ndim == 2:
                    state = np.pad(state, ((0, 0), (0, pad)), mode="constant")
                else:
                    state = np.pad(state, ((0, 0), (0, 0), (0, pad)), mode="constant")
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
            frame_history = cam_img if isinstance(cam_img, list) else [cam_img]
            rgb_history = []
            for frame in frame_history:
                rgb = self._decode_image(frame)
                if rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError(
                        f"Expected HWC image with 3 channels for {obs_key}, got shape {rgb.shape}"
                    )
                if obs_key in self._expected_image_shapes:
                    rgb = self._resize_image(
                        np.ascontiguousarray(rgb), self._expected_image_shapes[obs_key]
                    )
                rgb_history.append(np.ascontiguousarray(rgb))
            observation[obs_key] = self._images_to_tensor(rgb_history)

        observation["task"] = self.task

        return observation

    def _extract_arm_joints(self, arm_state: Any) -> np.ndarray:
        q = None
        if isinstance(arm_state, dict):
            if "q" in arm_state:
                q = np.asarray(arm_state["q"], dtype=np.float32).reshape(-1)
            elif "joint_state" in arm_state:
                q = np.asarray(arm_state["joint_state"], dtype=np.float32).reshape(-1)
        if q is None or q.size != 7:
            raise ValueError("Arm state missing valid joint positions.")
        return q

    def _extract_gripper_value(self, grip_state: Any) -> float:
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
        return gripper_val

    def _build_state_vector(self) -> np.ndarray:
        arm_states = self._get_recent_samples(self.arm_wrapper, self._state_history_length)
        grip_states = self._get_recent_samples(
            self.gripper_wrapper, self._state_history_length
        )

        state_history = []
        for arm_state, grip_state in zip(arm_states, grip_states):
            q = self._extract_arm_joints(arm_state)
            gripper_val = self._extract_gripper_value(grip_state)
            state_history.append(
                np.concatenate([q, np.asarray([gripper_val], dtype=np.float32)])
            )

        state = np.stack(state_history, axis=0)
        if self._state_history_length == 1 and self._expected_state_history_length <= 1:
            return state[0]
        return state

    def _build_images(self) -> Dict[str, Any]:
        images: Dict[str, Any] = {}
        for cam in self.cameras:
            frames = self._get_recent_samples(
                cam, self._image_history_lengths.get(cam.hw_name, 1)
            )
            if not frames:
                continue
            images[cam.hw_name] = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    h, w, c = frame.shape
                    images[cam.hw_name].append(
                        {
                            "height": int(h),
                            "width": int(w),
                            "channels": int(c),
                            "rgb_data": frame.tobytes(),
                        }
                    )
                else:
                    images[cam.hw_name].append(frame)
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
                frame_history = images[cam_name]
                frame = frame_history[-1] if isinstance(frame_history, list) else frame_history
                rgb = self._decode_image(frame)
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
        for collector in self.data_collectors:
            collector.start_buffering(clear_existing=True)
        #debug
        # self._check_startup_image()
        # self.last_timestamp = None
        super()._start_infering()

    def _infer_step(self) -> None:
        # if self.last_timestamp is None:
        #     self.last_timestamp = time.perf_counter()
        start_time = time.perf_counter()
        # Build observation from hardware
        observation = self._build_observation()
        
        try:
            # Preprocess observation
            observation = self.preprocessor(observation)
        except Exception as exc:
            image_shapes = {
                k: tuple(v.shape)
                for k, v in observation.items()
                if str(k).startswith("observation.images.") and hasattr(v, "shape")
            }
            raise RuntimeError(
                f"Preprocessor failed. image_shapes={image_shapes}, state_shape={tuple(observation['observation.state'].shape)}"
            ) from exc
        
        # Evaluate policy and postprocess each action in the predicted chunk.
        with torch.inference_mode():
        ###single action
        #       action = self.policy.select_action(observation)
        # action = action[:, :8]
        
        # # Postprocess action
        # action = self.postprocessor(action).float().cpu().numpy()
        # # print("post action:",action)
        
        # action_vec = action[0] if action.ndim == 2 else action

        ###action chunk
            action_chunk = self.policy.predict_action_chunk(observation)

        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(1)
        elif action_chunk.ndim != 3:
            raise RuntimeError(
                f"Expected action_chunk to have shape (B, T, D) or (B, D), got {tuple(action_chunk.shape)}"
            )

        batch_size, chunk_size, action_dim = action_chunk.shape
        action_dim_expected = 8  # 7 joints + 1 gripper
        post_action_chunk = torch.zeros((batch_size, chunk_size, action_dim_expected), dtype=torch.float32)
        for chunk_idx in range(chunk_size):
            single_action = action_chunk[:, chunk_idx, :]
            single_action = single_action[:, :8]
            processed_action = self.postprocessor(single_action)
            # pyzlc.info(f"Processed action chunk {chunk_idx}: {processed_action.float().cpu().numpy()}")
            post_action_chunk[:, chunk_idx, :] = processed_action

        post_action_chunk = post_action_chunk.float().cpu().numpy()
        for idx in range(len(post_action_chunk)):
            pyzlc.info(f"Postprocessed action chunk for batch {idx}: {post_action_chunk[idx]}")
        try:
            #single_action
            # self.control_pair.update_action(action_vec)
            #action chunk
            self.control_pair.update_action_chunk(post_action_chunk)
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
        for collector in self.data_collectors:
            collector.stop_buffering()
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

    def _close(self) -> None:
        for collector in self.data_collectors:
            try:
                collector.close()
            except Exception as exc:
                pyzlc.error(f"Failed to close collector {collector.hw_name}: {exc}")
        super()._close()
