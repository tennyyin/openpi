"""Policy transforms for the TRI/LBM BimanualBikeRotorInstall task (bimanual dual-Panda).

Input contract (after the repack transform in the data config):
  - observation/image              third-person view  (base_0_rgb)   <- scene_right_0
  - observation/left_wrist_image   left wrist view    (left_wrist)   <- wrist_left_plus
  - observation/right_wrist_image  right wrist view   (right_wrist)  <- wrist_right_plus
  - observation/state              16-d measured joint state
  - actions                        20-d cartesian xyzrot6g command   (training only)
  - prompt                         language instruction

Actions are the native LBM ``xyzrot6g`` command (per arm: xyz(3) + rot_6d(6) + gripper(1)),
i.e. absolute end-effector targets -- 20 dims total. We do NOT apply a delta-to-state
transform: the state is joint-space and the action is task-space, so a per-step delta
against the current state is not meaningful. Normalization (computed by
scripts/compute_norm_stats.py) handles the differing per-dimension scales.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

BIKE_ROTOR_ACTION_DIM = 20  # cartesian xyzrot6g, bimanual
BIKE_ROTOR_STATE_DIM = 16   # measured joint state, bimanual


def make_bike_rotor_example() -> dict:
    """Random input example matching the (post-repack) inference contract."""
    return {
        "observation/state": np.random.rand(BIKE_ROTOR_STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "install the rotor in the bike wheel and secure it by using a tool",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    # LeRobot stores/serves images as float32 (C,H,W); inference passes uint8 (H,W,C).
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BikeRotorInputs(transforms.DataTransformFn):
    """Map the bike-rotor dataset/inference dict into the model's expected input format."""

    # Determines which model will be used (pi0 vs pi0-FAST changes wrist masking).
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist = _parse_image(data["observation/left_wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])

        # This is a true 3-camera setup, so every slot is present (all masks True).
        mask_absent = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        _ = mask_absent  # both wrists exist here; kept for parity with other policies.

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training. Padding to the model action dim
        # is handled downstream by PadStatesAndActions in the model transforms.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BikeRotorOutputs(transforms.DataTransformFn):
    """Return the first 20 action dims (the rest is model padding). Inference only."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :BIKE_ROTOR_ACTION_DIM])}
