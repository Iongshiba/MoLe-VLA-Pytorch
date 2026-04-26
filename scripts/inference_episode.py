#!/usr/bin/env python3
"""
Run CogACT VLA model inference on all steps of an RLBench episode,
save predicted 7‑DoF actions (and ground truth) for later replay.
"""

import os
import sys

# Add path to LIFT3D so we can import RLBenchDataset
sys.path.insert(0, "/home/longshiba/projects/LIFT3D")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import zarr
from PIL import Image
from vla import load_vla
from scipy.spatial.transform import Rotation as R

# Only import RLBenchDataset from Lift3D – no graphics helpers needed
from lift3d.dataset import RLBenchDataset

# ----------------------------------------------------------------------
# Local replacement for Quaternion.ensure_positive_real_part
# ----------------------------------------------------------------------
def ensure_positive_real_part(quat, scalar_first=False):
    """
    Ensure the real part of the quaternion is positive.
    quat: np.array of shape (4,) – [qx, qy, qz, qw] if scalar_first=False,
          else [qw, qx, qy, qz].
    Returns the quaternion possibly flipped in sign.
    """
    q = np.array(quat)
    real_part = q[0] if scalar_first else q[-1]
    if real_part < 0:
        q = -q
    return q

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATASET_ZARR = "/home/longshiba/projects/LIFT3D/data/rlbench/close_box.zarr"   # your Zarr file path
EPISODE_INDEX = 0                              # which episode to evaluate
PROMPT = None                                  # if None, use the stored text
MODEL_NAME = "CogACT/CogACT-Base"              # or local path
ACTION_MODEL_TYPE = "DiT-B"
FUTURE_ACTION_WINDOW = 15
DEVICE = "cuda:0"

# ----------------------------------------------------------------------
# Load VLA model
# ----------------------------------------------------------------------
model = load_vla(
    MODEL_NAME,
    load_for_training=False,
    action_model_type=ACTION_MODEL_TYPE,
    future_action_window_size=FUTURE_ACTION_WINDOW,
)
model.vlm = model.vlm.to(torch.bfloat16)
model.to(DEVICE).eval()
print(f"Model Normalize Stats: {model.norm_stats.keys()}")
# ----------------------------------------------------------------------
# Load dataset and extract the chosen episode
# ----------------------------------------------------------------------
dataset = RLBenchDataset(DATASET_ZARR, split="custom", custom_split_size=EPISODE_INDEX+1)

# We need to know where the episode starts and ends
zarr_root = zarr.open_group(DATASET_ZARR, mode="r")
episode_ends = zarr_root["meta"]["episode_ends"][:]
start = 0 if EPISODE_INDEX == 0 else episode_ends[EPISODE_INDEX-1]
end = episode_ends[EPISODE_INDEX]

# Extract images, ground truth actions, and text for this episode
images = []          # list of PIL Images
gt_actions = []      # ground truth actions (already in RLBench format)
text = None

for idx in range(start, end):
    # Load image (dataset stores CHW, we need HWC)
    img_np = dataset._images[idx].transpose(1,2,0).astype(np.uint8)  # (H,W,3)
    images.append(Image.fromarray(img_np))
    
    # Ground truth action (8‑DoF: x,y,z,qx,qy,qz,qw,gripper)
    act = dataset._actions[idx]
    gt_actions.append(act)
    
    if text is None:
        text = dataset._texts[idx]   # language description

# Use stored prompt if none provided
if PROMPT is None:
    PROMPT = text
print(f"Task description: {PROMPT}")

# ----------------------------------------------------------------------
# Run inference on each image
# ----------------------------------------------------------------------
predicted_actions = []   # will store 8‑DoF actions in RLBench quaternion format

for i, img in enumerate(images):
    print(f"Step {i+1}/{len(images)}")
    # Predict actions (returns shape [future_action_window, 7])
    actions_7dof, _ = model.predict_action(
        img,
        PROMPT,
        unnorm_key="dataset/10tasks_selected_keyframe_state/rlbench/1.0.0/dataset_statistics_718c0cb69ab23a5355adb5c99f8026dd006399eb232748f52fc088768569c6c2.json",   # adjust to your dataset norm
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10,
    )
    # actions_7dof: [15, 7] – position (3) + euler angles (3) + gripper (1)
    # Take the first predicted action (step 0)
    first_action = actions_7dof[0]
    
    predicted_actions.append(first_action)

# ----------------------------------------------------------------------
# Save predicted actions and ground truth for replay
# ----------------------------------------------------------------------
out_dir = f"results/inference_episode_{EPISODE_INDEX}"
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, "predicted_actions.npy"), np.array(predicted_actions))
np.save(os.path.join(out_dir, "ground_truth_actions.npy"), np.array(gt_actions))

print(f"Saved predicted actions ({len(predicted_actions)} steps) to {out_dir}")