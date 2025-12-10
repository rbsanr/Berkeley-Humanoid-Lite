#!/usr/bin/env python
"""Quick test script for train_mujoco.py"""

import sys
import torch
import numpy as np
from omegaconf import OmegaConf

# Add to path
sys.path.insert(0, '/home/josh/Berkeley-Humanoid-Lite')

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg

print("Loading config...")
cfg = Cfg.from_arguments()

if not cfg:
    raise ValueError("Failed to load config.")

print("Initializing environment...")
env = MujocoSimulator(cfg)
raw_obs = env.reset()

print(f"Raw observation shape: {raw_obs.shape}")
print(f"Raw observation: {raw_obs[:10]}...")  # First 10 elements

print("\nInitializing controller...")
controller = RlController(cfg)

class DummyPolicy:
    def forward(self, obs):
        return np.zeros(len(cfg.action_indices), dtype=np.float32)

controller.policy = DummyPolicy()

print("Processing first observation...")
_ = controller.update(raw_obs.numpy())
obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)

print(f"Processed observation shape: {obs.shape}")
print(f"Observation dimension: {obs.shape[0]}")
print(f"Action dimension: {len(cfg.action_indices)}")

print("\nTesting a few environment steps...")
for i in range(5):
    # Random actions
    actions = torch.randn(len(cfg.action_indices)) * 0.1

    # Scale to joint positions
    if cfg.num_actions == cfg.num_joints:
        default_joint_positions = torch.tensor(cfg.default_joint_positions, dtype=torch.float32)
    else:
        default_joint_positions = torch.tensor(cfg.default_joint_positions[10:], dtype=torch.float32)

    scaled_actions = actions * cfg.action_scale + default_joint_positions

    print(f"Step {i+1}: action shape {actions.shape}, scaled shape {scaled_actions.shape}")

    raw_obs = env.step(scaled_actions)
    print(f"  Raw obs shape: {raw_obs.shape}")

    # Test reward computation
    from scripts.sim2sim.train_mujoco import compute_reward
    reward = compute_reward(env, raw_obs, actions, None)
    print(f"  Reward: {reward:.4f}")

print("\n✓ Test passed! Basic functionality works.")
