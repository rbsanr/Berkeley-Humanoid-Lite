# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.
"""
Fine-tuning script for Berkeley Humanoid Lite using MuJoCo simulation.

This script fine-tunes a pre-trained policy using PPO algorithm in MuJoCo environment.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController, OnnxPolicy
from berkeley_humanoid_lite.environments import MujocoSimulator, Cfg


class PPOBuffer:
    """Buffer for storing trajectories for PPO training."""
    
    def __init__(self, obs_dim: int, act_dim: int, size: int, gamma: float = 0.99, lam: float = 0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
    
    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, val: float, logp: float):
        """Store one timestep of agent-environment interaction."""
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    def finish_path(self, last_val: float = 0):
        """Compute advantages and returns for a completed trajectory."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute rewards-to-go (targets for value function)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer."""
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        
        return {
            'obs': torch.as_tensor(self.obs_buf, dtype=torch.float32),
            'act': torch.as_tensor(self.act_buf, dtype=torch.float32),
            'ret': torch.as_tensor(self.ret_buf, dtype=torch.float32),
            'adv': torch.as_tensor(self.adv_buf, dtype=torch.float32),
            'logp': torch.as_tensor(self.logp_buf, dtype=torch.float32),
        }
    
    @staticmethod
    def _discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
        """Compute discounted cumulative sums."""
        result = np.zeros_like(x)
        result[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            result[t] = x[t] + discount * result[t + 1]
        return result


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int] = [256, 256, 256]):
        super().__init__()
        
        # Actor network
        actor_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            actor_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ELU(),
            ])
            prev_size = hidden_size
        actor_layers.append(nn.Linear(prev_size, act_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network
        critic_layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            critic_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ELU(),
            ])
            prev_size = hidden_size
        critic_layers.append(nn.Linear(prev_size, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic."""
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def act(self, obs: torch.Tensor) -> Tuple[np.ndarray, float, float]:
        """Sample action and compute log probability and value."""
        with torch.no_grad():
            action_mean, value = self.forward(obs)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), logp.cpu().item(), value.cpu().item()
    
    def evaluate(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions."""
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        logp = dist.log_prob(act).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, value.squeeze(-1), entropy


def compute_reward(robot: MujocoSimulator, raw_obs: torch.Tensor, actions: torch.Tensor,
                   prev_actions: torch.Tensor = None) -> float:
    """
    Compute reward matching Isaac Lab's reward structure.

    Based on BerkeleyHumanoidLiteEnvCfg rewards.
    """
    reward = 0.0

    # Parse observations (raw MuJoCo format)
    # Format: [base_quat(4), base_ang_vel(3), joint_pos(num_actions), joint_vel(num_actions), mode(1), cmd_vel(3)]
    base_quat = raw_obs[0:4]
    base_ang_vel = raw_obs[4:7]

    num_actions = len(actions) if actions is not None else 22  # Default to 22
    joint_pos = raw_obs[7:7+num_actions]
    joint_vel = raw_obs[7+num_actions:7+num_actions*2]
    command_vel_x = raw_obs[7+num_actions*2+1]
    command_vel_y = raw_obs[7+num_actions*2+2]
    command_vel_yaw = raw_obs[7+num_actions*2+3]

    # Get actual base linear velocity from MuJoCo data
    base_lin_vel = torch.tensor(robot.mj_data.qvel[0:3], dtype=torch.float32)
    base_lin_vel_x = base_lin_vel[0]
    base_lin_vel_y = base_lin_vel[1]
    base_lin_vel_z = base_lin_vel[2]

    # === Task-space performance rewards ===
    # 1. Track linear velocity XY (weight: 2.0)
    lin_vel_error = torch.sqrt((command_vel_x - base_lin_vel_x)**2 +
                               (command_vel_y - base_lin_vel_y)**2)
    track_lin_vel = 2.0 * torch.exp(-lin_vel_error / 0.5).item()
    reward += track_lin_vel

    # 2. Track angular velocity Z (weight: 2.0)
    ang_vel_error = torch.abs(command_vel_yaw - base_ang_vel[2])
    track_ang_vel = 2.0 * torch.exp(-ang_vel_error / 0.5).item()
    reward += track_ang_vel

    # === Basic behavior rewards ===
    # 3. Penalize vertical velocity (weight: -0.1)
    reward += -0.1 * (base_lin_vel_z**2).item()
    
    # 4. Penalize xy angular velocity (weight: -0.05)
    ang_vel_xy = torch.sum(base_ang_vel[0:2]**2).item()
    reward += -0.05 * ang_vel_xy
    
    # 5. Keep orientation flat/upright (weight: -3.0)
    # Quaternion [w, x, y, z] - for upright, x and y should be near 0
    flat_orientation = base_quat[1]**2 + base_quat[2]**2
    reward += -3.0 * flat_orientation.item()
    
    # 6. Action rate smoothness (weight: -0.001)
    if prev_actions is not None:
        action_rate = torch.sum((actions - prev_actions)**2).item()
        reward += -0.001 * action_rate
    
    # 7. Joint torques (weight: -2.0e-5)
    # Approximate torques from actions
    dof_torques = torch.sum(actions**2).item()
    reward += -2.0e-5 * dof_torques
    
    # 8. Joint acceleration (weight: -1.0e-7)
    dof_acc = torch.sum(joint_vel**2).item()
    reward += -1.0e-7 * dof_acc
    
    # 9. Joint position limits (weight: -1.0)
    # Penalize if joints exceed 80% of their range
    joint_pos_penalty = torch.sum(torch.clamp(torch.abs(joint_pos) - 0.8, min=0.0)).item()
    reward += -1.0 * joint_pos_penalty
    
    # === Encouraging behaviors ===
    # 10. Penalize feet sliding (weight: -0.1)
    # Would need contact sensor data - skip for now

    # 11. Joint deviation penalties (encourage natural poses)
    # Only apply if we have full 22-joint robot
    if num_actions == 22:
        # Hip yaw and roll (weight: -0.5)
        hip_indices = [10, 11, 16, 17]  # hip_roll and hip_yaw for both legs
        joint_deviation_hip = torch.sum(torch.abs(joint_pos[hip_indices])).item()
        reward += -0.5 * joint_deviation_hip

        # Ankle roll (weight: -0.5)
        ankle_indices = [15, 21]  # ankle_roll for both legs
        joint_deviation_ankle = torch.sum(torch.abs(joint_pos[ankle_indices])).item()
        reward += -0.5 * joint_deviation_ankle

        # Shoulder joints (weight: -1.0)
        shoulder_indices = [0, 1, 2, 5, 6, 7]  # shoulder pitch, roll, yaw
        joint_deviation_shoulder = torch.sum(torch.abs(joint_pos[shoulder_indices])).item()
        reward += -1.0 * joint_deviation_shoulder

        # Elbow joints (weight: -1.0)
        elbow_indices = [3, 4, 8, 9]  # elbow pitch and roll
        joint_deviation_elbow = torch.sum(torch.abs(joint_pos[elbow_indices])).item()
        reward += -1.0 * joint_deviation_elbow

    # 12. Survival bonus (keep moving and upright)
    if base_quat[0].item() > 0.7:  # Robot is upright
        reward += 0.5

    return reward


def check_termination(robot: MujocoSimulator, raw_obs: torch.Tensor, ep_len: int, max_ep_len: int = 1000) -> Tuple[bool, str]:
    """
    Check termination conditions matching Isaac Lab.
    
    Returns:
        Tuple[bool, str]: (is_terminated, reason)
    """
    # CRITICAL DEBUG: This should print every single step
    if ep_len == 1:
        print(f"[DEBUG] check_termination called! ep_len={ep_len}")
    
    # Get actual values
    base_height = robot.mj_data.qpos[2]
    base_quat = raw_obs[0:4]
    quat_w = base_quat[0].item()
    
    # 1. Time out (max episode length)
    if ep_len >= max_ep_len:
        print(f"[TERM] timeout at step {ep_len}")
        return True, "timeout"
    
    # 2. Check if base is too low (robot has fallen)
    # if base_height < 0.4:
    #     print(f"[TERM] fallen! height={base_height:.3f} at step {ep_len}")
    #     return True, "fallen"
    
    # 3. Check orientation from quaternion
    if quat_w < 0.7:
        print(f"[TERM] bad_orientation! quat_w={quat_w:.3f} at step {ep_len}")
        return True, "bad_orientation"
    
    # Print status every 50 steps
    if ep_len % 50 == 0:
        print(f"[STATUS] Step {ep_len}: height={base_height:.3f}, quat_w={quat_w:.3f} - OK")
    
    return False, ""


def train_epoch(
    env: MujocoSimulator,
    actor_critic: ActorCritic,
    controller: RlController,
    optimizer: optim.Optimizer,
    buffer: PPOBuffer,
    steps_per_epoch: int,
    device: torch.device,
    clip_ratio: float = 0.2,
    train_iters: int = 80,
    target_kl: float = 0.01,
) -> Dict[str, float]:
    """Train for one epoch."""

    raw_obs = env.reset()

    # Reset controller's observation buffer
    controller.policy_observations.fill(0)
    controller.prev_actions.fill(0)

    # Process first observation using controller's logic
    obs_np = controller.update(raw_obs.numpy())  # This updates internal buffers
    obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)

    ep_ret, ep_len = 0, 0
    epoch_returns = []
    prev_actions = None

    # Collect trajectories
    for t in range(steps_per_epoch):
        # Get action from policy
        obs_tensor = obs.unsqueeze(0).to(device)
        action, logp, value = actor_critic.act(obs_tensor)

        # Execute action in environment (scale back to joint positions)
        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Scale actions to joint positions for environment
        if controller.cfg.num_actions == controller.cfg.num_joints:
            default_joint_positions = torch.tensor(controller.cfg.default_joint_positions, dtype=torch.float32)
        else:
            default_joint_positions = torch.tensor(controller.cfg.default_joint_positions[10:], dtype=torch.float32)

        scaled_actions = action_tensor * controller.cfg.action_scale + default_joint_positions
        raw_next_obs = env.step(scaled_actions)

        # Process next observation using controller's logic
        _ = controller.update(raw_next_obs.numpy())
        next_obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)

        # Compute reward
        reward = compute_reward(env, raw_next_obs, action_tensor, prev_actions)
        prev_actions = action_tensor.clone()

        # Store in buffer
        buffer.store(obs.numpy(), action, reward, value, logp)

        ep_ret += reward
        ep_len += 1

        obs = next_obs

        # Debug: print every 100 steps
        if t % 100 == 0:
            print(f"Training step {t}/{steps_per_epoch}, ep_len={ep_len}")

        # Check termination conditions
        terminated, reason = check_termination(env, raw_next_obs, ep_len)
        epoch_ended = t == steps_per_epoch - 1

        # Debug termination check
        if t % 100 == 0:
            print(f"  terminated={terminated}, reason='{reason}', epoch_ended={epoch_ended}")

        if terminated or epoch_ended:
            if terminated:
                # Add termination penalty (weight: -50.0)
                termination_penalty = -50.0
                buffer.rew_buf[buffer.ptr - 1] += termination_penalty
                ep_ret += termination_penalty
                print(f"Episode ended: {reason} at step {ep_len} (return: {ep_ret:.2f})")
            elif epoch_ended:
                print(f"Warning: trajectory cut off at {ep_len} steps")

            if not terminated or epoch_ended:
                obs_tensor = obs.unsqueeze(0).to(device)
                _, _, value = actor_critic.act(obs_tensor)
            else:
                value = 0

            buffer.finish_path(value)

            if terminated:
                epoch_returns.append(ep_ret)

            raw_obs = env.reset()
            controller.policy_observations.fill(0)
            controller.prev_actions.fill(0)
            _ = controller.update(raw_obs.numpy())
            obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)
            ep_ret, ep_len = 0, 0
            prev_actions = None
    
    # Update policy
    data = buffer.get()
    
    # Move data to device
    for key in data:
        data[key] = data[key].to(device)
    
    policy_loss_sum = 0
    value_loss_sum = 0
    entropy_sum = 0
    kl_sum = 0
    
    for i in range(train_iters):
        optimizer.zero_grad()
        
        # Evaluate actions
        logp, value, entropy = actor_critic.evaluate(data['obs'], data['act'])
        
        # Policy loss
        ratio = torch.exp(logp - data['logp'])
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * data['adv']
        policy_loss = -(torch.min(ratio * data['adv'], clip_adv)).mean()
        
        # Value loss
        value_loss = ((value - data['ret']) ** 2).mean()
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        
        # Update
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        optimizer.step()
        
        # Log metrics
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        entropy_sum += entropy.mean().item()
        
        # Check KL divergence for early stopping
        with torch.no_grad():
            logp_new, _, _ = actor_critic.evaluate(data['obs'], data['act'])
            kl = (data['logp'] - logp_new).mean().item()
            kl_sum += kl
            if kl > 1.5 * target_kl:
                print(f"Early stopping at iteration {i} due to KL divergence")
                break
    
    return {
        'policy_loss': policy_loss_sum / (i + 1),
        'value_loss': value_loss_sum / (i + 1),
        'entropy': entropy_sum / (i + 1),
        'kl': kl_sum / (i + 1),
        'return': np.mean(epoch_returns) if epoch_returns else 0,
    }


def pretrain_with_onnx(
    actor_critic: ActorCritic,
    onnx_policy: OnnxPolicy,
    env: MujocoSimulator,
    controller: RlController,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_samples: int = 1000,
    batch_size: int = 256,
) -> None:
    """
    Warm-start the policy by behavior cloning from ONNX model.
    """
    print(f"\n{'='*60}")
    print("Warm-starting with behavior cloning from ONNX model...")
    print(f"Collecting {num_samples} samples (this may take a few minutes)...")
    print(f"{'='*60}\n")

    # Collect demonstrations from ONNX policy
    obs_list = []
    act_list = []

    raw_obs = env.reset()
    controller.policy_observations.fill(0)
    controller.prev_actions.fill(0)
    _ = controller.update(raw_obs.numpy())
    obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)

    import sys
    for i in range(num_samples):
        # Progress indicator
        if (i + 1) % 100 == 0:
            progress = (i + 1) / num_samples * 100
            print(f"  Progress: {i + 1}/{num_samples} ({progress:.1f}%)")
            sys.stdout.flush()

        # Get action from ONNX policy
        obs_numpy = obs.numpy().reshape(1, -1)
        action_onnx = onnx_policy.forward(obs_numpy)[0]

        # Store demonstration
        obs_list.append(obs.numpy())
        act_list.append(action_onnx)

        # Execute action (scale to joint positions)
        action_tensor = torch.tensor(action_onnx, dtype=torch.float32)

        if controller.cfg.num_actions == controller.cfg.num_joints:
            default_joint_positions = torch.tensor(controller.cfg.default_joint_positions, dtype=torch.float32)
        else:
            default_joint_positions = torch.tensor(controller.cfg.default_joint_positions[10:], dtype=torch.float32)

        scaled_actions = action_tensor * controller.cfg.action_scale + default_joint_positions
        raw_obs = env.step(scaled_actions)

        _ = controller.update(raw_obs.numpy())
        obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)
    
    print(f"\n✓ Collected {num_samples} demonstrations")
    
    # Convert to tensors and move to device
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
    act_tensor = torch.tensor(np.array(act_list), dtype=torch.float32).to(device)
    
    # Behavior cloning training
    print("\nTraining actor with behavior cloning...")
    num_epochs = 30  # Reduced from 50 for faster training
    dataset_size = len(obs_list)
    
    for epoch in range(num_epochs):
        # Shuffle data
        indices = torch.randperm(dataset_size)
        
        total_loss = 0
        num_batches = 0
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            obs_batch = obs_tensor[batch_indices]
            act_batch = act_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            pred_actions = actor_critic.actor(obs_batch)
            
            # MSE loss
            loss = nn.functional.mse_loss(pred_actions, act_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    print("✓ Behavior cloning complete\n")


def main():
    """Main training function."""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune policy in MuJoCo")
    parser.add_argument('--config', type=str, default='./configs/policy_latest.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load (overrides config)')
    parser.add_argument('--no-pretrain', action='store_true',
                       help='Skip behavior cloning warm-start')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=4000,
                       help='Number of steps per epoch')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config file from {args.config}")
    from omegaconf import OmegaConf
    with open(args.config, "r") as f:
        cfg = OmegaConf.load(f)
    
    if not cfg:
        raise ValueError("Failed to load config.")
    
    # Override checkpoint path if provided
    if args.checkpoint:
        cfg.policy_checkpoint_path = args.checkpoint
        print(f"Overriding checkpoint path: {args.checkpoint}")
    
    # Training hyperparameters (matching Isaac Lab PPO config)
    epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    batch_size = steps_per_epoch
    learning_rate = args.lr
    
    # PPO hyperparameters from BerkeleyHumanoidLitePPORunnerCfg
    clip_ratio = 0.2
    gamma = 0.99
    lam = 0.95
    target_kl = 0.01
    num_learning_epochs = 5
    value_loss_coef = 1.0
    entropy_coef = 0.008
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/finetune_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=f"{exp_dir}/logs")
    
    # Save config
    with open(f"{exp_dir}/config.json", "w") as f:
        json.dump(vars(cfg), f, indent=4, default=str)
    
    # Initialize environment
    print("Initializing environment...")
    env = MujocoSimulator(cfg)
    raw_obs = env.reset()

    # Initialize controller for observation processing
    controller = RlController(cfg)
    # We'll use a dummy policy for now (only for observation processing)
    class DummyPolicy:
        def forward(self, obs):
            return np.zeros(len(cfg.action_indices), dtype=np.float32)
    controller.policy = DummyPolicy()

    # Process first observation to get dimensions
    _ = controller.update(raw_obs.numpy())
    obs = torch.tensor(controller.policy_observations[0], dtype=torch.float32)

    # Get dimensions
    obs_dim = obs.shape[0]  # Should match Isaac Lab format with history
    act_dim = len(cfg.action_indices)
    
    print(f"Raw observation dimension: {raw_obs.shape[0]}")
    print(f"Processed observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    
    # Initialize actor-critic (matching Isaac Lab network architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use [256, 128, 128] hidden dims to match Isaac Lab config
    actor_critic = ActorCritic(obs_dim, act_dim, hidden_sizes=[256, 128, 128]).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
    
    # Load pre-trained policy if specified
    onnx_policy = None
    if hasattr(cfg, 'policy_checkpoint_path') and cfg.policy_checkpoint_path:
        checkpoint_path = cfg.policy_checkpoint_path
        print(f"Loading pre-trained policy from {checkpoint_path}")
        
        if checkpoint_path.endswith('.pt'):
            # PyTorch model
            try:
                pretrained = torch.load(checkpoint_path, map_location=device, weights_only=False)
                
                # If it's a full model, extract state dict
                if isinstance(pretrained, nn.Module):
                    pretrained_state = pretrained.state_dict()
                else:
                    pretrained_state = pretrained
                
                # Try to load compatible weights
                actor_critic.load_state_dict(pretrained_state, strict=False)
                print("✓ Loaded pre-trained PyTorch weights")
            except Exception as e:
                print(f"✗ Could not load PyTorch weights: {e}")
                print("  Starting from scratch")
        
        elif checkpoint_path.endswith('.onnx'):
            # ONNX model - warm-start with behavior cloning
            if not args.no_pretrain:
                print("⚠ ONNX format detected - will warm-start with behavior cloning")
                try:
                    onnx_policy = OnnxPolicy(checkpoint_path)
                    # Reduced samples for faster warm-start (1000 instead of 5000)
                    pretrain_with_onnx(actor_critic, onnx_policy, env, controller,
                                     optimizer, device, num_samples=1000, batch_size=256)
                except Exception as e:
                    print(f"✗ Could not load ONNX policy: {e}")
                    print("  Starting from scratch")
            else:
                print("⚠ Skipping ONNX warm-start (--no-pretrain flag set)")
        
        else:
            print(f"✗ Unknown model format: {checkpoint_path}")
            print("  Starting from scratch")
    
    # Initialize buffer (with matching gamma and lambda)
    buffer = PPOBuffer(obs_dim, act_dim, steps_per_epoch, gamma=gamma, lam=lam)
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()

        # Train one epoch
        metrics = train_epoch(
            env, actor_critic, controller, optimizer, buffer, steps_per_epoch, device,
            clip_ratio=clip_ratio, train_iters=num_learning_epochs * 4, target_kl=target_kl
        )
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Return: {metrics['return']:.2f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")
        print(f"  KL: {metrics['kl']:.4f}")
        print(f"  Time: {epoch_time:.2f}s (Total: {total_time:.2f}s)")
        
        # Tensorboard logging
        for key, value in metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        writer.add_scalar("train/epoch_time", epoch_time, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"{exp_dir}/checkpoints/policy_epoch_{epoch + 1}.pt"
            torch.save(actor_critic.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = f"{exp_dir}/policy_final.pt"
    torch.save(actor_critic, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    writer.close()


if __name__ == "__main__":
    main()