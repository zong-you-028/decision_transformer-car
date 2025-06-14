import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gymnasium as gym
import highway_env
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import pickle
from typing import List, Tuple, Dict
import math
import cv2
import os
from PIL import Image
import seaborn as sns

class HighwayEvaluator:
    """
    Evaluate Decision Transformer on Highway-Env with animation capabilities
    """
    def __init__(self, model, env_name="highway-v0", device="cpu"):
        self.model = model.to(device)
        self.device = device

        cfg = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted",
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,
        }

        # ↓ 直接把 cfg 帶進來
        self.env = gym.make(env_name, config=cfg, render_mode="rgb_array")
                # === 新增：用來即時渲染的環境 ===
        self.cfg = {             # 與收集資料時相同
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy"],
                "absolute": False,
                "order": "sorted"
            },
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,
        }
        self.vis_env = gym.make("highway-v0",
                                config=self.cfg,
                                render_mode="rgb_array")  # 👈 關鍵
        self.vis_obs, _ = self.vis_env.reset()             # 初始化
        self.vis_step = 0                                  # 紀錄當前時刻
    def evaluate_with_animation(self, num_episodes=5, target_return=30, save_animation=True):
        """Evaluate model and create animation"""
        self.model.eval()
        
        all_frames = []
        all_episode_data = []
        
        with torch.no_grad():
            for episode in range(num_episodes):
                print(f"Recording episode {episode + 1}/{num_episodes}")
                
                obs, info = self.env.reset()
                episode_frames = []
                episode_info = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'returns_to_go': []
                }
                
                # Initialize episode data
                states = torch.zeros((1, 20, obs.shape[0] * obs.shape[1]), device=self.device)
                actions = torch.zeros((1, 20, 5), device=self.device)
                returns_to_go = torch.full(
                     (1, 20, 1),
                     float(target_return),           # 或者直接傳 float
                     device=self.device,
                     dtype=torch.float32             # 👈 關鍵
                )
                timesteps = torch.arange(20, device=self.device).unsqueeze(0)
                
                episode_return = 0
                t = 0
                done = truncated = False
                
                while not (done or truncated) and t < 40:  # Extended for better animation
                    # Render current state
                    frame = self.env.render()
                    episode_frames.append(frame)
                    
                    # Flatten observation
                    obs_flat = obs.flatten()
                    if t < 20:
                        states[0, t] = torch.tensor(obs_flat, device=self.device)
                    
                    # Get action from model
                    if t < 20:
                        if t == 0:
                            attention_mask = torch.zeros(1, 20, dtype=torch.bool, device=self.device)
                            attention_mask[0, 0] = True
                        else:
                            attention_mask = torch.zeros(1, 20, dtype=torch.bool, device=self.device)
                            attention_mask[0, :t+1] = True
                        
                        _, action_preds, _ = self.model(
                            states, actions, None, returns_to_go, timesteps, attention_mask
                        )
                        
                        action = action_preds[0, min(t, 19)].argmax().item()
                        
                        if t < 20:
                            actions[0, t] = F.one_hot(torch.tensor(action), num_classes=5).float()
                    else:
                        # Use simple heuristic for extended period
                        action = self._simple_heuristic(obs)
                    
                    # Step environment
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Store episode data
                    episode_info['states'].append(obs_flat)
                    episode_info['actions'].append(action)
                    episode_info['rewards'].append(reward)
                    
                    # Update returns to go
                    if t < 19:
                        returns_to_go[0, t+1:] -= reward
                        episode_info['returns_to_go'].append(returns_to_go[0, t+1, 0].item())
                    else:
                        episode_info['returns_to_go'].append(0)
                    
                    episode_return += reward
                    t += 1
                
                all_frames.append(episode_frames)
                all_episode_data.append(episode_info)
                print(f"Episode {episode + 1} return: {episode_return:.2f}")
        
        if save_animation:
            self.create_training_animation(all_frames, all_episode_data)
        
        return all_frames, all_episode_data
    
    def create_training_animation(self, all_frames, all_episode_data, filename='decision_transformer_highway.mp4'):
        """Create animated video showing model performance"""
        print("Creating animation...")
        
        # Setup figure for animation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Decision Transformer Highway Driving', fontsize=16)
        
        # Configure axes
        axes[0, 0].set_title('Highway Environment')
        axes[0, 0].axis('off')
        
        axes[0, 1].set_title('Action Distribution')
        axes[0, 1].set_xlabel('Action')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].set_title('Rewards Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Returns-to-Go')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Return-to-Go')
        axes[1, 1].grid(True)
        
        # Action names
        action_names = ['IDLE', 'FASTER', 'SLOWER', 'LANE_LEFT', 'LANE_RIGHT']
        
        def animate_frame(frame_idx):
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            # Calculate which episode and frame
            episode_idx = 0
            local_frame_idx = frame_idx
            
            for i, episode_frames in enumerate(all_frames):
                if local_frame_idx < len(episode_frames):
                    episode_idx = i
                    break
                local_frame_idx -= len(episode_frames)
            
            if episode_idx >= len(all_frames):
                return
            
            # Get current episode data
            episode_frames = all_frames[episode_idx]
            episode_data = all_episode_data[episode_idx]
            
            if local_frame_idx >= len(episode_frames):
                return
            
            # Display current environment frame
            axes[0, 0].imshow(episode_frames[local_frame_idx])
            axes[0, 0].set_title(f'Episode {episode_idx + 1}, Step {local_frame_idx + 1}')
            axes[0, 0].axis('off')
            
            # Action distribution up to current step
            if local_frame_idx > 0:
                current_actions = episode_data['actions'][:local_frame_idx]
                action_counts = [current_actions.count(i) for i in range(5)]
                
                bars = axes[0, 1].bar(action_names, action_counts, 
                                    color=['blue', 'green', 'red', 'orange', 'purple'])
                axes[0, 1].set_title('Action Distribution')
                axes[0, 1].set_xlabel('Action')
                axes[0, 1].set_ylabel('Frequency')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{int(height)}', ha='center', va='bottom')
            
            # Rewards over time
            if local_frame_idx > 0:
                rewards = episode_data['rewards'][:local_frame_idx]
                axes[1, 0].plot(rewards, 'b-', linewidth=2)
                axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[1, 0].set_title('Rewards Over Time')
                axes[1, 0].set_xlabel('Time Step')
                axes[1, 0].set_ylabel('Reward')
                axes[1, 0].grid(True)
            
            # Returns-to-go
            if local_frame_idx > 0 and local_frame_idx <= len(episode_data['returns_to_go']):
                returns_to_go = episode_data['returns_to_go'][:local_frame_idx]
                axes[1, 1].plot(returns_to_go, 'g-', linewidth=2)
                axes[1, 1].set_title('Returns-to-Go')
                axes[1, 1].set_xlabel('Time Step')
                axes[1, 1].set_ylabel('Return-to-Go')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
        
        # Calculate total frames
        total_frames = sum(len(frames) for frames in all_frames)
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=total_frames, 
            interval=200, blit=False, repeat=True
        )
        
        # Save animation
        print(f"Saving animation to {filename}...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Decision Transformer'), bitrate=1800)
        anim.save(filename, writer=writer)
        
        print(f"Animation saved as {filename}")
        plt.close()
    
    def _simple_heuristic(self, obs):
        """Simple heuristic for extended episodes"""
        ego_vehicle = obs[0]
        x, y, vx, vy = ego_vehicle[1], ego_vehicle[2], ego_vehicle[3], ego_vehicle[4]
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 15:
            return 1  # FASTER
        elif speed > 25:
            return 2  # SLOWER
        else:
            return 0  # IDLE
    
    def evaluate(self, num_episodes=100, target_return=30):
        """Standard evaluation without animation"""
        self.model.eval()
        
        episode_returns = []
        success_count = 0
        collision_count = 0
        
        with torch.no_grad():
            for episode in range(num_episodes):
                obs, info = self.env.reset()
                
                # Initialize episode data
                states = torch.zeros((1, 20, obs.shape[0] * obs.shape[1]), device=self.device)
                actions = torch.zeros((1, 20, 5), device=self.device)
                returns_to_go = torch.full(
                    (1, 20, 1),
                    float(target_return),
                    device=self.device,
                    dtype=torch.float32
                )
                timesteps = torch.arange(20, device=self.device).unsqueeze(0)
                
                episode_return = 0
                t = 0
                done = truncated = False
                
                while not (done or truncated) and t < 20:
                    # Flatten observation
                    obs_flat = obs.flatten()
                    states[0, t] = torch.tensor(obs_flat, device=self.device)
                    
                    # Get action from model
                    if t == 0:
                        attention_mask = torch.zeros(1, 20, dtype=torch.bool, device=self.device)
                        attention_mask[0, 0] = True
                    else:
                        attention_mask = torch.zeros(1, 20, dtype=torch.bool, device=self.device)
                        attention_mask[0, :t+1] = True
                    
                    _, action_preds, _ = self.model(
                        states, actions, None, returns_to_go, timesteps, attention_mask
                    )
                    
                    action = action_preds[0, t].argmax().item()
                    
                    # Step environment
                    obs, reward, done, truncated, info = self.env.step(action)
                    
                    # Update episode data
                    actions[0, t] = F.one_hot(torch.tensor(action), num_classes=5).float()
                    returns_to_go[0, t+1:] -= reward
                    episode_return += reward
                    
                    if info.get('crashed', False):
                        collision_count += 1
                    
                    t += 1
                
                episode_returns.append(episode_return)
                if episode_return > target_return * 0.8:  # Success threshold
                    success_count += 1
                
                if episode % 20 == 0:
                    print(f"Episode {episode}, Return: {episode_return:.2f}")
        
        results = {
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'episode_returns': episode_returns
        }
        
        return results

