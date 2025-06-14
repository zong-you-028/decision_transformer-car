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

class HighwayDataset(Dataset):
    """
    Dataset for Highway-Env trajectories
    """
    def __init__(self, trajectories: List[Dict], max_len: int = 20):
        self.trajectories = trajectories
        self.max_len = max_len
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # (時間, 15, 5) → (時間, 75)
        states_raw = np.array(traj['observations'])
        states = states_raw.reshape(states_raw.shape[0], -1)

        actions = np.array(traj['actions'])
        rewards = np.array(traj['rewards'])
        returns_to_go = self._calculate_returns_to_go(rewards)

        traj_len = min(len(states), self.max_len)

        padded_states = np.zeros((self.max_len, states.shape[1]), dtype=np.float32)  # (20, 75)
        padded_actions = np.zeros((self.max_len, 1), dtype=np.int64)
        padded_returns = np.zeros((self.max_len, 1), dtype=np.float32)
        padded_timesteps = np.zeros(self.max_len, dtype=np.int64)
        attention_mask = np.zeros(self.max_len, dtype=bool)

        padded_states[:traj_len] = states[:traj_len]
        padded_actions[:traj_len, 0] = actions[:traj_len]
        padded_returns[:traj_len, 0] = returns_to_go[:traj_len]
        padded_timesteps[:traj_len] = np.arange(traj_len)
        attention_mask[:traj_len] = True

        return {
            "states": torch.from_numpy(padded_states),
            "actions": torch.from_numpy(padded_actions),
            "returns_to_go": torch.from_numpy(padded_returns),
            "timesteps": torch.from_numpy(padded_timesteps),
            "attention_mask": torch.from_numpy(attention_mask),
            "style": traj.get("style", "normal")
        }

    def _calculate_returns_to_go(self, rewards):
        returns_to_go = []
        cumulative = 0
        for r in reversed(rewards):
            cumulative += r
            returns_to_go.append(cumulative)
        return list(reversed(returns_to_go))

class HighwayDataCollector:
    """
    Collect trajectories from Highway-Env
    """
    def __init__(self, env_name: str = 'highway-v0'):
    
        cfg = {        # 建議放成屬性，下面也能重用
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
            "duration": 5,
            "initial_spacing": 2,
            "collision_reward": -1,
            "reward_speed_range": [20, 30],
       }
        self.env = gym.make(env_name, config=cfg)
        
    def collect_random_trajectories(self, num_trajectories: int = 20):
        """Collect random trajectories"""
        trajectories = []
        
        for i in range(num_trajectories):
            obs, info = self.env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            done = truncated = False
            while not (done or truncated):
                action = self.env.action_space.sample()
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                
                obs, reward, done, truncated, info = self.env.step(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)
                
            trajectories.append(trajectory)
            
            if i % 100 == 0:
                print(f"Collected {i}/{num_trajectories} trajectories")
        
        return trajectories
    
    def collect_expert_trajectories(self, num_trajectories: int = 100):
        """Collect trajectories using a simple heuristic policy"""
        trajectories = []
        
        for i in range(num_trajectories):
            obs, info = self.env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': []
            }
            
            done = truncated = False
            while not (done or truncated):
                # Simple heuristic: maintain speed and avoid collisions
                action = self._heuristic_policy(obs)
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                
                obs, reward, done, truncated, info = self.env.step(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)
                
            trajectories.append(trajectory)
            
            if i % 50 == 0:
                print(f"Collected {i}/{num_trajectories} expert trajectories")
        
        return trajectories

    # === New methods for different driving styles ===
    def collect_aggressive_trajectories(self, num_trajectories: int = 100):
        """Collect trajectories using an aggressive driving policy"""
        trajectories = []

        for i in range(num_trajectories):
            obs, info = self.env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'style': 'aggressive'
            }

            done = truncated = False
            while not (done or truncated):
                action = self._aggressive_policy(obs)
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)

                obs, reward, done, truncated, info = self.env.step(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)

            trajectories.append(trajectory)

            if i % 50 == 0:
                print(f"Collected {i}/{num_trajectories} aggressive trajectories")

        return trajectories

    def collect_cautious_trajectories(self, num_trajectories: int = 100):
        """Collect trajectories using a cautious driving policy"""
        trajectories = []

        for i in range(num_trajectories):
            obs, info = self.env.reset()
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'style': 'cautious'
            }

            done = truncated = False
            while not (done or truncated):
                action = self._cautious_policy(obs)
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)

                obs, reward, done, truncated, info = self.env.step(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)

            trajectories.append(trajectory)

            if i % 50 == 0:
                print(f"Collected {i}/{num_trajectories} cautious trajectories")

        return trajectories
    
    def _heuristic_policy(self, obs):
        """Simple heuristic policy for Highway-Env"""
        # obs shape: (vehicles_count, features)
        ego_vehicle = obs[0]  # First vehicle is ego
        
        # Extract ego vehicle features
        x, y, vx, vy = ego_vehicle[1], ego_vehicle[2], ego_vehicle[3], ego_vehicle[4]
        
        # Simple policy: 
        # - If speed is low, accelerate
        # - If there's a vehicle close ahead, change lane or slow down
        # - Otherwise, maintain speed
        
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 15:
            return 1  # FASTER
        elif speed > 25:
            return 2  # SLOWER
        else:
            # Check for vehicles ahead
            for i in range(1, len(obs)):
                if obs[i][0] == 0:  # Vehicle not present
                    continue
                other_x, other_y = obs[i][1], obs[i][2]
                
                # If vehicle is ahead and close
                if other_x > x and abs(other_y - y) < 2 and other_x - x < 10:
                    if y > 0:  # If not in rightmost lane
                        return 4  # LANE_RIGHT
                    else:
                        return 2  # SLOWER
            
            return 0  # IDLE (maintain current speed)

    def _aggressive_policy(self, obs):
        """Aggressive driving policy"""
        ego = obs[0]
        x, y, vx, vy = ego[1], ego[2], ego[3], ego[4]
        speed = np.sqrt(vx**2 + vy**2)

        if speed < 30:
            return 1  # always try to accelerate
        # frequently change to left lane if possible
        for i in range(1, len(obs)):
            if obs[i][0] and obs[i][1] > x and abs(obs[i][2]-y) < 2 and obs[i][1]-x < 8:
                if y < 3:
                    return 3  # LANE_LEFT
                else:
                    return 1
        return 0

    def _cautious_policy(self, obs):
        """Cautious driving policy"""
        ego = obs[0]
        x, y, vx, vy = ego[1], ego[2], ego[3], ego[4]
        speed = np.sqrt(vx**2 + vy**2)

        if speed > 20:
            return 2  # SLOWER
        for i in range(1, len(obs)):
            if obs[i][0] and obs[i][1] > x and abs(obs[i][2]-y) < 2 and obs[i][1]-x < 15:
                return 2
        return 0

