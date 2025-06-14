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

class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Highway-Env
    """
    def __init__(self, 
                 state_dim: int,
                 act_dim: int,
                 hidden_size: int = 128,
                 max_length: int = 20,
                 n_layer: int = 3,
                 n_head: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.state_dim = state_dim
        self.act_dim = act_dim
        
        # Token embeddings
        self.embed_timestep = nn.Embedding(max_length, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=4*hidden_size,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_size)
        
        # Prediction heads
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Linear(hidden_size, act_dim)
        self.predict_return = nn.Linear(hidden_size, 1)
        
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        
        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # Time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Stack return, state, action embeddings
        # Shape: (batch, seq_len*3, hidden_size)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Create attention mask for stacked inputs
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=2
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # Apply transformer
        transformer_outputs = stacked_inputs
        for transformer in self.transformer:
            transformer_outputs = transformer(
                transformer_outputs, 
                src_key_padding_mask=~stacked_attention_mask
            )
        
        x = self.ln_f(transformer_outputs)
        
        # Reshape back to (batch, seq_len, 3, hidden_size)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # Get predictions
        return_preds = self.predict_return(x[:,2])  # Predict from action tokens
        state_preds = self.predict_state(x[:,2])
        action_preds = self.predict_action(x[:,1])  # Predict from state tokens
        
        return state_preds, action_preds, return_preds

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
            "attention_mask": torch.from_numpy(attention_mask)
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

class DecisionTransformerTrainer:
    """
    Trainer for Decision Transformer with visualization capabilities
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.train_losses = []
        self.eval_results = []
        
        # Setup for visualization
        plt.style.use('seaborn')
        self.fig = None
        self.axes = None
        
        # === 新增：初始化用來即時渲染的環境 ===
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
        
    def train_with_visualization(self, dataset, num_epochs=100, batch_size=64, eval_interval=10):
        """Train with real-time visualization"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup visualization
        self.setup_training_plots()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                # Move to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                returns_to_go = batch['returns_to_go'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Convert actions to one-hot
                actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=5).float()
                
                # Forward pass
                state_preds, action_preds, return_preds = self.model(
                    states, actions_one_hot, None, returns_to_go, timesteps, attention_mask
                )
                
                # Calculate losses
                action_loss = F.cross_entropy(
                    action_preds.reshape(-1, action_preds.size(-1)),
                    actions.reshape(-1),
                    reduction='none'
                )
                action_loss = (action_loss * attention_mask.reshape(-1)).mean()
                
                loss = action_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)
            
            # Periodic evaluation and visualization update
            if epoch % eval_interval == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
                
                # Quick evaluation
                if epoch > 0:
                    eval_result = self.quick_evaluation()
                    self.eval_results.append(eval_result)
                
                # Update plots
                self.update_training_plots(epoch)
        
        return self.train_losses
    
    def setup_training_plots(self):
        """Setup real-time training visualization"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Decision Transformer Training Progress', fontsize=16)
        
        # (0,0) 用來顯示環境影像
        self.axes[0, 0].set_title("Env Frame (per batch)")
        self.axes[0, 0].axis("off")
        
        # Loss plot
        self.axes[0, 1].set_title('Training Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True)
        
        # Action distribution
        self.axes[1, 0].set_title('Action Distribution in Last Batch')
        self.axes[1, 0].set_xlabel('Action')
        self.axes[1, 0].set_ylabel('Frequency')
        
        # Attention heatmap placeholder
        self.axes[1, 1].set_title('Attention Pattern (Last Layer)')
        
        plt.tight_layout()
        plt.ion()  # Turn on interactive mode
        
    def update_training_plots(self, epoch):
        """Update training visualization plots"""
        # Clear previous plots (except the environment frame)
        self.axes[0, 1].clear()
        self.axes[1, 0].clear()
        self.axes[1, 1].clear()
        
        # Loss plot
        self.axes[0, 1].plot(self.train_losses, 'b-', linewidth=2)
        self.axes[0, 1].set_title('Training Loss')
        self.axes[0, 1].set_xlabel('Epoch')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True)
        
        # Evaluation metrics
        if self.eval_results:
            eval_epochs = np.arange(0, len(self.eval_results) * 10, 10)
            returns = [r['mean_return'] for r in self.eval_results]
            success_rates = [r['success_rate'] * 100 for r in self.eval_results]
            
            ax2 = self.axes[0, 1].twinx()
            ax2.plot(eval_epochs, returns, 'g-', label='Mean Return', linewidth=2)
            ax2.plot(eval_epochs, success_rates, 'r-', label='Success Rate (%)', linewidth=2)
            ax2.set_ylabel('Evaluation Score')
            ax2.legend()
        
        # Action distribution (placeholder - would need actual batch data)
        actions = ['IDLE', 'FASTER', 'SLOWER', 'LANE_LEFT', 'LANE_RIGHT']
        action_counts = np.random.multinomial(100, [0.3, 0.2, 0.15, 0.2, 0.15])  # Placeholder
        bars = self.axes[1, 0].bar(actions, action_counts, color=['blue', 'green', 'red', 'orange', 'purple'])
        self.axes[1, 0].set_title('Action Distribution in Training')
        self.axes[1, 0].set_xlabel('Action')
        self.axes[1, 0].set_ylabel('Frequency')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            self.axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
        
        # Attention visualization (simplified)
        attention_matrix = np.random.rand(10, 10)  # Placeholder
        im = self.axes[1, 1].imshow(attention_matrix, cmap='Blues', aspect='auto')
        self.axes[1, 1].set_title('Attention Pattern (Simplified)')
        self.axes[1, 1].set_xlabel('Key Position')
        self.axes[1, 1].set_ylabel('Query Position')
        
        plt.tight_layout()
        plt.pause(0.1)  # Small pause to allow plot update
    
    def quick_evaluation(self, num_episodes=10):
        """Quick evaluation during training"""
        evaluator = HighwayEvaluator(self.model, device=self.device)
        results = evaluator.evaluate(num_episodes=num_episodes, target_return=25)
        return results
    
    def train(self, dataset, num_epochs=100, batch_size=64):
        """Training method with environment visualization"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup basic training plots
        self.setup_training_plots()
        
        train_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # ----- 1) 先讓模型跟環境互動一步 -----
                with torch.no_grad():
                    # 把 vis_obs flatten→tensor→餵模型決策
                    obs_flat = torch.tensor(self.vis_obs.flatten(),
                                            dtype=torch.float32, device=self.device).unsqueeze(0)
                    # 我們只需要一個 action，先做最簡單推論：
                    dummy_states   = obs_flat.unsqueeze(1).repeat(1, 1, 1)        # (1,1,75)
                    dummy_actions  = torch.zeros((1, 1, 5), device=self.device)   # (1,1,5)
                    dummy_rtg      = torch.zeros((1, 1, 1), device=self.device)   # (1,1,1)
                    dummy_ts       = torch.zeros((1, 1),   dtype=torch.long,
                                                device=self.device)
                    dummy_mask     = torch.ones((1, 1),    dtype=torch.bool,
                                                device=self.device)

                    _, act_pred, _ = self.model(dummy_states,
                                                dummy_actions,
                                                None, dummy_rtg, dummy_ts, dummy_mask)
                    act = act_pred[0, 0].argmax().item()

                # 環境 step & render
                self.vis_obs, _, done, truncated, _ = self.vis_env.step(act)
                frame = self.vis_env.render()

                if done or truncated:           # episode 結束就重新開始
                    self.vis_obs, _ = self.vis_env.reset()
                    self.vis_step = 0
                else:
                    self.vis_step += 1

                # 只在每個batch的第一個iteration顯示 frame
                if batch_idx % 10 == 0:  # 每10個batch更新一次畫面
                    self.axes[0, 0].clear()
                    self.axes[0, 0].imshow(frame)
                    self.axes[0, 0].set_title(f"Env Frame – step {self.vis_step}")
                    self.axes[0, 0].axis("off")
                    plt.pause(0.001)

                # ----- 2) 正常的訓練流程 -----
                # Move to device
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                returns_to_go = batch['returns_to_go'].to(self.device)
                timesteps = batch['timesteps'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Convert actions to one-hot
                actions_one_hot = F.one_hot(actions.squeeze(-1), num_classes=5).float()
                
                # Forward pass
                state_preds, action_preds, return_preds = self.model(
                    states, actions_one_hot, None, returns_to_go, timesteps, attention_mask
                )
                
                # Calculate losses
                action_loss = F.cross_entropy(
                    action_preds.reshape(-1, action_preds.size(-1)),
                    actions.reshape(-1),
                    reduction='none'
                )
                action_loss = (action_loss * attention_mask.reshape(-1)).mean()
                
                loss = action_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
                # Update loss plot
                self.axes[0, 1].clear()
                self.axes[0, 1].plot(train_losses, 'b-', linewidth=2)
                self.axes[0, 1].set_title('Training Loss')
                self.axes[0, 1].set_xlabel('Epoch')
                self.axes[0, 1].set_ylabel('Loss')
                self.axes[0, 1].grid(True)
                plt.pause(0.1)
        
        return train_losses
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

def main():
    """Main training and evaluation pipeline"""
    
    # 1. Load existing data or collect new trajectories
    data_file = 'highway_trajectories.pkl'

    if os.path.exists(data_file):
        print(f"Loading trajectories from {data_file}...")
        with open(data_file, 'rb') as f:
            all_trajectories = pickle.load(f)
    else:
        print("Collecting trajectories...")
        collector = HighwayDataCollector()

        # Collect mixed quality data
        random_trajs = collector.collect_random_trajectories(50)
        expert_trajs = collector.collect_expert_trajectories(30)
        all_trajectories = random_trajs + expert_trajs

        with open(data_file, 'wb') as f:
            pickle.dump(all_trajectories, f)
    
    # 2. Create dataset
    print("Creating dataset...")
    dataset = HighwayDataset(all_trajectories, max_len=20)
    
    # 3. Initialize model
    state_dim = 75  # 15 vehicles * 5 features
    act_dim = 5     # Highway-Env discrete actions
    
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=128,
        max_length=20,
        n_layer=3,
        n_head=8
    )
    
    # 4. Train model
    print("Training model...")
    trainer = DecisionTransformerTrainer(model)
    train_losses = trainer.train(dataset, num_epochs=100, batch_size=32)
    
    # Save model
    torch.save(model.state_dict(), 'decision_transformer_highway.pth')
    
    # 5. Evaluate model
    print("Evaluating model...")
    evaluator = HighwayEvaluator(model)
    results = evaluator.evaluate(num_episodes=50, target_return=25)
    
    print("\nEvaluation Results:")
    print(f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Collision Rate: {results['collision_rate']:.1%}")

if __name__ == '__main__':
    main()