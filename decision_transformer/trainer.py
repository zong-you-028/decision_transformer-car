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
        plt.style.use('seaborn-v0_8')
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
    
    def train(self, dataset, num_epochs=100, batch_size=64,
              record_animation: bool = False,
              animation_file: str = "training.mp4"):
        """Training method with optional animation recording"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if record_animation:
            self.record_frames = []
        
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
                if record_animation:
                    self.record_frames.append(frame)

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
        
        if record_animation:
            self.save_training_animation(animation_file)

        return train_losses

    def save_training_animation(self, filename: str):
        """Save recorded training frames as an MP4 file"""
        if not hasattr(self, "record_frames") or not self.record_frames:
            return

        print(f"Saving training animation to {filename}...")
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        im = plt.imshow(self.record_frames[0])

        def update(i):
            im.set_data(self.record_frames[i])
            return im,

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.record_frames), interval=200, blit=False
        )

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Decision Transformer'), bitrate=1800)
        anim.save(filename, writer=writer)
        plt.close(fig)
