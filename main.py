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
from typing import List, Tuple, Dict
import math
import cv2
from PIL import Image
import seaborn as sns

from decision_transformer.model import DecisionTransformer
from decision_transformer.data import (
    HighwayDataset,
    load_or_collect_trajectories,
)
from decision_transformer.trainer import DecisionTransformerTrainer
from decision_transformer.evaluator import HighwayEvaluator


def analyze_trajectories_by_style(trajectories, output_file="style_analysis.png"):
    """Create plots comparing returns and action usage for each driving style."""
    styles = defaultdict(list)
    for traj in trajectories:
        styles[traj.get("style", "normal")].append(traj)

    num_styles = len(styles)
    fig, axes = plt.subplots(num_styles, 2, figsize=(10, 4 * num_styles))

    if num_styles == 1:
        axes = np.array([axes])

    action_names = ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"]

    for idx, (style, trajs) in enumerate(sorted(styles.items())):
        returns = [sum(t["rewards"]) for t in trajs]
        axes[idx, 0].hist(returns, bins=10, color="skyblue")
        axes[idx, 0].set_title(f"{style.capitalize()} Returns")
        axes[idx, 0].set_xlabel("Return")
        axes[idx, 0].set_ylabel("Frequency")

        action_counts = np.zeros(len(action_names))
        for t in trajs:
            for a in t["actions"]:
                action_counts[a] += 1
        axes[idx, 1].bar(action_names, action_counts, color="salmon")
        axes[idx, 1].set_title(f"{style.capitalize()} Action Distribution")
        axes[idx, 1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Style analysis saved to {output_file}")

def main():
    """Main training and evaluation pipeline"""
    
    # 1. Load cached trajectories or collect new ones
    all_trajectories = load_or_collect_trajectories(
        file_path="highway_trajectories.pkl", num_trajs_per_style=20
    )
    
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
    train_losses = trainer.train(
        dataset,
        num_epochs=100,
        batch_size=32,
        record_animation=False,
        animation_file="training_animation.mp4",
    )

    analyze_trajectories_by_style(all_trajectories)
    
    # Save model
    torch.save(model.state_dict(), 'decision_transformer_highway.pth')
    
    # 5. Evaluate model
    print("Evaluating model...")
    evaluator = HighwayEvaluator(model)
    results = evaluator.evaluate(num_episodes=50, target_return=25)
    evaluator.evaluate_with_animation(num_episodes=5, target_return=25, save_animation=True)
    
    print("\nEvaluation Results:")
    print(f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Collision Rate: {results['collision_rate']:.1%}")

if __name__ == '__main__':
    main()
