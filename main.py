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

from decision_transformer.model import DecisionTransformer
from decision_transformer.data import HighwayDataset, HighwayDataCollector
from decision_transformer.trainer import DecisionTransformerTrainer
from decision_transformer.evaluator import HighwayEvaluator

def main():
    """Main training and evaluation pipeline"""
    
    # 1. Collect data
    print("Collecting trajectories...")
    collector = HighwayDataCollector()
    
    # Collect trajectories from three driving styles
    aggressive_trajs = collector.collect_aggressive_trajectories(20)
    cautious_trajs = collector.collect_cautious_trajectories(20)
    normal_trajs = collector.collect_expert_trajectories(20)
    all_trajectories = aggressive_trajs + cautious_trajs + normal_trajs
    
    # Save trajectories
    with open('highway_trajectories.pkl', 'wb') as f:
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
    train_losses = trainer.train(
        dataset,
        num_epochs=100,
        batch_size=32,
        record_animation=True,
        animation_file="training_animation.mp4",
    )
    
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
