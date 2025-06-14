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

