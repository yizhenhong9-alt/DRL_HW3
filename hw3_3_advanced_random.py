import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os
import copy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset

# Add Chapter 3 to sys.path to import Gridworld
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'DeepReinforcementLearningInAction', 'Chapter 3')))
from Gridworld import Gridworld

# 1. Dueling DQN Model with Batch Normalization
class DuelingDQNBN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQNBN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.BatchNorm1d(150),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(150, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(150, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

# 2. Replay Dataset
class ReplayDataset(Dataset):
    def __init__(self, replay_buffer, length=1000):
        self.buffer = replay_buffer
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return random.choice(self.buffer)

# 3. Lightning Module
class DQNLightning(pl.LightningModule):
    def __init__(self, gamma=0.9, lr=1e-3, batch_size=64, mem_size=5000):
        super().__init__()
        self.save_hyperparameters()
        self.model = DuelingDQNBN()
        self.target_model = DuelingDQNBN()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.replay_buffer = deque(maxlen=mem_size)
        self.game = Gridworld(size=4, mode='random')
        self.epsilon = 1.0
        self.total_wins = 0
        self.games_played = 0
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def select_action(self, state):
        if self.trainer.training and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                self.model.eval()
                q_vals = self.model(state)
                self.model.train()
                return torch.argmax(q_vals).item()
    
    def training_step(self, batch, batch_idx):
        for _ in range(5): 
            self.populate_buffer()
        
        s1, a, r, s2, d = batch
        
        Q1 = self.model(s1)
        with torch.no_grad():
            self.model.eval()
            next_actions = torch.argmax(self.model(s2), dim=1)
            self.model.train()
            
            Q2_target = self.target_model(s2)
            Q2_max = Q2_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            
        targets = r + self.hparams.gamma * (1 - d.float()) * Q2_max
        X = Q1.gather(1, a.unsqueeze(1)).squeeze()
        
        loss = self.loss_fn(X, targets)
        self.log('train_loss', loss)
        self.log('win_rate', (self.total_wins/self.games_played) if self.games_played > 0 else 0.0, prog_bar=True)
        
        if self.global_step % 200 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        return loss

    def populate_buffer(self):
        state1_ = self.game.board.render_np().reshape(1, 64).astype(float)
        state1 = torch.from_numpy(state1_).float()
        
        action = self.select_action(state1)
        action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.game.makeMove(action_map[action])
        
        reward = self.game.reward()
        state2_ = self.game.board.render_np().reshape(1, 64).astype(float)
        state2 = torch.from_numpy(state2_).float()
        
        done = True if reward != -1 else False
        self.replay_buffer.append((state1.squeeze(), action, reward, state2.squeeze(), done))
        
        if done:
            if reward == 10:
                self.total_wins += 1
            self.games_played += 1
            self.game = Gridworld(size=4, mode='random')
            if self.epsilon > 0.1:
                self.epsilon -= 0.001

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        while len(self.replay_buffer) < 500:
            self.populate_buffer()
        return DataLoader(ReplayDataset(self.replay_buffer, length=200), batch_size=self.hparams.batch_size, shuffle=True)

def train_advanced():
    print("Starting Advanced DQN Training (Random Mode) using PyTorch Lightning...")
    model = DQNLightning(lr=1e-3, batch_size=64)
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='train_loss', patience=30, mode='min')
    
    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=1.0, 
        callbacks=[early_stop],
        enable_checkpointing=False, # Disable checkpointing to avoid WinError 32
        logger=False # Disable logging to avoid file lock issues
    )
    
    trainer.fit(model)
    
    # 儲存權重檔 (PyTorch Lightning 需存取內部的 model.model)
    print("Saving model to hw3_3_model.pth...")
    torch.save(model.model.state_dict(), 'hw3_3_model.pth')
    
    test_random(model.model)

def test_random(model):
    print("\nTesting on 100 Random Maps...")
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    wins = 0
    model.eval()
    for _ in range(100):
        game = Gridworld(size=4, mode='random')
        mov = 0
        while mov < 15:
            state_ = game.board.render_np().reshape(1, 64).astype(float)
            state = torch.from_numpy(state_).float()
            with torch.no_grad():
                q_vals = model(state)
            action = torch.argmax(q_vals).item()
            game.makeMove(action_set[action])
            if game.reward() != -1:
                if game.reward() > 0: wins += 1
                break
            mov += 1
    print(f"Win Rate: {wins/100:.2%}")

if __name__ == "__main__":
    train_advanced()
