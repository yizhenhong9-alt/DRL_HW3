import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os
import copy
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Add Chapter 3 to sys.path to import Gridworld
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'DeepReinforcementLearningInAction', 'Chapter 3')))
from Gridworld import Gridworld

# 1. NoisyLinear Layer for Exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return nn.functional.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                                        self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(x, self.weight_mu, self.bias_mu)

# 2. Rainbow DQN Architecture (Dueling + Noisy)
class RainbowDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(RainbowDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Value stream (using NoisyLinear)
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, 1)
        )
        
        # Advantage stream (using NoisyLinear)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, output_dim)
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# 3. Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.stack(states), torch.tensor(actions), torch.tensor(rewards, dtype=torch.float32), 
                torch.stack(next_states), torch.tensor(dones, dtype=torch.bool), indices, torch.tensor(weights, dtype=torch.float32))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# 4. Lightning Module
class RainbowLightning(pl.LightningModule):
    def __init__(self, gamma=0.99, lr=1e-3, batch_size=64, mem_size=5000, n_step=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = RainbowDQN()
        self.target_model = RainbowDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.replay_buffer = PrioritizedReplayBuffer(mem_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.game = Gridworld(size=4, mode='random')
        
        self.total_wins = 0
        self.games_played = 0
        self.cur_steps = 0
        self.beta = 0.4
        
    def forward(self, x):
        return self.model(x)
    
    def select_action(self, state):
        with torch.no_grad():
            # In Rainbow, we keep the model in train mode to use Noisy Layers for exploration
            q_vals = self.model(state)
            return torch.argmax(q_vals).item()
    
    def on_train_batch_start(self, batch, batch_idx):
        self.model.reset_noise()
        self.target_model.reset_noise()

    def training_step(self, batch, batch_idx):
        for _ in range(5): self.populate_buffer()
        
        # Increase beta for IS weights over time
        self.beta = min(1.0, self.beta + 0.0001)
        
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.hparams.batch_size, self.beta)
        states, actions, rewards, next_states, dones, weights = states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device), weights.to(self.device)
        
        Q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            # Double DQN: use model to select action, target_model to evaluate
            next_actions = self.model(next_states).argmax(1)
            Q_targets_next = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            # Multi-step target: R_sum + gamma^n * Q_target(s_n, a_best)
            targets = rewards + (self.hparams.gamma ** self.hparams.n_step) * Q_targets_next * (~dones).float()
            
        td_errors = torch.abs(Q_values - targets).detach()
        loss = (weights * (Q_values - targets)**2).mean()
        
        # Update PER priorities
        self.replay_buffer.update_priorities(indices, td_errors.cpu().numpy() + 1e-6)
        
        self.log('train_loss', loss)
        self.log('win_rate', (self.total_wins/self.games_played) if self.games_played > 0 else 0.0, prog_bar=True)
        
        if self.global_step % 200 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            
        return loss

    def get_n_step_info(self):
        reward = sum([self.hparams.gamma**i * t[2] for i, t in enumerate(self.n_step_buffer)])
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return reward, next_state, done

    def populate_buffer(self):
        state1_ = self.game.board.render_np().reshape(1, 64).astype(float)
        state1 = torch.from_numpy(state1_).float()
        
        action = self.select_action(state1)
        action_map = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
        self.game.makeMove(action_map[action])
        
        reward = self.game.reward()
        state2_ = self.game.board.render_np().reshape(1, 64).astype(float)
        state2 = torch.from_numpy(state2_).float()
        
        self.cur_steps += 1
        # Add 15-step limit to match testing conditions and avoid infinite loops
        done = True if (reward != -1 or self.cur_steps >= 15) else False
        
        self.n_step_buffer.append((state1.squeeze(), action, reward, state2.squeeze(), done))
        
        if len(self.n_step_buffer) == self.hparams.n_step:
            r_n, s_n, d_n = self.get_n_step_info()
            s_start, a_start, _, _, _ = self.n_step_buffer[0]
            self.replay_buffer.append(s_start, a_start, r_n, s_n, d_n)
            
        if done:
            if reward == 10: self.total_wins += 1
            self.games_played += 1
            self.game = Gridworld(size=4, mode='random')
            self.n_step_buffer.clear()
            self.cur_steps = 0

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # We don't use standard DataLoader for PER because we need to update priorities
        # But Lightning requires a dataloader. We'll return a dummy one.
        class DummyDataset(Dataset):
            def __len__(self): return 200
            def __getitem__(self, idx): return (torch.zeros(64), 0, 0.0, torch.zeros(64), False)
        
        while len(self.replay_buffer) < 500:
            self.populate_buffer()
        return DataLoader(DummyDataset(), batch_size=self.hparams.batch_size)

def train_rainbow():
    print("Starting Rainbow DQN Training (Random Mode) using PyTorch Lightning...")
    model = RainbowLightning(lr=1e-3, batch_size=64, n_step=3)
    
    early_stop = EarlyStopping(monitor='train_loss', patience=30, mode='min')
    
    trainer = pl.Trainer(
        max_epochs=100,
        gradient_clip_val=1.0, 
        callbacks=[early_stop],
        enable_checkpointing=False,
        logger=False
    )
    
    trainer.fit(model)
    
    print("Saving Rainbow model to hw3_4_model.pth...")
    torch.save(model.model.state_dict(), 'hw3_4_model.pth')
    
    test_results = test_random(model.model)
    
    # Plotting comparison bar chart
    methods = ['Advanced DQN (HW3-3)', 'Rainbow DQN (HW3-4)']
    
    # Try to read HW3-3 win rate from file, otherwise use default
    hw3_3_win_rate = 0.05
    if os.path.exists('hw3_3_win_rate.txt'):
        with open('hw3_3_win_rate.txt', 'r') as f:
            try:
                hw3_3_win_rate = float(f.read())
            except:
                pass

    win_rates = [hw3_3_win_rate, test_results] 
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, win_rates, color=['#3498db', '#e74c3c'])
    
    plt.title('Win Rate Comparison in Random Mode')
    plt.ylabel('Win Rate')
    plt.ylim(0, max(win_rates) * 1.5) # Leave space for labels
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('hw3_4_comparison_bar.png')
    plt.close()
    print("Saved hw3_4_comparison_bar.png")

def test_random(model):
    print("\nTesting Rainbow DQN on 100 Random Maps...")
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
    win_rate = wins / 100
    print(f"Win Rate: {win_rate:.2%}")
    return win_rate

if __name__ == "__main__":
    train_rainbow()

# Rainbow DQN 元件說明：
# 1. Double DQN：透過主網路選擇動作，目標網路評估動作，解決 Q 值高估問題。
# 2. Dueling Architecture：將 Q 網路拆分為狀態價值 V(s) 與優勢函數 A(s,a)，提升學習效率。
# 3. Prioritized Experience Replay (PER)：根據 TD-error 優先抽取重要的樣本，並使用重要性採樣 (IS) 修正權重。
# 4. Multi-step Returns (N-step)：累積 3 步的獎勵進行更新，加速獎勵傳播，減少偏差。
# 5. Noisy Networks：在全連接層中加入參數化噪聲，實現自動探索，取代傳統的 epsilon-greedy。
