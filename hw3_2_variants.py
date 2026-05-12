import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os
import copy

# Add Chapter 3 to sys.path to import Gridworld
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'DeepReinforcementLearningInAction', 'Chapter 3')))
from Gridworld import Gridworld

# 1. Basic DQN Model
class BasicDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(BasicDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 2. Dueling DQN Model
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )
        
    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

def train(model_type='basic', use_double=False, epochs=1000):
    game = Gridworld(size=4, mode='player')
    
    if model_type == 'dueling':
        model = DuelingDQN()
    else:
        model = BasicDQN()
        
    target_model = copy.deepcopy(model)
    target_model.eval()
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    gamma = 0.9
    epsilon = 0.5 # Start with lower epsilon for player mode since it's harder? Actually 1.0 is standard.
    # But let's use 0.5 to speed up convergence as it's a small grid.
    epsilon = 1.0
    batch_size = 32
    mem_size = 1000
    replay = deque(maxlen=mem_size)
    max_moves = 50
    update_target_freq = 100 # Steps to update target network
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []
    total_wins = 0
    win_record = []
    
    steps = 0
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1, 64).astype(float)
        state1 = torch.from_numpy(state1_).float()
        
        status = 1
        mov = 0
        while status == 1:
            steps += 1
            mov += 1
            
            qval = model(state1)
            if random.random() < epsilon:
                action_index = np.random.randint(0, 4)
            else:
                action_index = np.argmax(qval.data.numpy())
            
            action = action_set[action_index]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64).astype(float)
            state2 = torch.from_numpy(state2_).float()
            
            reward = game.reward()
            done = True if reward != -1 else False
            
            replay.append((state1, action_index, reward, state2, done))
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                s1_batch = torch.cat([s[0] for s in minibatch])
                a_batch = torch.Tensor([s[1] for s in minibatch]).long()
                r_batch = torch.Tensor([s[2] for s in minibatch])
                s2_batch = torch.cat([s[3] for s in minibatch])
                d_batch = torch.Tensor([s[4] for s in minibatch])
                
                Q1 = model(s1_batch)
                
                with torch.no_grad():
                    if use_double:
                        # Double DQN: Use online model to select action, target model to evaluate
                        next_actions = torch.argmax(model(s2_batch), dim=1)
                        Q2_target = target_model(s2_batch)
                        Q2_max = Q2_target.gather(1, next_actions.unsqueeze(1)).squeeze()
                    else:
                        # Basic DQN: Use target model for both
                        Q2_target = target_model(s2_batch)
                        Q2_max = torch.max(Q2_target, dim=1)[0]
                
                targets = r_batch + gamma * (1 - d_batch) * Q2_max
                X = Q1.gather(1, a_batch.unsqueeze(1)).squeeze()
                
                loss = loss_fn(X, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # Update target network
            if steps % update_target_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
            if done or mov > max_moves:
                if reward == 10:
                    total_wins += 1
                status = 0
        
        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)
            
        if (i + 1) % 100 == 0:
            win_rate = total_wins / 100.0
            win_record.append(win_rate)
            # print(f"Epoch {i+1}, Win Rate (Last 100): {win_rate:.2f}, Epsilon: {epsilon:.2f}")
            total_wins = 0
            
    # 儲存最佳模型權重
    print(f"Saving model to hw3_2_model.pth...")
    torch.save(model.state_dict(), 'hw3_2_model.pth')
    
    return win_record

def compare_variants():
    print("Comparing DQN Variants in Player Mode...")
    epochs = 1000
    
    print("\nTraining Basic DQN...")
    basic_wins = train(model_type='basic', use_double=False, epochs=epochs)
    
    print("Training Double DQN...")
    double_wins = train(model_type='basic', use_double=True, epochs=epochs)
    
    print("Training Dueling DQN + Double DQN...")
    dueling_wins = train(model_type='dueling', use_double=True, epochs=epochs)
    
    print("\nComparison Results (Win Rate per 100 epochs):")
    print("Epochs\tBasic\tDouble\tDueling+Double")
    for i in range(len(basic_wins)):
        print(f"{(i+1)*100}\t{basic_wins[i]:.2f}\t{double_wins[i]:.2f}\t{dueling_wins[i]:.2f}")

if __name__ == "__main__":
    compare_variants()
