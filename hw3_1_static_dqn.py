import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os

# Add Chapter 3 to sys.path to import Gridworld
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'DeepReinforcementLearningInAction', 'Chapter 3')))
from Gridworld import Gridworld

def train_dqn():
    # 1. 環境設定 (Static Mode)
    game = Gridworld(size=4, mode='static')
    
    # 2. 模型架構
    l1 = 64 # Input size (4x4x4)
    l2 = 150
    l3 = 100
    l4 = 4  # Action set: 0:u, 1:d, 2:l, 3:r
    
    model = nn.Sequential(
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, l3),
        nn.ReLU(),
        nn.Linear(l3, l4)
    )
    
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. 超參數
    gamma = 0.9
    epsilon = 1.0
    batch_size = 32
    mem_size = 1000
    replay = deque(maxlen=mem_size)
    epochs = 1000
    max_moves = 50
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    losses = []
    
    print("Starting training...")
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        # Get state and add some noise as in the book example to help with identical states if needed
        # but for basic DQN we can just use the flat representation.
        state1_ = game.board.render_np().reshape(1, 64).astype(float)
        state1 = torch.from_numpy(state1_).float()
        
        status = 1
        mov = 0
        
        while status == 1:
            mov += 1
            # Epsilon-greedy action selection
            qval = model(state1)
            qval_ = qval.data.numpy()
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
            
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64).astype(float)
            state2 = torch.from_numpy(state2_).float()
            
            reward = game.reward()
            
            # Check if game over
            done = True if reward != -1 else False
            
            # Store experience
            exp = (state1, action_, reward, state2, done)
            replay.append(exp)
            
            state1 = state2
            
            # 4. Experience Replay Training
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                
                # Prepare batches
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
                
                # Predict Q values for current states
                Q1 = model(state1_batch)
                
                # Predict Q values for next states (Target network not used in basic DQN)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                # Calculate target Q values
                # Target = r + gamma * max(Q(s')) if not done, else r
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                
                # Get Q values for the actions taken
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            if done or mov > max_moves:
                status = 0
                
        # Decay epsilon
        if epsilon > 0.1:
            epsilon -= (1.0 / epochs)
            
        if i % 100 == 0:
            print(f"Epoch {i}/{epochs}, Epsilon: {epsilon:.2f}")

    print("Training finished.")
    
    # 儲存權重檔
    print("Saving model to hw3_1_model.pth...")
    torch.save(model.state_dict(), 'hw3_1_model.pth')
    
    # Verify performance
    win = test_model(model)
    print(f"Final test on static environment: {'Won' if win else 'Lost'}")
    
    # Save model
    # torch.save(model.state_dict(), 'HW3_Solutions/dqn_static.pth')

def test_model(model):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    game = Gridworld(size=4, mode='static')
    state_ = game.board.render_np().reshape(1, 64).astype(float)
    state = torch.from_numpy(state_).float()
    
    print("Initial State:")
    print(game.display())
    
    status = 1
    mov = 0
    while status == 1 and mov < 15:
        qval = model(state)
        action_ = np.argmax(qval.data.numpy())
        action = action_set[action_]
        print(f"Move {mov}: {action}")
        game.makeMove(action)
        print(game.display())
        
        reward = game.reward()
        if reward != -1:
            if reward > 0:
                print("Game Won!")
                return True
            else:
                print("Game Lost!")
                return False
        
        state_ = game.board.render_np().reshape(1, 64).astype(float)
        state = torch.from_numpy(state_).float()
        mov += 1
    
    print("Failed to win within 15 moves.")
    return False

if __name__ == "__main__":
    train_dqn()
