import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    epsilons = []
    
    print("Starting training...")
    
    for i in range(epochs):
        epsilons.append(epsilon)
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
    
    # 儲存 epsilon 曲線
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons)
    plt.title('Epsilon Decay Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    plt.grid(True)
    plt.text(len(epsilons)*0.6, 0.8, 'Exploration to Exploitation', fontsize=12, color='blue')
    plt.savefig('hw3_1_epsilon_decay.png')
    plt.close()
    print("Saved hw3_1_epsilon_decay.png")

    # Verify performance
    win = test_model(model)
    print(f"Final test on static environment: {'Won' if win else 'Lost'}")
    return model

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

def visualize_path(model):
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    game = Gridworld(size=4, mode='static')
    
    # Gridworld board layout for static mode
    # 0: empty, 1: player, 2: wall, 3: pit, 4: goal (approx based on typical Gridworld)
    # However, let's just use the display() output to identify elements or render_np
    
    path = []
    state_ = game.board.render_np().reshape(1, 64).astype(float)
    state = torch.from_numpy(state_).float()
    
    # Find initial positions of items
    # In Chapter 3 Gridworld, components are stored in game.components
    # But let's keep it simple and just record player positions (row, col)
    # The board is 4x4, render_np is 4x4x4 (one-hot for player, wall, pit, goal)
    
    def get_pos(game):
        for name, obj in game.board.components.items():
            if name == 'Player': return obj.pos
        return (0,0)

    def get_all_objects(game):
        objs = {}
        for name, obj in game.board.components.items():
            objs[name] = obj.pos
        return objs

    objects = get_all_objects(game)
    path.append(objects['Player'])
    
    status = 1
    mov = 0
    while status == 1 and mov < 15:
        qval = model(state)
        action_ = np.argmax(qval.data.numpy())
        action = action_set[action_]
        game.makeMove(action)
        path.append(get_pos(game))
        
        reward = game.reward()
        if reward != -1:
            status = 0
        
        state_ = game.board.render_np().reshape(1, 64).astype(float)
        state = torch.from_numpy(state_).float()
        mov += 1

    # Plotting the grid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(True)
    
    # Gridworld coordinates are (row, col), (0,0) is top-left
    # Matplotlib coordinates (x, y), (0,0) is bottom-left
    # We need to convert: x = col, y = 3 - row
    
    def convert(pos):
        return (pos[1] + 0.5, 3 - pos[0] + 0.5)

    # Draw objects
    for name, pos in objects.items():
        x, y = convert(pos)
        if name == 'Player':
            ax.add_patch(patches.Circle((x, y), 0.3, color='blue', label='Start (P)'))
            ax.text(x, y, 'P', ha='center', va='center', color='white', fontweight='bold')
        elif name == 'Goal':
            ax.add_patch(patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, color='green'))
            ax.text(x, y, '+', ha='center', va='center', color='white', fontsize=20, fontweight='bold')
        elif name == 'Pit':
            ax.add_patch(patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, color='red'))
            ax.text(x, y, '-', ha='center', va='center', color='white', fontsize=20, fontweight='bold')
        elif name == 'Wall':
            ax.add_patch(patches.Rectangle((x-0.4, y-0.4), 0.8, 0.8, color='gray'))
            ax.text(x, y, 'W', ha='center', va='center', color='white', fontweight='bold')

    # Draw path
    for i in range(len(path) - 1):
        p1 = convert(path[i])
        p2 = convert(path[i+1])
        ax.annotate('', xy=p2, xytext=p1, arrowprops=dict(arrowstyle='->', lw=3, color='orange'))

    plt.title('HW3-1: Agent Path Visualization')
    plt.gca().invert_yaxis() # To match grid display where (0,0) is top
    # Wait, if I use 3-row, then 0 is at y=3. (3-0=3). 3 is at y=0. (3-3=0).
    # So y=3 is top, y=0 is bottom. This is standard.
    # Actually, let's just use row, col and invert y axis.
    ax.set_ylim(4, 0)
    ax.set_xlim(0, 4)
    
    plt.savefig('hw3_1_path_visualization.png')
    plt.close()
    print("Saved hw3_1_path_visualization.png")

if __name__ == "__main__":
    model = train_dqn()
    visualize_path(model)
