import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import random
import numpy as np
import time

# Setting up the environment and device
env = gym.make('CartPole-v1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Creating memory buffer for experience replay
class ReplayBuffer:

    def __init__(self,capacity):
        self.dq = deque( [], maxlen=capacity) 

    def add( self, state,action,next_state,reward,done ):
        self.dq.append( (state,action,next_state,reward,done) )

    def sample(self, batch_size ):
        return random.sample( self.dq, batch_size ) 

    def __len__(self):
        return len(self.dq)

# Creating the DQN Model
class DQN( nn.Module ):

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear( in_dims, 64 ),
            nn.ReLU(),
            nn.Linear( 64, 64 ),
            nn.ReLU(),
            nn.Linear( 64, out_dims ),
        )

    def forward(self,x):
        return self.network(x)
    
# Set seeds for reproducibility
def set_seeds(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Epsilon Greedy Action Selection
def action_selection( state, epsilon ) :
    if random.random() > epsilon:
        # expoitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor( state ).unsqueeze(0).to(device)
            return policy_net( state_tensor ).max(1)[1].item()
    else:
        # exploration
        return env.action_space.sample()
    
    
# Hyperparameters
min_memory = 1000
max_memory = 10000
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10
learning_rate = 0.001

# Initialize the replay buffer
buffer = ReplayBuffer( max_memory ) 

# Initialize the DQN model ( ie. policy network and target network )
state_dimensions = env.observation_space.shape[0]
action_dimensions = env.action_space.n


# Note : here we define two networks. the target network is the delayed copy of the policy network.
# The target network is used to avoid moving target problem. ( Provides stable Q-values targets for learning by not imediately updating the network after every step. )

policy_net = DQN( state_dimensions, action_dimensions ).to(device) 
target_net = DQN( state_dimensions, action_dimensions ).to(device) 

# Load the weights and set target_net to eval mode
target_net.load_state_dict( policy_net.state_dict() )
target_net.eval()

# Initialize the optimizer and loss function
optimizer = torch.optim.Adam( policy_net.parameters() , lr= learning_rate ) 
critetion = nn.MSELoss()


def Train(episodes = 400):

    epsilon = epsilon_start
    
    for episode in range(episodes):
        
        total_reward = 0 
        done = False
        state,_ = env.reset()
    
        while not done : 
    
            action = action_selection( state, epsilon ) 
            new_state, reward, truncated, terminated, _ = env.step(action)
            done = truncated or terminated
            buffer.add( state,action,new_state,reward,done ) 
            state = new_state
            total_reward += reward
    
    
            if len(buffer) > min_memory :
    
                data = buffer.sample( batch_size ) 
                state_arr, action_arr, new_state_arr, reward_arr, done_arr = zip(*data)
    
                # converting data into tensor 
                state_tensors = torch.FloatTensor( np.array(state_arr) ).to(device)
                action_tensors = torch.LongTensor( np.array(action_arr) ).unsqueeze(1).to(device)
                new_state_tensors = torch.FloatTensor( np.array(new_state_arr) ).to(device)
                reward_tensors = torch.FloatTensor( np.array(reward_arr) ).to(device)
                done_tensors = torch.FloatTensor( np.array(done_arr) ).to(device)
    
    
                # Current Q Values
                current_q_values = policy_net( state_tensors ) 
                current_q_values = current_q_values.gather(1,action_tensors)
    
                # maximum Q Values
                with torch.no_grad():
                    max_next_q_values = target_net( new_state_tensors )
                    max_next_q_values = max_next_q_values.max(1)[0]
    
                # expected Q Values
                expected_q_values = reward_tensors + max_next_q_values * gamma * ( 1 - done_tensors ) 
    
                # calculate loss
                loss = critetion( current_q_values.squeeze(), expected_q_values ) 
    
                # optimizing the model 
                optimizer.zero_grad()
                loss.backward()
                for i in policy_net.parameters():
                    i.grad.data.clamp_(-1,1)
                optimizer.step()
    
            if done:
                break
        
        if episode % target_update == 0 :
            target_net.load_state_dict( policy_net.state_dict() ) 
        
        epsilon = max( epsilon_end, epsilon * epsilon_decay ) 
        
        if (episode + 1) % 10 == 0:
            print(f"Episode : {episode + 1} Total Reward : {total_reward:.4f} Epsilon : {epsilon:.4f}")
			
def Evaluate( evaluation_episodes = 5 ) :
    print('Evaluation started...')
    eval_env = gym.make('CartPole-v1',render_mode='human')
    
    for episode in range(evaluation_episodes):
        total_reward = 0 
        state,_ = eval_env.reset()
        done = False
        while not done:
            time.sleep(0.03)
            with torch.no_grad():
                state_tensor = torch.FloatTensor( state ).unsqueeze(0).to(device)
                action = policy_net( state_tensor ).max(1)[1].item()
            state,reward,terminated,truncated,_ = eval_env.step(action)
            done = truncated or terminated
            total_reward += reward
            if done :
                break
        print(f"Episode : {(episode + 1)} Total_Reward : {total_reward:.4f} ")
    eval_env.close()
		
try:
    set_seeds()
    print("Starting training...")
    Train()
    Evaluate()
finally:
    env.close()


