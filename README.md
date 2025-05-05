# CartPole-DQN-PyTorch

A PyTorch implementation of Deep Q-Network (DQN) to solve the CartPole-v1 environment from OpenAI Gymnasium.

![CartPole Environment](https://gymnasium.farama.org/_images/cart_pole.gif)

## Project Overview

This project implements a Deep Q-Network (DQN) to solve the CartPole-v1 reinforcement learning environment. The agent successfully learns to balance the pole on the cart, achieving the maximum possible score of 500 points.

### Key Features:
- Experience replay for stable learning
- Target network to reduce the moving target problem
- Epsilon-greedy exploration strategy
- Neural network approximation of Q-values

## Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/CartPole-DQN-PyTorch.git
cd CartPole-DQN-PyTorch

# Install dependencies
pip install torch numpy gymnasium
```

## How to Run

Run the main script to train and evaluate the agent:

```bash
python cartpole_pytorch.py
```

This will:
1. Train the DQN agent for 400 episodes
2. Show training progress with rewards and exploration rate
3. Evaluate the trained model with visual rendering

## Implementation Details

### Components:
- **Replay Buffer**: Stores experiences (state, action, next_state, reward, done) for off-policy learning
- **DQN Model**: A neural network with two hidden layers (64 neurons each)
- **Target Network**: A separate network for stable Q-value targets
- **Epsilon-Greedy Policy**: Balances exploration and exploitation

### Hyperparameters:
- Learning rate: 0.001
- Discount factor (gamma): 0.99
- Epsilon decay: 0.995
- Replay buffer size: 10,000
- Batch size: 64
- Target network update frequency: 10 episodes

## Results

The agent successfully learns to balance the pole, achieving the maximum reward of 500 consistently after approximately 380 episodes of training.

### Training Progress:
- Initial episodes: Mostly random exploration (< 50 reward)
- Middle episodes: Beginning to learn useful patterns (100-200 rewards)
- Final episodes: Optimal policy (500 rewards)

### Evaluation Results:
- Episode 1: 500.0000
- Episode 2: 500.0000
- Episode 3: 500.0000
- Episode 4: 500.0000
- Episode 5: 500.0000

## Project Structure

- `cartpole_pytorch.py`: Main script containing the DQN implementation

## References

- [OpenAI Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) - Environment details
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
