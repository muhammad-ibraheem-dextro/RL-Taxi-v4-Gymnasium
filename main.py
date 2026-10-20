import gymnasium as gym
import numpy as np
from agent import QLearningAgent
from utils import plot_training_progress, record_gif

def train():
    # 1. Environment aur Agent banayein (Ab yahan Taxi-v4 use ho raha hai)
    env = gym.make("Taxi-v4")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    agent = QLearningAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.99)
    
    episodes = 2500
    rewards_history = []

    print("Training Started! Please wait...")
    
    # 2. Training Loop
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # Har 500 episodes ke baad progress print karein
        if (ep + 1) % 500 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode: {ep+1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # 3. Model aur Graph save karein
    agent.save_model("q_table.npy")
    print("Model weights saved as q_table.npy")
    plot_training_progress(rewards_history)
    env.close()
    
    # 4. GIF Record karein (Yahan bhi Taxi-v4 use hoga)
    env_render = gym.make("Taxi-v4", render_mode="rgb_array")
    record_gif(env_render, agent)
    env_render.close()
    
    print("All tasks completed successfully!")

if __name__ == "__main__":
    train()