import matplotlib.pyplot as plt
import imageio
import numpy as np

def plot_training_progress(rewards, window=100, filename="training_progress.png"):
    print("Generating training graph...")
    moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', alpha=0.3)
    plt.plot(moving_avg, label=f'Moving Average ({window} ep)', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Q-Learning Training Progress on Taxi-v4')
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close() # Yeh line program ko atakne se bachati hai
    print(f"Graph saved successfully as {filename}")

def record_gif(env, agent, filename="taxi_agent.gif", max_steps=100):
    print("Recording GIF of the trained agent...")
    frames = []
    state, _ = env.reset()
    for _ in range(max_steps):
        frames.append(env.render())
        action = agent.choose_action(state, exploit_only=True)
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break
            
    imageio.mimsave(filename, frames, fps=10)
    print(f"GIF saved successfully at {filename}")