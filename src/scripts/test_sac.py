import gymnasium as gym
import torch
from stable_baselines3 import SAC
from env.relational_pick_and_place_env import RelationalPickAndPlaceEnv  # Adjust if your import path differs
import time
import numpy as np

def evaluate(model_path, num_episodes=5):
    # Create the environment with rendering ON
    env = RelationalPickAndPlaceEnv(render=True)

    # Load the trained model
    model = SAC.load(model_path)

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Model predicts the action given the current observation
            action, _states = model.predict(obs, deterministic=True)

            # Step the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            time.sleep(0.01)  # Slow down to real-time approx

        print(f"Episode {ep+1} finished with reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    model_path = "sac_franka_model.zip"  # Put your model file path here
    evaluate(model_path)
