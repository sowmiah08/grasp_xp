import gym
from stable_baselines3 import PPO

from src.env.relational_pick_and_place_env import RelationalPickAndPlaceEnv


def main():
    # Create the environment
    env = RelationalPickAndPlaceEnv(render=True)

    # Loading the trained policy
    model = PPO.load("trained_model.zip")

    # Run for a few episodes to see how it performs
    for episode in range(5):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        print(f"Episode {episode+1} finished with reward {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()
