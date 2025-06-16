import gym
from stable_baselines3 import PPO
from src.env.relational_pick_and_place_env import RelationalPickAndPlaceEnv

def main():
    env = RelationalPickAndPlaceEnv(render=False)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("trained_model.zip")

    env.close()


if __name__ == "__main__":
    main()
