import gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from env.relational_pick_and_place_env import RelationalPickAndPlaceEnv



# Create the training environment wrapped with Monitor for logging
env = RelationalPickAndPlaceEnv(render=False)
env = Monitor(env)

# Initialize the SAC model with the environment
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("sac_franka_model.zip")

print("Training finished")

# Close the environment
env.close()
