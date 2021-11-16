import gym
from stable_baselines3 import PPO, A2C

# create the environment
env = gym.make('CartPole-v1')
env.reset()

# instantiate the learning algorithm for the cartpole environment and learn
# according to this paper: https://arxiv.org/pdf/1810.01940.pdf 
# actor critic models like A2C work well on cartpole
# value function works better, but is not in stable baselines
model = A2C("MlpPolicy", env)

# Use PPO https://arxiv.org/pdf/1707.06347.pdf: better performance than A2C on more complicated Atari games
# model = PPO("MlpPolicy", env)
model.learn(total_timesteps=10000)

model.save("./TrainedModels/cartpole_a2c")

# assume that the model has been trained
#model = A2C.load("./TrainedModels/cartpole_a2c")
#model = PPO.load("./TrainedModels/cartpole_a2c")

# use trained model to generate data
# need state, action, and image sequences ie. at each timestep: https://github.com/clvrai/spirl#adding-a-new-dataset-for-model-training
# will need to format dataloader to take this data as input
timesteps = 1000
obs=env.reset()
with open("./data/cartpole_a2c_data.txt", "w") as f:
    for t in range(timesteps):
        action, states = model.predict(obs, deterministic=True)
        observation, reward, done, info = env.step(action)
        image = env.render(mode="rgb_array")
        env.render # for visualization
        out = ",".join([str(t), str(states), str(image)])
        f.write(out)
        f.write("\n")