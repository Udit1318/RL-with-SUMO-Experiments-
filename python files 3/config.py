import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters (separated from init)
alpha = 0.04
gamma = 0.88
epsilon = 0.95
epsilon_decay = 0.995
epsilon_min = 0.01
train_episodes = 300
test_episodes = 100
episode_length = 10800
num_runs = 10
sumoBinary = "sumo"
sumoConfig = r"/home/narayan/Desktop/My Sumo/Stage 13(Homogeneous Network)/SUMO files/stage13.sumocfg"
