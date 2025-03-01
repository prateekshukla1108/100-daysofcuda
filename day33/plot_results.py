import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('training_log.csv')
plt.figure(); plt.plot(data['Episode'], data['TotalReward']); plt.xlabel('Episode'); plt.ylabel('Total Reward'); plt.title('Episode Reward Curve'); plt.savefig('episode_reward.png')
plt.figure(); plt.plot(data['Episode'], data['Steps']); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.title('Steps per Episode'); plt.savefig('steps_per_episode.png')
plt.figure(); plt.plot(data['Episode'], data['Epsilon']); plt.xlabel('Episode'); plt.ylabel('Epsilon'); plt.title('Epsilon Decay Curve'); plt.savefig('epsilon_decay.png')
