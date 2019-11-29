from obstacle_avoidance_env import ArmEnv
from DPPO import PPO
import tensorflow as tf

EP_MAX = 20
EP_LEN = 300
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0005                # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 5             # loop update operation n-steps
EPSILON = 0.2               # Clipped surrogate objective
MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])
GLOBAL_PPO = PPO()
saver = tf.train.Saver()
saver.restore(GLOBAL_PPO.sess, './weak_avoidance_model/PPO_model')
env.set_fps(30)
while True:
    s = env.reset()
    for t in range(400):
        env.render()
        s = env.step(GLOBAL_PPO.choose_action(s))[0]