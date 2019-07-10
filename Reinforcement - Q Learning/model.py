# Reference https://github.com/keon/deep-q-learning/blob/master/dqn.py

import random
import gym
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995

model = Sequential()
model.add(Dense(256, input_dim=state_size, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer='adam')

if os.path.isfile('model.h5'):
    model.load_weights("model.h5")
else:
    done = False

    for e in range(1000):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        memory = []
        for time in range(500):
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                action = np.argmax(model.predict(state)[0])

            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            target = reward
            if not done:
                target = (reward + gamma * np.amax(model.predict(next_state)[0]))
            target_f = model.predict(state)
            target_f[0][action] = target
            model.train_on_batch(state, target_f)
            if epsilon > 0.01:
                epsilon *= epsilon_decay

            state = next_state
            if done:
                print("episode: {}, score: {}".format(e, time))
                break
    model.save_weights("model.h5")

done = False
state = env.reset()
state = np.reshape(state, [1, state_size])
i = 0
while not done:
    env.render()
    action = np.argmax(model.predict(state)[0])
    next_state, _, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_size])
    state = next_state
    i += 1
print("score: {}".format(i))
