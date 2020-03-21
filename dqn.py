import tensorflow as tf
import numpy as np
from OOPSnake import snake
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten
from collections import deque
import random
from tqdm import tqdm
from tensorflow.keras import Model,Input

s = snake()
NUM_ACTIONS = 4
BATCH_SIZE = 64
NUM_EPISODES = 10000
NUM_STEPS = 200
statex = s.reset()
statex = np.array(statex).reshape(1,-1)

OBSERVATION_SPACE = s.reset().shape

class Agent:
	def __init__(self):
		# self.dqn_model = self.CNNModel()
		# self.target_model = self.CNNModel()
		self.dqn_model = self.DenseModel()
		self.target_model = self.DenseModel()
		self.memory = deque(maxlen=1000)
		self.epsilon = 1.0
		self.decay = 0.995
		self.gamma = 0.99

	def DenseModel(self):
		model = Sequential()
		model.add(Dense(64,activation='relu',input_shape=OBSERVATION_SPACE))
		model.add(Dense(64,activation='relu'))
		model.add(Dense(NUM_ACTIONS,activation='linear'))
		model.compile(loss='mse',optimizer='RMSprop')

		return model		

	def update_model(self):
		# self.target_model.build()
		self.target_model.set_weights(self.dqn_model.get_weights())

	def store(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))


	def CNNModel(self):
		model = Sequential()
		model.add(Input(shape=OBSERVATION_SPACE))
		model.add(Conv2D(256,(3,3)))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))

		model.add(Conv2D(256,(3,3)))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.2))

		model.add(Flatten())
		model.add(Dense(64,activation='relu'))
		model.add(Dense(NUM_ACTIONS,activation='linear'))
		model.compile(loss='mse',optimizer='RMSprop')

		return model

	def action_taken(self,state):
		if np.random.random() <= self.epsilon:
			z = np.random.randint(0,4)
			return z
		else:
			z = self.dqn_model.predict(np.expand_dims(np.array(state),axis=0))[0]
			return np.argmax(z)

	def train(self):
		if len(self.memory) < BATCH_SIZE:
			return
		minibatch = random.sample(self.memory,BATCH_SIZE)
		current_states = np.zeros((BATCH_SIZE,4))
		current_states = np.zeros((BATCH_SIZE,4))

		current_states = np.array([i[0] for i in minibatch],dtype=np.float32)
		current_qs_list = self.dqn_model.predict(current_states)

		new_states = np.array([i[3] for i in minibatch],dtype=np.float32)
		future_qs_list = self.target_model.predict(new_states)

		x = []
		y = []

		for index,(current_state,action,reward,new_state,done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + self.gamma * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			x.append(current_state)
			y.append(current_qs)

		self.dqn_model.compile(loss='mse',optimizer='RMSprop')
		self.dqn_model.fit(np.array(x),np.array(y),batch_size = BATCH_SIZE,verbose=0)

agent = Agent()

for e in tqdm(range(NUM_EPISODES)):
	tot_reward = 0
	state = s.reset()/255.0
	state = np.array(state)
	steps = 0
	for i in range(NUM_STEPS):
		action = agent.action_taken(state)
		# s.render()
		next_state,reward,done = s.step(action)
		tot_reward += reward
		steps += 1
		next_state = np.array(next_state)/255.0
		agent.store(state,action,reward,next_state,done)

		if len(agent.memory) >= BATCH_SIZE:
			agent.train()

		state = next_state

		if done:
			agent.update_model()
			break

	agent.epsilon *= agent.decay
	print(f'\ne : {e+1} \treward : {tot_reward} \tsteps : {steps}')
