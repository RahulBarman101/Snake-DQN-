from PIL import Image
import numpy as np
import cv2

class snake:
	def __init__(self):
		self.SIZE = 20
		self.FOOD_REWARD = 25
		self.STEP_PENALTY = 1
		self.x1 = np.random.randint(0,self.SIZE)
		self.y1 = np.random.randint(0,self.SIZE)
		self.foodx = np.random.randint(0,self.SIZE)
		self.foody = np.random.randint(0,self.SIZE)
		self.white = (255,255,255)
		self.red = (0,0,255)
		self.green = (0,255,0)

	def action(self,action):
		if action == 0:
			self.move(x=-1,y=0)
		elif action == 1:
			self.move(x=1,y=0)
		elif action == 2:
			self.move(x=0,y=-1)
		elif action == 3:
			self.move(x=0,y=1)

	def move(self,x,y):
		self.x1 += x
		self.y1 += y

	def render(self):
		env = np.zeros((self.SIZE,self.SIZE,3),dtype=np.uint8)
		env[self.foodx][self.foody] = self.green
		env[self.x1][self.y1] = self.red
		img = Image.fromarray(env,'RGB')
		img = img.resize((300,300))
		cv2.imshow('image',np.array(img))
		if cv2.waitKey(1) == ord('q'):
			return

	def reset(self):
		self.x1 = np.random.randint(0,self.SIZE)
		self.y1 = np.random.randint(0,self.SIZE)
		self.foodx = np.random.randint(0,self.SIZE)
		self.foody = np.random.randint(0,self.SIZE)
		return np.array((self.x1-self.foodx,self.y1-self.foody))

	def getImage(self):
		env = np.zeros((self.SIZE,self.SIZE,3),dtype=np.uint8)
		env[self.foodx][self.foody] = self.green
		env[self.x1][self.y1] = self.red
		img = Image.fromarray(env,'RGB')
		return img

	def step(self,action):
		self.action(action)

		if self.x1 == self.foodx and self.y1 == self.foody:
			print('\nyumm!!!')

		reward = -self.STEP_PENALTY
		done = False
		if self.x1 == self.foodx and self.y1 == self.foody:
			reward = self.FOOD_REWARD
			self.foodx = np.random.randint(0,self.SIZE)
			self.foody = np.random.randint(0,self.SIZE)
		elif self.x1 < 0 or self.y1 < 0:
			self.x1,self.y1 = 0,0
			done = True
		elif self.x1 == self.SIZE or self.y1 == self.SIZE:
			self.x1,self.y1 = self.SIZE-1,self.SIZE-1
			done = True

		# return (np.array(self.getImage()),reward,done)
		return ((self.x1-self.foodx,self.y1-self.foody),reward,done)
