
import pickle
import numpy
import os
import numpy as np
import time
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2

from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras import backend as K
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from cnn.config import *
MODEL_PATH = "cnn/model/"

class CNNModel:
	"""docstring for CNNModel"""
	def __init__(self):
		super(CNNModel, self).__init__()
		self.model = self.get_model()


	def train(self,args):
		self.model.fit_generator(**args)
		# self.model.fit_generator(train_samples,epochs=args['epochs'],
		# 						steps_per_epoch=args['steps_per_epoch'],verbose=2)


	def predict(self,test_samples,batch_size=1):
		return self.model.predict(test_samples,batch_size=batch_size)


	def predict_classes(self,predictions):
		rounded_predictions = []

		for pred in predictions:
			rounded_predictions.append(np.argmax(pred))

		return rounded_predictions

	def evaluate_samples(self,generator,steps):
		self.model.evaluate_generator(generator,steps)

	def predict_samples(self,test_samples,steps=1,verbose=1):
		return self.model.predict_generator(test_samples,steps=len(test_samples)/BATCH_SIZE,verbose = verbose)

	def compile(self,args):
		# lr=0.001,loss='categorical_crossentropy',metrics ='accuracy'
		self.model.compile(**args)
		# self.model.compile(Adam(lr=args['learning_rate']),
		# 						loss=args['loss'],
		# 						metrics=[args['metrics']]
		# 						)


	def load(self,name):
		self.model= load_model(name)


	def save(self):
		self.model.save(MODEL_PATH+'{0}.h5'.format(time.time()))


	def get_model_vgg16(self):
		vgg16_model = VGG16(input_shape=(64,64,3))
		model = Sequential()
		for layer in vgg16_model.layers:
			model.add(layer)

		model.layers.pop()

		for layer in model.layers:
			layer.trainable=False

		model.add(Dense(2,activation='softmax'))

		return model


	def get_model(self):
		target_size = 96
		input_tensor = Input(shape=(IMAGE_DIM, IMAGE_DIM, 3))
		base_model = MobileNetV2(
	        include_top=False,
	        weights='imagenet',
	        input_tensor=input_tensor,
	        input_shape=(IMAGE_DIM, IMAGE_DIM, 3),
	        pooling='avg')
		for layer in base_model.layers:
			layer.trainable = False
		op = Dense(256, activation='relu')(base_model.output)
		op = Dropout(.25)(op)
		output_tensor = Dense(2, activation='softmax')(op)
		model = Model(inputs=input_tensor, outputs=output_tensor)
		return model

	def naive_model(self):
		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(96,96,3),activation='relu'))
		model.add(Conv2D(32, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Conv2D(64, (3, 3),activation='relu'))
		model.add(Conv2D(64, (3, 3),activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(256,activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2,activation='softmax'))

		return model
