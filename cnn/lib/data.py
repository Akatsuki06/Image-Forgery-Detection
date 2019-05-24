import os
import numpy as np
import pickle
from cnn.config import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
# from PIL import Image
import cv2

from cnn.lib.preprocess import normalizeChannel



class Data:
	"""docstring for Data"""
	def __init__(self):
		super(Data, self).__init__()




	def prepare_dataset(self):

		fake_images = self.get_images(FAKE_PATH)
		real_images = self.get_images(REAL_PATH)
		# fake_masks = self.get_images(fake_mask_path)
		labels = [0]*len(real_images)+[1]*len(fake_images)
		X_train, X_test, y_train, y_test = train_test_split(fake_images+real_images,
							labels, test_size=0.2,
								stratify=labels)

		return X_train,X_test,y_train,y_test





	def get_data_batches(self):
		datagen = ImageDataGenerator()
		train_batches = datagen.flow_from_directory(TRAIN_PATH,
											target_size=(IMAGE_DIM,IMAGE_DIM),
											classes=['real','fake'],
											batch_size = BATCH_SIZE, 
											shuffle = False,
        									class_mode='categorical',
											)
		validatation_batches = datagen.flow_from_directory(VALID_PATH,
											target_size=(IMAGE_DIM,IMAGE_DIM),
											classes=['real','fake'],
											batch_size = BATCH_SIZE, 
											shuffle = False,
        									class_mode='categorical',
											)
		# print(len(train_batches))

		return train_batches,validatation_batches
		# imgs,labels = next(train_batches)

	def get_batch(self,path):
		data_gen = ImageDataGenerator()
		generator = data_gen.flow_from_directory(path,
											target_size=(IMAGE_DIM,IMAGE_DIM),
											classes=['real','fake'],
											batch_size = BATCH_SIZE, 
											shuffle = False,
        									class_mode='categorical',

											)
		return generator


	def get_test_batch(self):
		test_gen = ImageDataGenerator()
		test_generator = test_gen.flow_from_directory(TEST_PATH,
											target_size=(IMAGE_DIM,IMAGE_DIM),
											batch_size = BATCH_SIZE, 
											shuffle = False,
										)

		return test_generator




	def import_image(self,image_url):
		print("current dir: ",os.getcwd())

		# img = Image.open(image_url)
		img = cv2.imread(image_url)
		# img = cv2.resize(img, dsize=(IMAGE_DIM,IMAGE_DIM), interpolation=cv2.INTER_CUBIC)
		# img = img.resize((IMAGE_DIM,IMAGE_DIM), Image.ANTIALIAS)
		# img = img.resize((IMAGE_DIM,IMAGE_DIM))
		img = np.resize(img,(IMAGE_DIM,IMAGE_DIM))
		# img = np.array(img)
		print(img.shape)
		img = normalizeChannel(img)
		img = img.reshape(1,IMAGE_DIM, IMAGE_DIM, IMAGE_CHANNEL)
		# //preprocess()
		return img




	# def plot(self):
	#
