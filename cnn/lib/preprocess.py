
# to grayscale or to binary
# noise removal
# edge detection
# resize
# image augmentation
# median filtering



import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from cnn.config import FAKE_PATH,  REAL_PATH
import numpy as np

def normalizeChannel(img):
	output = np.zeros_like(img)
	print("Normalising")
	if len(img.shape)<3:# one channel
		output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		print("single channel image")
	if len(img.shape)==3:
		if img.shape[-1]==1:
			print("single channel image")
			output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		elif img.shape[-1]==2:
			print("2 channel image")
			output[:,:,0] = img[:,:,0]
			output[:,:,1] = img[:,:,1]
		elif img.shape[-1]==4:
			print("4 channel image")
			output = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	return output


def get_images(path):
	images = os.listdir(path)
	print('len of images',len(images))
	output = []

	for image in images:
		img = cv2.imread(path+image)
		if len(img.shape)<3:
			continue
		if img.shape[2]==4:
			continue
		output.append(image)
	return output


def zwhiten():

	datagen = ImageDataGenerator(zca_whitening=True)

	all_real_images = get_images(REAL_PATH)
	print(all_real_images)

	datagen.fit(X_train)


	for real_image in all_real_images:
		img = load_img('{0}{1}'.format(REAL_PATH,real_image))
		img_arr = img_to_array(img)
		img_arr = img_arr.reshape((1,) + img_arr.shape)
		for batch in datagen.flow(
		    img_arr,
		    batch_size=1,
		    # target_size=(224,224,3),
		    save_to_dir='cnn/data/augmented/real/',
		    save_prefix='aug_zca_{0}'.format(real_image),
		    save_format='jpeg'):
		   pass





def guassianBlur(img):
	blur = cv2.GaussianBlur(res,(15,15),0)
	return blur


def median(img):
	median = cv2.medianBlur(img,15)
	return median


def augmentation_rotation():
	datagen = ImageDataGenerator(
		    rotation_range=45,
		    width_shift_range=0.25,
		    height_shift_range=0.2,
		    shear_range=0.2,
		    zoom_range=0.25,
		    horizontal_flip=True,
		    fill_mode='nearest'
		    )

	all_real_images = get_images(REAL_PATH)
	all_fake_images = get_images(FAKE_PATH)
	# print(all_real_images)

	for real_image in all_real_images:
		img = load_img('{0}{1}'.format(REAL_PATH,real_image))
		img_arr = img_to_array(img)
		img_arr = img_arr.reshape((1,) + img_arr.shape)
		i = 0
		for batch in datagen.flow(
		    img_arr,
		    batch_size=2,
		    save_to_dir='data/augmented/real/',
		    save_prefix='aug_rot_{0}'.format(real_image),
		    save_format='jpeg'):
		    i += 1
		    if i > 3:
		        break

	for fake_image in all_fake_images:
		img = load_img('{0}{1}'.format(FAKE_PATH,fake_image))
		img_arr = img_to_array(img)
		img_arr = img_arr.reshape((1,) + img_arr.shape)
		i = 0
		for batch in datagen.flow(
		    img_arr,
		    batch_size=2,
		    save_to_dir='data/augmented/fake/',
		    save_prefix='aug_rot_{0}'.format(fake_image),
		    save_format='jpeg'):
		    i += 1
		    if i > 3:
		        break





# augmentation_rotation()
