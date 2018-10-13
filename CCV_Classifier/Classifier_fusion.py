# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:28:41 2016

@author: ljm
"""

import pickle
import os
import tensorflow as tf
import sys
import time
import numpy as np
import random
from keras import Sequential
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
## Works Model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## Dir
basepath = 'caches/'
BGDatapath = basepath + 'BackgroundData_fc8'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'FlowData_lstm1'
ObjDatapath = basepath + 'Obj_feature'
PoseDatapath = basepath + 'pose_feature_CCV'


catagories = ['Basketball','Baseball','Beach','Biking','Birthday','Graduation','IceSkating','NonMusicPerformance','Parade','Playground','Skiing','Soccer','Swimming','WeddingCeremony','WeddingDance']
catagories = [c.upper() for c in catagories]
# pkl_file = open('Console/action_mean_vector_101.pkl','rb')
# action_mean_vector = pickle.load(pkl_file)
locals()

def get_feature(featurepath, filename):
	if os.path.isfile(os.path.join(featurepath, 'train', filename)):
		with open(os.path.join(featurepath, 'train', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	elif os.path.isfile(os.path.join(featurepath, 'test', filename)):
		with open(os.path.join(featurepath, 'test', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	else:
		with open(os.path.join(featurepath, 'CCV_Classifier', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	return ret

predict_matrix = np.zeros((15,15))
if __name__ == '__main__':
	#    Training
	BGData_mean = 0.
	BGData_max = 1.
	print('%02d:%02d:%02d' % (time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec),)
	print('Loading train data...')
	picklelist = os.listdir(os.path.join(ObjDatapath, 'train'))[::1]
	random.shuffle(picklelist)
	train_data = None
	train_label = np.array([])
	for pickle_filename in picklelist:
		RGBData = get_feature(RGBDatapath, pickle_filename.split('_')[1]+'_'+pickle_filename.split('_')[2])
		RGB_LEN = RGBData.shape[0]
		for i in range(RGB_LEN,6,1):
			RGBData = np.append(RGBData,[RGBData[RGB_LEN-1]],0)
		FlowData = get_feature(FlowDatapath, pickle_filename)
		# #VGG verison BGData & ObjData
		# RGBData = np.array([RGBData[i*16:i*16+16].mean(0) for i in range(6)])
		FlowData = np.array([FlowData[i*16:i*16+16].mean(0) for i in range(6)])
		MRData = np.append(RGBData, FlowData,1)
		# load pose feature
		PoseData = None
		PoseData_TMP = get_feature(PoseDatapath,pickle_filename)
		PoseData_TMP[0] /= 320
		PoseData_TMP[1] /= 240
		for i in range(PoseData_TMP.shape[0]):
			for j in range(PoseData_TMP.shape[1]):
				if PoseData is None:
					PoseData = PoseData_TMP[i][j]
				else:
					PoseData = np.append(PoseData,PoseData_TMP[i][j])
		ObjData = get_feature(ObjDatapath,pickle_filename).mean(0)/2
		BGData = get_feature(BGDatapath,pickle_filename)
		ObjData = [ObjData]*6
		PoseData = [PoseData]*6
		p = np.append(RGBData,ObjData,1)
		p = np.append(p,BGData,1)
		p = np.append(p,PoseData,1)
		# p = RGBData
		if train_data is not None:
			train_data = np.append(train_data,p, 0)
		else:
			train_data = p
		gt = pickle_filename.split('_')[1]
		## coarse classify
		train_label = np.append(train_label,[catagories.index(gt)]*6)
	train_label = np_utils.to_categorical(train_label,15)
	print(train_data.shape)
	model = Sequential()
	model.add(Dense(128, input_shape=(train_data.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.6))
	model.add(Dense(15, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	## model fit
	model.fit(train_data, train_label, shuffle=True,batch_size=128,epochs=200, verbose=1)
	picklelist = os.listdir(os.path.join(ObjDatapath, 'test'))[::1]
	# picklelist.sort()
	predict_result = []
	groundtruth = []
	test_data = None
	test_label = np.array([])
	ground_label = np.array([])
	for pickle_filename in picklelist:
		RGBData = get_feature(RGBDatapath, pickle_filename.split('_')[1]+'_'+pickle_filename.split('_')[2])
		RGB_LEN = RGBData.shape[0]
		for i in range(RGB_LEN,6,1):
			RGBData = np.append(RGBData,[RGBData[RGB_LEN-1]],0)
		FlowData = get_feature(FlowDatapath, pickle_filename)
		#VGG verison BGData & ObjData
		# RGBData = np.array([RGBData[i*16:i*16+16].mean(0) for i in range(6)])
		FlowData = np.array([FlowData[i*16:i*16+16].mean(0) for i in range(6)])
		MRData = np.append(RGBData, FlowData,1)
		## load pose feature## load pose feature
		PoseData = None
		PoseData_TMP = get_feature(PoseDatapath,pickle_filename)
		PoseData_TMP[0] /= 320
		PoseData_TMP[1] /= 240
		for i in range(PoseData_TMP.shape[0]):
			for j in range(PoseData_TMP.shape[1]):
				if PoseData is None:
					PoseData = PoseData_TMP[i][j]
				else:
					PoseData = np.append(PoseData,PoseData_TMP[i][j])
		ObjData = get_feature(ObjDatapath,pickle_filename).mean(0)/2
		BGData = get_feature(BGDatapath,pickle_filename)
		ObjData = [ObjData]*6
		PoseData = [PoseData]*6
		test_data = np.append(RGBData,ObjData,1)
		test_data = np.append(test_data,BGData,1)
		test_data = np.append(test_data,PoseData,1)
		# test_data = np.array(RGBData)
		## Get label
		gt = pickle_filename.split('_')[1]
		test_label = catagories.index(gt)
		ground_label = np.append(ground_label,test_label)
		##scores
		scores = model.predict(test_data)
		preds = []
		score = scores.sum(axis=0)
		dist = []
		for i in range(1):
			preds.append(score.argmax())
			# dist.append(np.linalg.norm(test_data.sum(axis=0)-action_mean_vector[preds[i]].sum(axis=0)))
		# if preds[dist.index(min(dist))] == test_label:
		#     pred = test_label
		# else:
		pred = preds[0]
		predict_result.append(pred)  # mode of the pred
		groundtruth.append(test_label)
		if test_label == pred:
			pass
		# else:
		# 	print(pickle_filename, 'GT:', test_label, ' Pred:', pred)

	print('linearSVM accuracy on test data:', (np.array(predict_result) == np.array(groundtruth)).mean())
	# predict_truth = np.array(model.predict(test_data))
	for i in range(ground_label.shape[0]):
		predict_label = predict_result[i]
		predict_matrix[int(ground_label[i])][predict_label] += 1
	for i in range(predict_matrix.shape[0]):
		print(predict_matrix[i][i]/predict_matrix[i].sum())
	for i in range(predict_matrix.shape[0]):
		print(catagories[i])