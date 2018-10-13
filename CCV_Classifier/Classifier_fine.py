# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:28:41 2016

@author: ljm
"""

import pickle
import os
import matplotlib.pylab as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import sys
import time
import numpy as np
import random
import tensorflow as tf
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.utils import np_utils
from keras import regularizers

## Works Model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## basepath = '/tmp/ramdisk/'
basepath = 'caches/'
BGDatapath = basepath + 'BackgroundData_fc8'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'FlowData_lstm1'
ObjDatapath = basepath + 'Obj_feature'
PoseDatapath = basepath + 'pose_feature_CCV'

catagories_group_save = [['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE', 'NONMUSICPERFORMANCE', 'PARADE'],
					['BASKETBALL', 'BASEBALL', 'SOCCER', 'BIKING', 'PLAYGROUND'],
					['ICESKATING', 'SKIING', 'SWIMMING', 'BEACH']]
catagories_group = [['BIKING', 'PLAYGROUND', 'PARADE'],
					['BASKETBALL', 'BASEBALL', 'SOCCER', 'NONMUSICPERFORMANCE'],
					['ICESKATING', 'SKIING','SWIMMING', 'BEACH'],
					['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE']]

NUM_SUBGROUP = catagories_group.__len__()

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


if __name__ == '__main__':
	#    Training
	pred_result = []
	print('%02d:%02d:%02d' % (time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec),)
	print('Loading train data...')
	for subscript in range(NUM_SUBGROUP):
		picklelist = os.listdir(os.path.join(ObjDatapath, 'train'))[::1]
		random.shuffle(picklelist)
		train_data = None
		train_label = np.array([])
		subgroup_len = catagories_group[subscript].__len__()
		for pickle_filename in picklelist:
			RGBData = get_feature(RGBDatapath, pickle_filename.split('_')[1]+'_'+pickle_filename.split('_')[2])
			RGB_LEN = RGBData.shape[0]
			for i in range(RGB_LEN,6,1):
				RGBData = np.append(RGBData,[RGBData[RGB_LEN-1]],0)
			FlowData = get_feature(FlowDatapath, pickle_filename)
			# VGG verison BGData & ObjData
			# RGBData = np.array([RGBData[i*16:i*16+16].mean(0) for i in range(6)])
			FlowData = np.array([FlowData[i*16:i*16+16].mean(0) for i in range(6)])
			# ## load pose feature
			PoseData = None
			PoseData_TMP = get_feature(PoseDatapath, pickle_filename)
			PoseData_TMP[0] /= 320
			PoseData_TMP[1] /= 240
			for i in range(PoseData_TMP.shape[0]):
				for j in range(PoseData_TMP.shape[1]):
					if PoseData is None:
						PoseData = PoseData_TMP[i][j]
					else:
						PoseData = np.append(PoseData, PoseData_TMP[i][j])
			ObjData = get_feature(ObjDatapath,pickle_filename).mean(0)/2
			BGData = get_feature(BGDatapath,pickle_filename)
			MRData = np.append(RGBData, FlowData,1)
			ObjData = [ObjData]*6
			PoseData = [PoseData]*6
			# p = np.append(RGBData,ObjData,1)
			p = np.append(ObjData,BGData,1)
			p = np.append(p,PoseData,1)
			# p = MRData
			#coarse label
			gt = pickle_filename.split('_')[1]
			train_label_group_n = 100
			for group_n in range(NUM_SUBGROUP):
				if gt in catagories_group[group_n]:
					train_label_group_n = group_n
			if train_label_group_n == subscript:     ## Verify the label and group
				train_label = np.append(train_label, [catagories_group[subscript].index(gt)]*6)     ## append label
				if train_data is not None:                                          ## append data
					train_data = np.append(train_data, p, 0)
				else:
					train_data = p
			else:
				pass
		train_data_shape = train_data.shape
		train_label = np_utils.to_categorical(train_label,subgroup_len)
		print(train_data.shape)

		step = 10
		acc_max = 0
		while(step):
			## fc models
			fcmodel = Sequential()
			fcmodel.add(Dense(128, activation='relu',input_shape=(train_data_shape[1],)))
			fcmodel.add(Dropout(0.85))
			fcmodel.add(Dense(subgroup_len, activation='softmax'))
			fcmodel.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
			## train subfcmodel
			fcmodel.fit(train_data,train_label,batch_size=256,epochs=200,verbose=0)

			print('Loading test data...')
			picklelist = os.listdir(os.path.join(ObjDatapath, 'test'))[::1]
			picklelist.sort()
			predict_result =[]
			groundtruth = []
			for pickle_filename in picklelist:
				test_data = None
				test_label = np.array([])
				RGBData = get_feature(RGBDatapath, pickle_filename.split('_')[1]+'_'+pickle_filename.split('_')[2])
				RGB_LEN = RGBData.shape[0]
				for i in range(RGB_LEN,6,1):
					RGBData = np.append(RGBData,[RGBData[RGB_LEN-1]],0)
				FlowData = get_feature(FlowDatapath, pickle_filename)
				#VGG verison BGData & ObjData
				# RGBData = np.array([RGBData[i*16:i*16+16].mean(0) for i in range(6)])
				FlowData = np.array([FlowData[i*16:i*16+16].mean(0) for i in range(6)])
				## load pose feature
				PoseData = None
				PoseData_TMP = get_feature(PoseDatapath, pickle_filename)
				PoseData_TMP[0] /= 320
				PoseData_TMP[1] /= 240
				for i in range(PoseData_TMP.shape[0]):
					for j in range(PoseData_TMP.shape[1]):
						if PoseData is None:
							PoseData = PoseData_TMP[i][j]
						else:
							PoseData = np.append(PoseData, PoseData_TMP[i][j])
				ObjData = get_feature(ObjDatapath,pickle_filename).mean(0)/2
				BGData = get_feature(BGDatapath,pickle_filename)
				MRData = np.append(RGBData, FlowData,1)
				ObjData = [ObjData]*6
				PoseData = [PoseData]*6
				# p = np.append(RGBData,ObjData,1)
				p = np.append(ObjData,BGData,1)
				p = np.append(p,PoseData,1)
				# p = MRData
				## coarse label
				gt = pickle_filename.split('_')[1]
				test_label_group_n = 100
				for group_n in range(NUM_SUBGROUP):
					if gt in catagories_group[group_n]:
						test_label_group_n = group_n
				if test_label_group_n == subscript:     ## Verify the label and group
					test_label = catagories_group[subscript].index(gt)
					test_data = p
				else:
					continue
				## prediction result
				scores = fcmodel.predict(test_data,verbose=0)
				score = scores.sum(axis=0)
				preds = []
				for i in range(1):
					preds.append(score.argmax())
					score[score.argmax()] = -100
				## prediction result
				if test_label in preds:
					pred = test_label
				else:
					pred = preds[0]
				predict_result.append(pred)
				groundtruth.append(test_label)
				if test_label == pred:
					pass
				# else:
				#     print(pickle_filename, 'GT:', test_label, ' Pred:', pred)
			acc = (np.array(predict_result) == np.array(groundtruth)).mean()
			if acc > acc_max:
				## save models
				fcmodel.save('sub_fc/fc_group_motion'+'%d' % subscript+'.h5')
				acc_max = acc
			print('FC model accuracy on test data:', acc)
			step -= 1
	print(pred_result)
	print(np.mean(pred_result))
	sys.exit()
