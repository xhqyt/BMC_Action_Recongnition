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
BGDatapath = basepath + 'Scene_feature_fix'
# RGBDatapath = basepath + 'RGBData_lstm1'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'flow_feature'
ObjDatapath = basepath + 'Obj_feature_fix'
PoseDatapath = basepath + 'pose_feature_fix'
# PoseDatapath = 'caches/Pose_lstm'
catagories_group = [['GolfSwing','JavelinThrow','PoleVault','ThrowDiscus','Archery','Shotput','HammerThrow',],
					['BoxingPunchingBag','BoxingSpeedBag','Punch','Fencing','TableTennisShot','SumoWrestling',],
					['Drumming','PlayingCello','PlayingDaf','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PlayingDhol','PlayingFlute',],
					['BaseballPitch','BasketballDunk','Bowling','Billiards','VolleyballSpiking','Basketball','CricketShot',],
					['ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BrushingTeeth','ShavingBeard','Haircut','HeadMassage',],
					['CuttingInKitchen','Knitting','PizzaTossing','MoppingFloor','Hammering','WritingOnBoard','BlowingCandles','Typing','Mixing',],
					['HulaHoop','Nunchucks','YoYo','SkateBoarding','TrampolineJumping','IceDancing','SalsaSpin','JumpingJack','JugglingBalls','SoccerJuggling'],
					['BodyWeightSquats','HandStandPushups','HandstandWalking','PullUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','WallPushups','BenchPress','CleanAndJerk','PushUps','Lunges',],
					['Diving','Skijet','FrontCrawl','Surfing','BreastStroke','Kayaking','Skiing','Rowing','CliffDiving','Rafting',],
					['StillRings','PommelHorse','ParallelBars','LongJump','HighJump','UnevenBars','JumpRope','BalanceBeam','FloorGymnastics','TennisSwing','CricketBowling',],
					['BabyCrawling','WalkingWithDog','BandMarching','MilitaryParade','SkyDiving','Biking','HorseRiding','HorseRace','SoccerPenalty','FieldHockeyPenalty','FrisbeeCatch',]]
catagories = ['GolfSwing','JavelinThrow','PoleVault','ThrowDiscus','Archery','Shotput','HammerThrow','BoxingPunchingBag','BoxingSpeedBag','Punch','Fencing','TableTennisShot','SumoWrestling','Drumming','PlayingCello','PlayingDaf','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PlayingDhol','PlayingFlute','BaseballPitch','BasketballDunk','Bowling','Billiards','VolleyballSpiking','Basketball','CricketShot','ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BrushingTeeth','ShavingBeard','Haircut','HeadMassage','CuttingInKitchen','Knitting','PizzaTossing','MoppingFloor','Hammering','WritingOnBoard','BlowingCandles','Typing','Mixing','HulaHoop','Nunchucks','YoYo','SkateBoarding','TrampolineJumping','IceDancing','SalsaSpin','JumpingJack','JugglingBalls','SoccerJuggling','BodyWeightSquats','HandStandPushups','HandstandWalking','PullUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','WallPushups','BenchPress','CleanAndJerk','PushUps','Lunges','Diving','Skijet','FrontCrawl','Surfing','BreastStroke','Kayaking','Skiing','Rowing','CliffDiving','Rafting','StillRings','PommelHorse','ParallelBars','LongJump','HighJump','UnevenBars','JumpRope','BalanceBeam','FloorGymnastics','TennisSwing','CricketBowling','BabyCrawling','WalkingWithDog','BandMarching','MilitaryParade','SkyDiving','Biking','HorseRiding','HorseRace','SoccerPenalty','FieldHockeyPenalty','FrisbeeCatch',]
w = 0.85
NUM_SUBGROUP = 11
NUM_CLASSES = catagories.__len__()
# pkl_file = open('Console/action_mean_vector_101.pkl','rb')
# action_mean_vector = pickle.load(pkl_file)

def get_feature(featurepath, filename):
	if os.path.isfile(os.path.join(featurepath, 'train', filename)):
		with open(os.path.join(featurepath, 'train', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	else:
		with open(os.path.join(featurepath, 'test', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	return ret


if __name__ == '__main__':
	#    Training
	BGData_mean = 0.
	BGData_max = 1.
	print('%02d:%02d:%02d' % (time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec),)
	print('Loading train data...')
	picklelist = os.listdir(os.path.join(RGBDatapath, 'train'))[::1]
	# picklelist.sort()
	train_data = None
	train_label = np.array([])
	predict_matrix = np.zeros((101,101))
	for pickle_filename in picklelist:
		RGBData = get_feature(RGBDatapath, pickle_filename)
		FlowData = get_feature(FlowDatapath, pickle_filename)
		# #VGG verison BGData & ObjData
		# BGData = get_feature(BGDatapath, pickle_filename.split('.')[0]+"1.pickle")
		# BGData_TMP = get_feature(BGDatapath, pickle_filename.split('.')[0]+"2.pickle")
		# BGData = np.row_stack((BGData,BGData_TMP))
		# ObjData = get_feature(ObjDatapath, pickle_filename.split('.')[0]+"1.pickle")
		# ObjData_TMP = get_feature(ObjDatapath, pickle_filename.split('.')[0]+"2.pickle")
		# ObjData = np.row_stack((ObjData,ObjData_TMP))
		# RGBData = np.append(RGBData[0:15].mean(0),RGBData[16:31].mean(0))
		# FlowData = np.array([FlowData[i * 16:i * 16 + 16].mean(0) for i in range(2)])
		# #        MRData = RGBData * w + FlowData * (1 - w)
		MRData = np.append(RGBData, FlowData/10)
		BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
		PoseData = get_feature(PoseDatapath,pickle_filename[:-7]+'.pkl')
		ObjData = get_feature(ObjDatapath,pickle_filename[:-7]+'.pkl').mean(0)
		# p = [np.append(MRData, BGData.sum(0)/10)]
		# p = [np.append(BGData.sum(0)/10, ObjData/3)]
		# p = [np.append(p,PoseData)]
		p = [MRData]
		if train_data is not None:
			train_data = np.append(train_data,p, 0)
		else:
			train_data = p
		gt = pickle_filename.split('_')[1]
		train_label = np.append(train_label,catagories.index(gt))
	train_label = np_utils.to_categorical(train_label,NUM_CLASSES)
	print(train_data.shape)
	model = Sequential()
	model.add(Dense(128, input_shape=(train_data.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.6))
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	## model fit
	model.fit(train_data, train_label, shuffle=True,batch_size=128,epochs=100, verbose=1)

	picklelist = os.listdir(os.path.join(RGBDatapath, 'test'))[::1]
	picklelist.sort()
	predict_result = []
	groundtruth = []
	test_data = None
	test_label = np.array([])
	for pickle_filename in picklelist:
		RGBData = get_feature(RGBDatapath, pickle_filename)
		FlowData = get_feature(FlowDatapath, pickle_filename)
		#VGG verison BGData & ObjData
		# BGData = get_feature(BGDatapath, pickle_filename.split('.')[0]+"1.pickle")
		# BGData_TMP = get_feature(BGDatapath, pickle_filename.split('.')[0]+"2.pickle")
		# BGData = np.row_stack((BGData,BGData_TMP))
		# ObjData = get_feature(ObjDatapath, pickle_filename.split('.')[0]+"1.pickle")
		# ObjData_TMP = get_feature(ObjDatapath, pickle_filename.split('.')[0]+"2.pickle")
		# ObjData = np.row_stack((ObjData,ObjData_TMP))
		# RGBData = np.append(RGBData[0:15].mean(0),RGBData[16:31].mean(0))
		# FlowData = np.array([FlowData[i * 16:i * 16 + 16].mean(0) for i in range(2)])
		# #        MRData = RGBData * w + FlowData * (1 - w)
		MRData = np.append(RGBData, FlowData/10)
		# test_data = np.append(BGData/10, MRData, 1)
		# test_data = np.append(test_data, ObjData/10, 1)
		BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
		PoseData = get_feature(PoseDatapath,pickle_filename[:-7]+'.pkl')
		ObjData = get_feature(ObjDatapath,pickle_filename[:-7]+'.pkl').mean(0)
		# test_data = np.array([np.append(MRData, BGData.sum(0)/10)])
		# test_data = np.array([np.append(BGData.sum(0)/10, ObjData/3)])
		# test_data = np.array([np.append(test_data,PoseData)])
		test_data = np.array([MRData])
		gt = pickle_filename.split('_')[1]
		## coarse classify
		test_label = catagories.index(gt)
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

	# for i in range(predict_result.__len__()):
	# 	predict_matrix[groundtruth[i]][predict_result[i]] += 1
	# for i in range(predict_matrix.shape[0]):
	# 	print(predict_matrix[i][i]/predict_matrix[i].sum())
	# for i in range(predict_matrix.shape[0]):
	# 	print(catagories[i])
	print('Model accuracy on test data:', (np.array(predict_result) == np.array(groundtruth)).mean())
	sys.exit()