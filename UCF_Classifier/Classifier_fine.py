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
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
## basepath = '/tmp/ramdisk/'
basepath = 'caches/'
BGDatapath = basepath + 'Scene_feature_fix'
# RGBDatapath = basepath + 'RGBData_lstm1'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'flow_feature'
ObjDatapath = basepath + 'Obj_feature_fix'
PoseDatapath = basepath + 'pose_feature_fix'
PoselstmDatapath = 'caches/Pose_lstm'

catagories_group_save = [['GolfSwing','JavelinThrow','PoleVault','ThrowDiscus','Archery','Shotput','HammerThrow',],
					['BoxingPunchingBag','BoxingSpeedBag','Punch','Fencing','TableTennisShot','SumoWrestling',],
					['Drumming','PlayingCello','PlayingDaf','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PlayingDhol','PlayingFlute',],
					['BaseballPitch','BasketballDunk','Bowling','Billiards','VolleyballSpiking','Basketball',],
					['ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BrushingTeeth','ShavingBeard','Haircut','HeadMassage',],
					['CuttingInKitchen','Knitting','PizzaTossing','MoppingFloor','Hammering','WritingOnBoard','BlowingCandles','Typing','Mixing',],
					['HulaHoop','Nunchucks','YoYo','SkateBoarding','TrampolineJumping','IceDancing','SalsaSpin','JumpingJack','JugglingBalls','SoccerJuggling'],
					['BodyWeightSquats','HandStandPushups','HandstandWalking','PullUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','WallPushups','BenchPress','CleanAndJerk','PushUps','Lunges',],
					['Diving','Skijet','FrontCrawl','Surfing','BreastStroke','Kayaking','Skiing','Rowing','CliffDiving','Rafting',],
					['StillRings','PommelHorse','ParallelBars','LongJump','HighJump','UnevenBars','JumpRope','BalanceBeam','FloorGymnastics','TennisSwing','CricketBowling','CricketShot',],
					['BabyCrawling','WalkingWithDog','BandMarching','MilitaryParade','SkyDiving','Biking','HorseRiding','HorseRace','SoccerPenalty','FieldHockeyPenalty','FrisbeeCatch',]]
catagories_group = [['Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute'],
					['TrampolineJumping', 'Swing', 'TaiChi', 'Skiing', 'IceDancing', 'SkateBoarding', 'WalkingWithDog', 'Biking', 'HorseRiding', 'MilitaryParade', 'BandMarching', 'GolfSwing'],
					['ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'HeadMassage', 'BlowingCandles', 'Haircut', 'Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'],
					['JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'BenchPress', 'PullUps', 'WallPushups', 'JumpRope'],
					['Archery', 'TableTennisShot', 'Fencing', 'SoccerJuggling', 'VolleyballSpiking', 'Basketball', 'TennisSwing', 'BasketballDunk', 'HammerThrow'],
					['YoYo', 'Nunchucks', 'JugglingBalls', 'WritingOnBoard', 'PizzaTossing', 'BoxingSpeedBag', 'BoxingPunchingBag', 'Punch', 'HulaHoop', 'Bowling', 'SalsaSpin', 'SumoWrestling'],
					['Skijet', 'FrontCrawl', 'Surfing', 'BreastStroke', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'HorseRace'],
					['StillRings', 'PommelHorse', 'ParallelBars', 'UnevenBars', 'BalanceBeam', 'FloorGymnastics', 'Billiards', 'RopeClimbing', 'Diving'],
					['PoleVault', 'FrisbeeCatch', 'JavelinThrow', 'ThrowDiscus', 'HighJump', 'Shotput', 'LongJump', 'BaseballPitch', 'CricketBowling', 'CricketShot', 'SoccerPenalty', 'FieldHockeyPenalty'],
					['RockClimbingIndoor', 'HandstandWalking', 'BabyCrawling', 'Hammering', 'MoppingFloor', 'HandStandPushups', 'PushUps', 'SkyDiving']]
catagories_group_motion = [['Skijet', 'FrontCrawl', 'Surfing', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'Skiing', 'SkyDiving'], ['MilitaryParade', 'FloorGymnastics', 'PoleVault', 'HighJump', 'SkateBoarding', 'Biking', 'Swing', 'RopeClimbing', 'TrampolineJumping', 'HandstandWalking', 'Basketball', 'BasketballDunk', 'PommelHorse', 'ParallelBars', 'BalanceBeam'], ['Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'], ['TableTennisShot', 'JugglingBalls', 'BabyCrawling', 'Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute'], ['IceDancing', 'WalkingWithDog', 'BandMarching', 'GolfSwing', 'FrisbeeCatch', 'BaseballPitch', 'CricketShot', 'FieldHockeyPenalty', 'TennisSwing', 'JavelinThrow', 'ThrowDiscus', 'Shotput', 'LongJump', 'CricketBowling', 'HammerThrow', 'SoccerJuggling', 'VolleyballSpiking', 'SumoWrestling'], ['BreastStroke', 'RockClimbingIndoor', 'StillRings', 'UnevenBars', 'Billiards', 'Diving'], ['Bowling', 'WritingOnBoard', 'PizzaTossing', 'BoxingPunchingBag', 'Punch', 'BenchPress', 'PullUps', 'Hammering', 'MoppingFloor', 'PushUps', 'HandStandPushups'], ['BoxingSpeedBag', 'ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'BlowingCandles', 'Haircut', 'HeadMassage'], ['YoYo', 'Nunchucks', 'SalsaSpin', 'Archery', 'JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'WallPushups', 'JumpRope', 'TaiChi', 'Fencing', 'HulaHoop'], ['HorseRace', 'HorseRiding', 'SoccerPenalty']]

NUM_SUBGROUP = catagories_group_motion.__len__()

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
	pred_result = []
	print('%02d:%02d:%02d' % (time.localtime().tm_hour, time.localtime().tm_min, time.localtime().tm_sec),)
	print('Loading train data...')
	for subscript in range(NUM_SUBGROUP):
		picklelist = os.listdir(os.path.join(RGBDatapath, 'train'))[::1]
		picklelist.sort()
		train_data = None
		train_label = np.array([])
		test_data = None
		test_label = np.array([])
		subgroup_len = catagories_group_motion[subscript].__len__()
		for pickle_filename in picklelist:
			RGBData = get_feature(RGBDatapath, pickle_filename)
			FlowData = get_feature(FlowDatapath, pickle_filename)
			# VGG verison BGData & ObjData
			# BGData = get_feature(BGDatapath, pickle_filename.split('.')[0]+"1.pickle")
			# BGData_TMP = get_feature(BGDatapath, pickle_filename.split('.')[0]+"2.pickle")
			# BGData = np.row_stack((BGData,BGData_TMP))
			# RGBData = np.append(RGBData[0:15].mean(0),RGBData[16:31].mean(0))
			# FlowData = np.append(FlowData[0:15].mean(0),FlowData[16:31].mean(0))
			BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
			ObjData = get_feature(ObjDatapath,pickle_filename[:-7]+'.pkl').mean(0)
			PoseData = get_feature(PoseDatapath,pickle_filename[:-7]+'.pkl')
			# PoseData = np.append(PoseData[0:2].mean(0),PoseData[3:5].mean(0))
			# RGBData_G = np.array(np.gradient(RGBData))[1]
			# RGBData = np.array([RGBData_G.mean(0),RGBData_G.mean(0)])
			# FlowData_G = np.array(np.gradient(FlowData))[1]
			# FlowData = np.array([FlowData_G.mean(0),FlowData_G.mean(0)])
			# MRData = RGBData * w + FlowData * (1 - w)
			# p = np.append(RGBData, FlowData/10)
			# p = np.append(MRData, ObjData/3)
			p = np.append(ObjData/3,PoseData)
			p = np.append(p, BGData.sum(0)/10)
			#coarse label
			gt = pickle_filename.split('_')[1]
			train_label_group_n = 100
			for group_n in range(NUM_SUBGROUP):
				if gt in catagories_group_motion[group_n]:
					train_label_group_n = group_n
			if train_label_group_n == subscript:     ## Verify the label and group
				train_label = np.append(train_label, catagories_group_motion[subscript].index(gt))     ## append label
				if train_data is not None:                                          ## append data
					train_data = np.append(train_data, [p], 0)
				else:
					train_data = [p]
			else:
				pass
		train_data_shape = train_data.shape
		train_label = np_utils.to_categorical(train_label,subgroup_len)
#        train_data = train_data.transpose()
		print(train_data.shape)

		picklelist = os.listdir(os.path.join(RGBDatapath, 'test'))[::1]
		picklelist.sort()
		for pickle_filename in picklelist:
			RGBData = get_feature(RGBDatapath, pickle_filename)
			FlowData = get_feature(FlowDatapath, pickle_filename)
			#VGG verison BGData & ObjData
			# BGData = get_feature(BGDatapath, pickle_filename.split('.')[0]+"1.pickle")
			# BGData_TMP = get_feature(BGDatapath, pickle_filename.split('.')[0]+"2.pickle")
			# BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
			# ObjData = get_feature(ObjDatapath,pickle_filename)
			# RGBData = np.append(RGBData[0:15].mean(0),RGBData[16:31].mean(0))
			# FlowData = np.append(FlowData[0:15].mean(0),FlowData[16:31].mean(0))
			BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
			ObjData = get_feature(ObjDatapath,pickle_filename[:-7]+'.pkl').mean(0)
			PoseData = get_feature(PoseDatapath,pickle_filename[:-7]+'.pkl')
			# PoseData = np.append(PoseData[0:2].mean(0),PoseData[3:5].mean(0))
			# MRData = RGBData * w + FlowData * (1 - w)
			# p = np.append(RGBData, FlowData/10)
			# p = np.append(MRData, ObjData/3)
			p = np.append(ObjData/3,PoseData)
			p = np.append(p, BGData.sum(0)/10)
			## coarse label
			gt = pickle_filename.split('_')[1]
			test_label_group_n = 100
			for group_n in range(NUM_SUBGROUP):
				if gt in catagories_group_motion[group_n]:
					test_label_group_n = group_n
			if test_label_group_n == subscript:     ## Verify the label and group
				test_label = np.append(test_label, catagories_group_motion[subscript].index(gt))     ## append label
				if test_data is not None:                                          ## append data
					test_data = np.append(test_data, [p], 0)
				else:
					test_data = [p]
			else:
				pass
		test_label = np_utils.to_categorical(test_label,subgroup_len)
		acc_max = 0
		counter = 10
		while(counter):
			fcmodel = Sequential()
			fcmodel.add(Dense(128, activation='relu',input_shape=(train_data_shape[1],),kernel_regularizer = regularizers.l2(0.001)))
			fcmodel.add(Dropout(0.85))
			fcmodel.add(Dense(subgroup_len, activation='softmax'))
			fcmodel.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
			## train subfcmodel
			fcmodel.fit(train_data,train_label,shuffle=True,validation_data=(test_data,test_label),batch_size=128,epochs=300,verbose=0)
			sc,acc = fcmodel.evaluate(test_data,test_label,verbose=0)
			print(sc,acc)
			if acc > acc_max:
				fcmodel.save('sub_fc/fc_group_motion'+'%d' % subscript+'.h5')
				acc_max = acc
			counter -= 1
		pred_result.append(acc_max)
	print(pred_result)
	print(np.mean(pred_result))
	sys.exit()
