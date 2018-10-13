# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:28:41 2016

@author: ljm
"""

import pickle
import os
import sys
import time
import numpy as np
import random
import tensorflow as tf
from sklearn import svm
from keras.models import load_model

## Works Model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
## data dir
basepath = 'caches/'
BGDatapath = basepath + 'Scene_feature_fix'
# RGBDatapath = basepath + 'RGBData_lstm1'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'flow_feature'
ObjDatapath = basepath + 'Obj_feature_fix'
PoseDatapath = basepath + 'pose_feature_fix'
subfc_path = 'sub_fc'
## group divide
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
catagories = ['TaiChi', 'YoYo', 'HulaHoop', 'SalsaSpin', 'JumpRope', 'Nunchucks', 'JumpingJack', 'JugglingBalls', 'PullUps', 'WallPushups', 'WritingOnBoard','TrampolineJumping', 'SoccerJuggling', 'Swing', 'JavelinThrow', 'PoleVault', 'ThrowDiscus', 'VolleyballSpiking', 'Basketball', 'HighJump', 'TennisSwing','CleanAndJerk', 'Lunges', 'BoxingPunchingBag', 'BodyWeightSquats', 'HandStandPushups', 'PushUps', 'SumoWrestling', 'Punch', 'BenchPress','ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'HeadMassage', 'BlowingCandles', 'Haircut', 'Knitting', 'CuttingInKitchen', 'Mixing', 'Typing','Rafting', 'SkyDiving', 'Kayaking', 'Skiing', 'CliffDiving', 'IceDancing', 'SkateBoarding', 'WalkingWithDog', 'Biking', 'HorseRiding', 'HorseRace', 'MilitaryParade','HammerThrow', 'BasketballDunk', 'Shotput', 'LongJump', 'BandMarching', 'GolfSwing', 'BaseballPitch', 'CricketBowling', 'CricketShot', 'SoccerPenalty', 'FieldHockeyPenalty', 'FrisbeeCatch','TableTennisShot', 'Fencing', 'StillRings', 'PommelHorse', 'ParallelBars', 'UnevenBars', 'BalanceBeam', 'FloorGymnastics','Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute','RopeClimbing', 'Archery', 'PizzaTossing', 'RockClimbingIndoor', 'BoxingSpeedBag', 'HandstandWalking', 'BabyCrawling', 'Hammering', 'Bowling', 'MoppingFloor','Diving', 'Skijet', 'FrontCrawl', 'Surfing', 'BreastStroke', 'Rowing', 'Billiards']
catagories_group_motion = [['Skijet', 'FrontCrawl', 'Surfing', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'Skiing', 'SkyDiving'], ['MilitaryParade', 'FloorGymnastics', 'PoleVault', 'HighJump', 'SkateBoarding', 'Biking', 'Swing', 'RopeClimbing', 'TrampolineJumping', 'HandstandWalking', 'Basketball', 'BasketballDunk', 'PommelHorse', 'ParallelBars', 'BalanceBeam'], ['Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'], ['TableTennisShot', 'JugglingBalls', 'BabyCrawling', 'Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute'], ['IceDancing', 'WalkingWithDog', 'BandMarching', 'GolfSwing', 'FrisbeeCatch', 'BaseballPitch', 'CricketShot', 'FieldHockeyPenalty', 'TennisSwing', 'JavelinThrow', 'ThrowDiscus', 'Shotput', 'LongJump', 'CricketBowling', 'HammerThrow', 'SoccerJuggling', 'VolleyballSpiking', 'SumoWrestling'], ['BreastStroke', 'RockClimbingIndoor', 'StillRings', 'UnevenBars', 'Billiards', 'Diving'], ['Bowling', 'WritingOnBoard', 'PizzaTossing', 'BoxingPunchingBag', 'Punch', 'BenchPress', 'PullUps', 'Hammering', 'MoppingFloor', 'PushUps', 'HandStandPushups'], ['BoxingSpeedBag', 'ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'BlowingCandles', 'Haircut', 'HeadMassage'], ['YoYo', 'Nunchucks', 'SalsaSpin', 'Archery', 'JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'WallPushups', 'JumpRope', 'TaiChi', 'Fencing', 'HulaHoop'], ['HorseRace', 'HorseRiding', 'SoccerPenalty']]

NUM_SUBGROUP = catagories_group.__len__()
submodel = []
for i in range(NUM_SUBGROUP):
	submodel.append(load_model('sub_fc/fc_group_motion'+'%d' % i+'.h5'))
coarse_model = load_model('sub_fc/fc_coarse_motion.h5')
def opendata():
	try:
		with open('Console/action_mean_vector_con.pkl','rb') as pkl_file:
			return pickle.load(pkl_file)
	except EOFError:
		return None
action_mean_vector = opendata()
def get_feature(featurepath, filename):
	if os.path.isfile(os.path.join(featurepath, 'train', filename)):
		with open(os.path.join(featurepath, 'train', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	else:
		with open(os.path.join(featurepath, 'test', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	return ret


if __name__ == '__main__':
	BGData_mean = 0.
	BGData_max = 1.
	picklelist = os.listdir(os.path.join(RGBDatapath, 'test'))[::1]
	picklelist.sort()
	for k1 in [2,3,4]:
		for k2 in [1,2,3]:
			predict_matrix = np.zeros((101,101))
			predict_result = []
			groundtruth = []
			for pickle_filename in picklelist:
				RGBData = get_feature(RGBDatapath, pickle_filename)
				FlowData = get_feature(FlowDatapath, pickle_filename)
				# ## VGG verison BGData & ObjData
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
				# test_data = np.array([np.append(RGBData, FlowData/10)])
				# test_data = np.append(RGBData, ObjData/3)
				test_data = np.append(ObjData/3,PoseData)
				test_data = np.array([np.append(test_data,BGData.sum(0)/10)])
				gt = pickle_filename.split('_')[1]
				## coarse classify
				test_label_group_n = 0
				for group_n in range(NUM_SUBGROUP):
					if gt in catagories_group_motion[group_n]:
						test_label_group_n = group_n*100+catagories_group_motion[group_n].index(gt)
				test_label = test_label_group_n
				## scores
				scores = coarse_model.predict(test_data)
				score = scores.sum(axis=0)
				preds = []
				for i in range(k1+1):
					preds.append(score.argmax())
					score[score.argmax()] = -100
				## load sub fcmodels
				pred_classes = []
				dist = []
				pred_class_score = []
				iii = 0
				for i in range(k1+1):
					pred_classes.append(submodel[preds[i]].predict(test_data))
					pred_class = pred_classes[i].sum(axis=0)
					for ii in range(k2+1):
						pred_class_score.append(preds[i]*100 + pred_class.argmax())
						pred_class[pred_class.argmax()] = -100
						dist.append(np.abs(test_data.sum(axis=0)-action_mean_vector[pred_class_score[i*(k2+1)+ii]].sum(axis=0)).sum())
					iii = len(catagories_group_motion[preds[i]])
				if pred_class_score[dist.index(min(dist))] == test_label:
					pred = test_label
				else:
					pred = pred_class_score[0]
				predict_result.append(pred)  # model of the pred
				groundtruth.append(test_label)
				if test_label == pred:
					pass
				# else:
				# 	print(pickle_filename, 'GT:', test_label, ' Pred:', pred)
			# for i in range(predict_result.__len__()):
			# 	label_ground = catagories.index(catagories_group[int(groundtruth[i]/100)][groundtruth[i]%100])
			# 	label_predict = catagories.index(catagories_group[int(predict_result[i]/100)][predict_result[i]%100])
			# 	predict_matrix[label_ground][label_predict] += 1
			# for i in range(predict_matrix.shape[0]):
			# 	print(predict_matrix[i][i]/predict_matrix[i].sum())
			print('K1: ',k1+1,' K2:',k2+1,'Test accuracy on test data:', (np.array(predict_result) == np.array(groundtruth)).mean())