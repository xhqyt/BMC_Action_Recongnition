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
config.gpu_options.allow_growth = True
## data dir
basepath = 'caches/'
BGDatapath = basepath + 'BackgroundData_fc8'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'FlowData_lstm1'
ObjDatapath = basepath + 'Obj_feature'
PoseDatapath = basepath + 'pose_feature_CCV'
## group divide
catagories_group_save = [['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE', 'NONMUSICPERFORMANCE', 'PARADE'],
					['BASKETBALL', 'BASEBALL', 'SOCCER', 'BIKING', 'PLAYGROUND'],
					['ICESKATING', 'SKIING', 'SWIMMING', 'BEACH']]
catagories_group = [['BIKING', 'PLAYGROUND', 'PARADE'],
					['BASKETBALL', 'BASEBALL', 'SOCCER', 'NONMUSICPERFORMANCE'],
					['ICESKATING', 'SKIING','SWIMMING', 'BEACH'],
					['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE']]
catagories = ['Basketball','Baseball','Beach','Biking','Birthday','Graduation','IceSkating','NonMusicPerformance','Parade','Playground','Skiing','Soccer','Swimming','WeddingCeremony','WeddingDance']
catagories = [c.upper() for c in catagories]
NUM_SUBGROUP = catagories_group.__len__()
submodel = []
for i in range(NUM_SUBGROUP):
	submodel.append(load_model('sub_fc/fc_group_next'+'%d' % i+'.h5'))
coarse_model = load_model('sub_fc/fc_coarse_next.h5')
def opendata():
	try:
		with open('Console/action_mean_vector_next.pkl','rb') as pkl_file:
			return pickle.load(pkl_file)
	except EOFError:
		return None
action_mean_vector = opendata()

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
	BGData_mean = 0.
	BGData_max = 1.
	picklelist = os.listdir(os.path.join(ObjDatapath, 'test'))[::1]
	picklelist.sort()
	for k1 in [2]:
		for k2 in [1]:
			predict_result = []
			groundtruth = []
			ground_truth = np.array([])
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
				p = np.append(RGBData,ObjData,1)
				p = np.append(p,BGData,1)
				p = np.append(p,PoseData,1)
				# p = MRData
				test_data = p
				# test_data = np.array([RGBData])
				gt = pickle_filename.split('_')[1]
				## coarse classify
				test_label_group_n = 0
				for group_n in range(NUM_SUBGROUP):
					if gt in catagories_group[group_n]:
						test_label_group_n = group_n*100+catagories_group[group_n].index(gt)
				test_label = test_label_group_n
				ground_truth = np.append(ground_truth,test_label)
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
						dist.append(np.abs(test_data.mean(axis=0)-action_mean_vector[pred_class_score[i*(k2+1)+ii]].mean(axis=0)).sum())
					iii = len(catagories_group[preds[i]])
				if pred_class_score[dist.index(min(dist))] == test_label:
					pred = test_label
				else:
					pred = pred_class_score[0]
				predict_result.append(pred)  # model of the pred
				groundtruth.append(test_label)
				if test_label == pred:
					pass
				# else:
				#     print(pickle_filename, 'GT:', test_label, ' Pred:', pred)
			print('K1: ',k1+1,' K2:',k2+1,'Test accuracy on test data:', (np.array(predict_result) == np.array(groundtruth)).mean())

			for i in range(ground_truth.shape[0]):
				ground_label = catagories_group[int(ground_truth[i]/100)][ground_truth[i]%100]
				pred_label = catagories_group[int(predict_result[i]/100)][predict_result[i]%100]
				predict_matrix[catagories.index(ground_label)][catagories.index(pred_label)] += 1
			for i in range(predict_matrix.shape[0]):
				print(predict_matrix[i][i] / predict_matrix[i].sum())
			for i in range(predict_matrix.shape[0]):
				print(catagories_group[i])