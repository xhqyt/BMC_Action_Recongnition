##建立101个类的均值向量作为特征
import pickle
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os

## Works Model
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
## data dir
basepath = '../caches/'
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

NUM_SUBGROUP = catagories_group.__len__()
test_data = None
test_label = np.array([])
picklelist = os.listdir(os.path.join(ObjDatapath, 'train'))[::1]
picklelist.sort()
data = {}
data_label = {}
for pickle_filename in picklelist:
	RGBData = get_feature(RGBDatapath, pickle_filename.split('_')[1]+'_'+pickle_filename.split('_')[2])
	RGB_LEN = RGBData.shape[0]
	for i in range(RGB_LEN,6,1):
		RGBData = np.append(RGBData,[RGBData[RGB_LEN-1]],0)
	FlowData = get_feature(FlowDatapath, pickle_filename)
	# VGG verison BGData & ObjData
	# RGBData = np.array([RGBData[i * 16:i * 16 + 16].mean(0) for i in range(6)])
	FlowData = np.array([FlowData[i * 16:i * 16 + 16].mean(0) for i in range(6)])
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
	ObjData = get_feature(ObjDatapath, pickle_filename).mean(0)/2
	BGData = get_feature(BGDatapath, pickle_filename)
	MRData = np.append(RGBData, FlowData, 1)
	ObjData = [ObjData] * 6
	PoseData = [PoseData] * 6
	# coarse label
	# test_data = np.append(RGBData,ObjData,1)
	# test_data = np.append(ObjData,BGData,1)
	# test_data = np.append(test_data,PoseData,1)
	test_data = np.array([MRData.mean(0)])
	# test_data = np.array([FlowData])
	gt = pickle_filename.split('_')[1]
	## coarse classify
	test_label_group_n = 0
	for group_n in range(NUM_SUBGROUP):
		if gt in catagories_group[group_n]:
			test_label_group_n = group_n*100 + catagories_group[group_n].index(gt)
	test_label = test_label_group_n
	if test_label in data:
		data[test_label] = data[test_label] + test_data
		data_label[test_label] += 1
	else:
		data[test_label] = test_data
		data_label[test_label] = 1
for key in data.keys():
	data[key] /= data_label[key]
save_file = open('action_mean_vector_motion.pkl','wb')
pickle.dump(data,save_file)
save_file.close()