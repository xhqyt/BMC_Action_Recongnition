import numpy as np
import pickle
import os
from keras.utils import np_utils
from yue.json import parse_json
from yue.json import read_classes
from yue.load_data import get_feature
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
Root_Path = '../dataset/'
Motion_Path = '../CNN_LSTM/video-classification-3d-cnn-pytorch/'
Obj_Path = Root_Path + 'Obj_feature'
Pose_Path = Root_Path + 'pose_feature'
Scene_Path = Root_Path + 'Scene_feature'
classes_file =  '../dataset/list/class_names.txt'
RGB_train_json_file = Motion_Path + 'hmdb51_resnext101_64f_train_data.json'
RGB_test_json_file = Motion_Path + 'hmdb51_resnext101_64f_test_data.json'
Flow_train_json_file = Motion_Path + 'hmdb51_flow_resnext101_train_data.json'
Flow_test_json_file = Motion_Path + 'hmdb51_flow_resnext101_test_data.json'
classes = read_classes(classes_file)
CLASS_NUM = np.shape(classes)[0]
category_group_save = [['clap', 'hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'turn', 'walk', 'wave'],
				  ['cartwheel', 'catch', 'dribble', 'flic_flac', 'handstand', 'kick_ball', 'pushup', 'shoot_ball', 'somersault'],
				  ['climb', 'climb_stairs', 'dive', 'fall_floor', 'jump', 'pick', 'pullup', 'push', 'ride_bike', 'ride_horse', 'run'],
				  ['brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'],
				  ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'shoot_gun', 'swing_baseball', 'sword_exercise', 'sword', 'throw']]
category_group = [['clap', 'turn', 'wave', 'brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'],
				  ['climb_stairs', 'jump', 'pullup', 'ride_bike', 'ride_horse', 'run', 'kick_ball'],
				  ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'swing_baseball', 'sword_exercise', 'sword', 'throw', 'catch', 'dribble', 'shoot_ball'],
				  ['hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'walk', 'shoot_gun', 'pick', 'push', 'pushup'],
				  ['climb', 'dive', 'fall_floor', 'cartwheel', 'flic_flac', 'handstand', 'somersault']]
category_group_motion = [['clap', 'turn', 'wave', 'brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'], ['climb', 'dive', 'fall_floor', 'climb_stairs', 'jump', 'pullup', 'ride_bike', 'ride_horse', 'run', 'kick_ball'], ['hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'walk', 'shoot_gun', 'pick', 'push', 'pushup', 'hit', 'kick', 'throw'], ['cartwheel', 'flic_flac', 'handstand', 'somersault', 'catch', 'shoot_ball'], ['draw_sword', 'fencing', 'golf', 'shoot_bow', 'swing_baseball', 'sword_exercise', 'sword', 'dribble']]

NUM_SUBGROUP = category_group.__len__()
if __name__ == '__main__':
	data = {}
	data_label = {}
	## load train data
	print('Begin loading data...')
	file_list = os.listdir(Obj_Path + '/train')
	file_list.sort()
	train_data = None
	train_label = np.array([])
	for file in file_list:
		ObjData = get_feature(Obj_Path,file,'train').max(axis=0)/5
		PoseData = get_feature(Pose_Path,file,'train')
		SceneData = get_feature(Scene_Path,file, 'train').max(0)
		p = np.append(ObjData,SceneData/5)
		p = np.append(p,PoseData)
		if train_data is None:
			train_data = [p]
		else:
			train_data = np.append(train_data,[p],0)
		gt = file.split('.')[0][:-4]
		train_label_group_n = -1
		for group_n in range(NUM_SUBGROUP):
			if gt in category_group[group_n]:
				train_label_group_n = group_n*100+category_group[group_n].index(gt)
		train_label = np.append(train_label,train_label_group_n)
	## load motion feature & fusion features
	# RGBData_train,_ = parse_json(RGB_train_json_file,classes_file,data_type='max')
	# FlowData_train,_ = parse_json(Flow_train_json_file,classes_file,data_type='max')
	# train_data = np.append(FlowData_train/10,train_data,1)
	# train_data = np.append(RGBData_train,train_data,1)
	# train_data = np.append(RGBData_train,FlowData_train,1)
	print(train_data.shape)
	for i in range(train_data.shape[0]):
		if train_label[i] in data:
			data[train_label[i]] = data[train_label[i]] + train_data[i]
			data_label[train_label[i]] += 1
		else:
			data[train_label[i]] = train_data[i]
			data_label[train_label[i]] = 1
	for key in data.keys():
		data[key] /= data_label[key]
	save_file = open(Root_Path + '/vector_map/action_mean_vector_motion.pkl', 'wb')
	pickle.dump(data, save_file)
	save_file.close()