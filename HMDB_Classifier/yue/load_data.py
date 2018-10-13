import pickle
import os
import numpy as np
import yue.json
def load_pkl(path):
	try:
		with open(path,'rb') as pkl_file:
			return pickle.load(pkl_file,encoding='bytes')
	except EOFError:
		print('Error load pickle file!')
		return None

def get_feature(feature_path, filename,model):
	if model == 'train':
		with open(os.path.join(feature_path, 'train', filename), 'rb') as f:
			ret = np.array(pickle.load(f, encoding='latin1'), dtype=np.float)
	elif model == 'test':
		with open(os.path.join(feature_path, 'test', filename), 'rb') as f:
			ret = np.array(pickle.load(f, encoding='latin1'), dtype=np.float)
	else:
		print('You Must type the mode as train or test!')
	return ret
class Dataset_HMDB51:
	# def get_feature_dir(dataset='UCF101'):
	# 	assert dataset in ['UCF101','HMDB51','CCV'],'No such dataset!'
	Root_Path = '../dataset/'
	Motion_Path = '../CNN_LSTM/video-classification-3d-cnn-pytorch/'
	Obj_Path = Root_Path + 'Obj_feature'
	Pose_Path = Root_Path + 'pose_feature'
	Scene_Path = Root_Path + 'Scene_feature'
	classes_file = '../dataset/list/class_names.txt'
	RGB_train_json_file = Motion_Path + 'hmdb51_resnext101_64f_train_data.json'
	RGB_test_json_file = Motion_Path + 'hmdb51_resnext101_64f_test_data.json'
	Flow_train_json_file = Motion_Path + 'hmdb51_flow_resnext101_train_data.json'
	Flow_test_json_file = Motion_Path + 'hmdb51_flow_resnext101_test_data.json'
	classes = yue.json.read_classes(classes_file)
	NUM_CLASSES = np.shape(classes)[0]
	category_group_save = [['clap', 'hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'turn', 'walk', 'wave'],
					  ['cartwheel', 'catch', 'dribble', 'flic_flac', 'handstand', 'kick_ball', 'pushup', 'shoot_ball', 'somersault'],
					  ['climb', 'climb_stairs', 'dive', 'fall_floor', 'jump', 'pick', 'pullup', 'push', 'ride_bike', 'ride_horse', 'run'],
					  ['brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'],
					  ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'shoot_gun', 'swing_baseball', 'sword_exercise', 'sword', 'throw']]
	category_group = [['clap', 'turn', 'wave', 'brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke','talk'],
						   ['climb_stairs', 'jump', 'pullup', 'ride_bike', 'ride_horse', 'run', 'kick_ball'],
						   ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'swing_baseball', 'sword_exercise', 'sword','throw', 'catch', 'dribble', 'shoot_ball'],
						   ['hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'walk', 'shoot_gun', 'pick', 'push', 'pushup'],
						   ['climb', 'dive', 'fall_floor', 'cartwheel', 'flic_flac', 'handstand', 'somersault']]
	category_group_motion = [['clap', 'turn', 'wave', 'brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke','talk'],
					  ['climb', 'dive', 'fall_floor', 'climb_stairs', 'jump', 'pullup', 'ride_bike', 'ride_horse', 'run','kick_ball'],
					  ['hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'walk', 'shoot_gun', 'pick', 'push', 'pushup', 'hit','kick', 'throw'],
					  ['cartwheel', 'flic_flac', 'handstand', 'somersault', 'catch', 'shoot_ball'],
					  ['draw_sword', 'fencing', 'golf', 'shoot_bow', 'swing_baseball', 'sword_exercise', 'sword', 'dribble']]
