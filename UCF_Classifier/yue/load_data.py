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
	Motion_Path = Root_Path + 'Motion_feature'
	Obj_Path = Root_Path + 'Obj_feature'
	Pose_path = Root_Path + 'pose_feature'
	classes_file =  Root_Path + 'class_names.txt'
	RGB_train_json_file = Motion_Path + '/train_data.json'
	RGB_test_json_file = Motion_Path + '/test_data.json'
	Flow_train_json_file = Motion_Path + '/flow_train_data.json'
	Flow_test_json_file = Motion_Path + '/flow_test_data.json'
	classes = yue.json.read_classes(classes_file)
	NUM_CLASSES = np.shape(classes)[0]
	category_group = [['clap', 'hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'turn', 'walk', 'wave'],
					  ['cartwheel', 'catch', 'dribble', 'flic_flac', 'handstand', 'kick_ball', 'pushup', 'shoot_ball', 'somersault'],
					  ['climb', 'climb_stairs', 'dive', 'fall_floor', 'jump', 'pick', 'pullup', 'push', 'ride_bike', 'ride_horse', 'run'],
					  ['brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'],
					  ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'shoot_gun', 'swing_baseball', 'sword_exercise', 'sword', 'throw']]