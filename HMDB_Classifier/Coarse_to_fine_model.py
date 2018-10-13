import numpy as np
import time
import os
from yue.json import parse_json
from yue.json import read_classes
from yue.load_data import get_feature
from yue.load_data import load_pkl
from keras.models import load_model
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
NUM_CLASSES = np.shape(classes)[0]
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
category = ['clap', 'hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'turn', 'walk', 'wave','cartwheel', 'catch', 'dribble', 'flic_flac', 'handstand', 'kick_ball', 'pushup', 'shoot_ball', 'somersault','climb', 'climb_stairs', 'dive', 'fall_floor', 'jump', 'pick', 'pullup', 'push', 'ride_bike', 'ride_horse', 'run','brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk','draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'shoot_gun', 'swing_baseball', 'sword_exercise', 'sword', 'throw']
NUM_SUBGROUP = category_group.__len__()
fine_model = []
for i in range(NUM_SUBGROUP):
	fine_model.append(load_model('../dataset/fine_model/fine_model_next_64f_'+'%d' % i+'.h5'))
coarse_model = load_model('../dataset/coarse_model/coarse_model_next_64f.h5')
action_mean_vector = load_pkl(Root_Path + '/vector_map/action_mean_vector_next_64f.pkl')
if __name__ == '__main__':
	RGBData_test,_ =parse_json(RGB_test_json_file,classes_file,data_type='max')
	FlowData_test,_ =parse_json(Flow_test_json_file,classes_file,data_type='max')
	## load test data
	file_list = os.listdir(Obj_Path + '/test')
	file_list.sort()
	test_data = None
	test_label = np.array([])
	pred_matrix = np.zeros((51,51))
	for k1 in [0]:
		for k2 in [1]:
			predict_result = []
			ground_truth = []
			counter_test = 0
			for file in file_list:
				ObjData = get_feature(Obj_Path,file,'test').max(axis=0)/5
				PoseData = get_feature(Pose_Path,file,'test')
				SceneData = get_feature(Scene_Path,file, 'test').max(0)
				p = np.append(RGBData_test[counter_test],FlowData_test[counter_test]/10)
				p = np.append(p,ObjData)
				# p = np.append(p,SceneData/5)
				p = np.append(p,PoseData)
				test_data = np.array([p])
				## coarse label
				gt = file.split('.')[0][:-4]
				test_label_group_n = -1
				for group_n in range(NUM_SUBGROUP):
					if gt in category_group[group_n]:
						test_label_group_n = group_n*100+category_group[group_n].index(gt)
				test_label = test_label_group_n
				## scores
				scores = coarse_model.predict(test_data)
				score = scores.sum(axis=0)
				preds = []
				for i in range(k1 + 1):
					preds.append(score.argmax())
					score[score.argmax()] = -100
				## load sub fcmodels
				pred_classes = []
				dist = []
				pred_class_score = []
				iii = 0
				for i in range(k1 + 1):
					pred_classes.append(fine_model[preds[i]].predict(test_data))
					pred_class = pred_classes[i].sum(axis=0)
					for ii in range(k2 + 1):
						pred_class_score.append(preds[i] * 100 + pred_class.argmax())
						pred_class[pred_class.argmax()] = -100
						dist.append(np.abs(test_data.sum(axis=0)-action_mean_vector[pred_class_score[i*(k2+1)+ii]].sum(axis=0)).sum())
					iii = len(category_group[preds[i]])
				if pred_class_score[dist.index(min(dist))] == test_label:
					pred = test_label
				else:
					pred = pred_class_score[0]
				predict_result.append(pred)  # model of the pred
				ground_truth.append(test_label)
				if test_label == pred:
					pass
				# else:
				# 	print(file, 'GT:', test_label, ' Pred:', pred)
				counter_test += 1
			print('K1: ',k1+1,' K2:',k2+1,'Test accuracy on test data:', (np.array(predict_result) == np.array(ground_truth)).mean())
		for i in range(ground_truth.__len__()):
			ground_label = category_group[int(ground_truth[i]/100)][ground_truth[i]%100]
			pred_label = category_group[int(predict_result[i]/100)][predict_result[i]%100]
			pred_matrix[category.index(ground_label)][category.index(pred_label)] += 1
		for i in range(pred_matrix.shape[0]):
			print(pred_matrix[i][i]/pred_matrix[i].sum())
		for i in range(pred_matrix.shape[0]):
			print(category[i])