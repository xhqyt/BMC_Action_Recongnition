import numpy as np
import os
from sklearn.cluster import KMeans
from yue.json import read_classes
from yue.load_data import load_pkl
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
Root_Path = '../dataset/'
Motion_Path = Root_Path + 'Motion_feature'
Obj_Path = Root_Path + 'Obj_feature'
classes_file =  '../dataset/list/class_names.txt'
train_json_file = Motion_Path + '/train_data.json'
test_json_file = Motion_Path + '/test_data.json'
mean_vector_file = Root_Path + 'vector_map/action_mean_vector_next_64f.pkl'
category_group = [['clap', 'turn', 'wave', 'brush_hair', 'chew', 'drink', 'eat', 'kiss', 'laugh', 'pour', 'smile', 'smoke', 'talk'],
				  ['climb_stairs', 'jump', 'pullup', 'ride_bike', 'ride_horse', 'run', 'kick_ball'],
				  ['draw_sword', 'fencing', 'golf', 'hit', 'kick', 'shoot_bow', 'swing_baseball', 'sword_exercise', 'sword', 'throw', 'catch', 'dribble', 'shoot_ball'],
				  ['hug', 'punch', 'shake_hands', 'sit', 'situp', 'stand', 'walk', 'shoot_gun', 'pick', 'push', 'pushup'],
				  ['climb', 'dive', 'fall_floor', 'cartwheel', 'flic_flac', 'handstand', 'somersault']]
classes = read_classes(classes_file)
CLASS_NUM = np.shape(classes)[0]

action_mean_vector = load_pkl(mean_vector_file)
data = None
data_label = []
CLUSTER_NUM = 5
for i in action_mean_vector:
	if data is None:
		data = [action_mean_vector[i]]
	else:
		data = np.append(data,[action_mean_vector[i]],0)
	data_label.append(category_group[int(i/100)][int(i%100)])
while(1):
	estimator = KMeans(n_clusters=CLUSTER_NUM)
	estimator.fit(data)
	label_pred = estimator.labels_
	label_pred = label_pred.tolist()
	label_num = []
	for i in range(CLUSTER_NUM):
		label_num.append(label_pred.count(i))
	if min(label_num) >= 6 and max(label_num) <= 20:
		break
catagories_group_new = [[],[],[],[],[],[],[],[],[],[]]
for i in range(CLASS_NUM):
	catagories_group_new[label_pred[i]].append(data_label[i])
print(catagories_group_new)

