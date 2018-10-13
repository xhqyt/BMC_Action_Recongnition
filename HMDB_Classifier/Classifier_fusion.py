import numpy as np
import time
import os
from keras.utils import np_utils
from keras import Sequential
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
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
NUM_CLASSES = np.shape(classes)[0]
predict_matrix = np.zeros((51,51))
if __name__ == '__main__':
	RGBData_train, _ = parse_json(RGB_train_json_file, classes_file, data_type='max')
	RGBData_test, _ = parse_json(RGB_test_json_file, classes_file, data_type='max')
	FlowData_train, _ = parse_json(Flow_train_json_file, classes_file, data_type='max')
	FlowData_test, _ = parse_json(Flow_test_json_file, classes_file, data_type='max')
	## load train data
	counter_train = 0
	print('Begin loading data...')
	file_list = os.listdir(Obj_Path + '/train')
	file_list.sort()
	train_data = None
	train_label = np.array([])
	for file in file_list:
		ObjData = get_feature(Obj_Path, file, 'train').max(axis=0)/5
		PoseData = get_feature(Pose_Path, file, 'train')
		SceneData = get_feature(Scene_Path,file, 'train').max(0)
		gt = file.split('.')[0][:-4]
		train_label = np.append(train_label, classes.index(gt))  ## append label
		p = np.append(RGBData_train[counter_train], FlowData_train[counter_train]/10)
		p = np.append(p,ObjData)
		# p = np.append(ObjData,SceneData/5)
		p = np.append(p,PoseData)
		if train_data is not None:  ## append data
			train_data = np.append(train_data, [p], 0)
		else:
			train_data = [p]
		counter_train += 1
	train_label = np_utils.to_categorical(train_label, NUM_CLASSES)
	print(train_data.shape)
	## load test data
	counter_test = 0
	file_list = os.listdir(Obj_Path + '/test')
	file_list.sort()
	test_data = None
	test_label = np.array([])
	for file in file_list:
		ObjData = get_feature(Obj_Path, file, 'test').max(axis=0)/5
		PoseData = get_feature(Pose_Path, file, 'test')
		SceneData = get_feature(Scene_Path,file, 'test').max(0)
		gt = file.split('.')[0][:-4]
		test_label = np.append(test_label, classes.index(gt))
		p = np.append(RGBData_test[counter_test],FlowData_test[counter_test]/10)
		p = np.append(p,ObjData)
		# p = np.append(ObjData,SceneData/5)
		p = np.append(p,PoseData)
		if test_data is not None:  ## append data
			test_data = np.append(test_data, [p], 0)
		else:
			test_data = [p]
		counter_test += 1
	ground_label = test_label
	test_label = np_utils.to_categorical(test_label, NUM_CLASSES)
	print(test_data.shape)
	## load motion feature & fusion features
	model = Sequential()
	model.add(Dense(128, input_shape=(train_data.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
	model.add(Dropout(0.6))
	model.add(Dense(NUM_CLASSES, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
	## model fit
	model.fit(train_data, train_label, shuffle=True, validation_data=(test_data, test_label), batch_size=128,epochs=100, verbose=1)
	print(model.evaluate(test_data,test_label))
	predict_truth = np.array(model.predict(test_data))
	for i in range(ground_label.shape[0]):
		predict_label = predict_truth[i].argmax()
		predict_matrix[int(ground_label[i])][predict_label] += 1
	for i in range(predict_matrix.shape[0]):
		print(predict_matrix[i][i]/predict_matrix[i].sum())
	for i in range(predict_matrix.shape[0]):
		print(classes[i])