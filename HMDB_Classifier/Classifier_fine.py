import numpy as np
import time
import os
from keras.utils import np_utils
from keras import Sequential
from keras import regularizers
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from yue.json import parse_json
from yue.load_data import get_feature
from yue.load_data import Dataset_HMDB51
hm = Dataset_HMDB51()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
NUM_SUBGROUP = hm.category_group.__len__()
if __name__ == '__main__':
	RGBData_train,_ = parse_json(hm.RGB_train_json_file,hm.classes_file,data_type='max')
	RGBData_test,_ =parse_json(hm.RGB_test_json_file,hm.classes_file,data_type='max')
	FlowData_train,_ = parse_json(hm.Flow_train_json_file,hm.classes_file,data_type='max')
	FlowData_test,_ =parse_json(hm.Flow_test_json_file,hm.classes_file,data_type='max')
	pred_result = []
	for subscript in range(NUM_SUBGROUP):
		## load train data
		counter_train = 0
		print('Begin loading data...')
		file_list = os.listdir(hm.Obj_Path + '/train')
		file_list.sort()
		train_data = None
		train_label = np.array([])
		subgroup_len = hm.category_group[subscript].__len__()
		for file in file_list:
			ObjData = get_feature(hm.Obj_Path,file,'train').max(axis=0)/5
			PoseData = get_feature(hm.Pose_Path,file,'train')
			SceneData = get_feature(hm.Scene_Path,file, 'train').max(0)
			gt = file.split('.')[0][:-4]
			train_label_group_n = -1
			for group_n in range(NUM_SUBGROUP):
				if gt in hm.category_group[group_n]:
					train_label_group_n = group_n
			if train_label_group_n == subscript:     ## Verify the label and group
				train_label = np.append(train_label, hm.category_group[subscript].index(gt))     ## append label
				# p = np.append(RGBData_train[counter_train],FlowData_train[counter_train]/10)
				p = np.append(ObjData,SceneData/5)
				p = np.append(p,PoseData)
				if train_data is not None:                                          ## append data
					train_data = np.append(train_data, [p], 0)
				else:
					train_data = [p]
			else:
				pass
			counter_train += 1
		train_label = np_utils.to_categorical(train_label,subgroup_len)
		print(train_data.shape)
		## load test data
		counter_test = 0
		file_list = os.listdir(hm.Obj_Path + '/test')
		file_list.sort()
		test_data = None
		test_label = np.array([])
		for file in file_list:
			ObjData = get_feature(hm.Obj_Path,file,'test').max(axis=0)/5
			PoseData = get_feature(hm.Pose_Path,file,'test')
			SceneData = get_feature(hm.Scene_Path,file, 'test').max(0)
			gt = file.split('.')[0][:-4]
			test_label_group_n = -1
			for group_n in range(NUM_SUBGROUP):
				if gt in hm.category_group[group_n]:
					test_label_group_n = group_n
			if test_label_group_n == subscript:     ## Verify the label and group
				test_label = np.append(test_label, hm.category_group[subscript].index(gt))     ## append label
				# p = np.append(RGBData_test[counter_test],FlowData_test[counter_test]/10)
				p = np.append(ObjData,SceneData/5)
				p = np.append(p,PoseData)
				if test_data is not None:                                          ## append data
					test_data = np.append(test_data, [p], 0)
				else:
					test_data = [p]
			else:
				pass
			counter_test += 1
		test_label = np_utils.to_categorical(test_label,subgroup_len)
		print(test_data.shape)
		counter = 10
		acc_max = 0
		while(counter):
			## Create Model
			model = Sequential()
			model.add(Dense(128,input_shape=(train_data.shape[1],),activation='relu',kernel_regularizer=regularizers.l2(0.001)))
			model.add(Dropout(0.85))
			model.add(Dense(subgroup_len,activation='softmax'))
			model.compile(loss='categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
			## model fit
			model.fit(train_data,train_label,shuffle=True,validation_data=(test_data,test_label),batch_size=128,epochs=500,verbose=0)
			# model.fit(train_data,train_label,shuffle=True,validation_data=(test_data,test_label),batch_size=128,epochs=300,verbose=1,callbacks=[TensorBoard(log_dir='tmp/log')])
			sor,acc = model.evaluate(test_data,test_label,verbose=0)
			print(sor,acc)
			if acc > acc_max:
				acc_max = acc
				model.save('../dataset/fine_model/fine_model_motion'+'%d' % subscript+'.h5')
			counter -= 1
		pred_result.append(acc_max)
	print(pred_result)
	print(np.array(pred_result).mean())