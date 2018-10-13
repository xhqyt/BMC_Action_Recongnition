import numpy as np
import json

def read_classes(path):
	return list(eval(open(path, 'r').read()).values())
def parse_json(json_file, classes_file,data_type='max'):
	classes =read_classes(classes_file)
	with open(json_file) as f:
		jdata = json.load(f)
	nsample = len(jdata)
	assert nsample > 0
	ndim = len(jdata[0]['clips'][0]['features'])

	mean_data = np.zeros((nsample, ndim))
	max_data = np.zeros((nsample, ndim))
	label = np.zeros(nsample)
	for idx, data in enumerate(jdata):
		label[idx] = classes.index(data['video'].split('.')[0][:-4])
		curr_data = []
		for clip in data['clips']:
			clip_feature = np.array(clip['features'])
			curr_data.append(clip_feature)
		curr_data = np.array(curr_data)
		curr_mean_data = np.mean(curr_data, axis=0)
		curr_max_data = np.max(curr_data, axis=0)
		mean_data[idx, :] = curr_mean_data
		max_data[idx, :] = curr_max_data
	mean_data = np.array(mean_data)
	max_data = np.array(max_data)
	label = np.array(label)

	if data_type == 'max':
		return max_data, label
	elif data_type == 'mean':
		return mean_data,label
	else:
		print('Wrong Data Type!Please enter max or mean as data type :)')
