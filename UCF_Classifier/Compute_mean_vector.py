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
BGDatapath = basepath + 'Scene_feature_fix'
RGBDatapath = basepath + 'rgb_feature'
FlowDatapath = basepath + 'flow_feature'
ObjDatapath = basepath + 'Obj_feature_fix'
PoseDatapath = basepath + 'pose_feature_fix'
# PoselstmDatapath = basepath + 'Pose_lstm'
subfc_path = 'sub_fc'
def get_feature(featurepath, filename):
	if os.path.isfile(os.path.join(featurepath, 'train', filename)):
		with open(os.path.join(featurepath, 'train', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	else:
		with open(os.path.join(featurepath, 'test', filename), 'rb') as f:
			ret = np.array(pickle.load(f,encoding='latin1'), dtype=np.float)
	return ret
catagories_group_save = [['GolfSwing','JavelinThrow','PoleVault','ThrowDiscus','Archery','Shotput','HammerThrow',],
					['BoxingPunchingBag','BoxingSpeedBag','Punch','Fencing','TableTennisShot','SumoWrestling',],
					['Drumming','PlayingCello','PlayingDaf','PlayingGuitar','PlayingPiano','PlayingSitar','PlayingTabla','PlayingViolin','PlayingDhol','PlayingFlute',],
					['BaseballPitch','BasketballDunk','Bowling','Billiards','VolleyballSpiking','Basketball',],
					['ApplyEyeMakeup','ApplyLipstick','BlowDryHair','BrushingTeeth','ShavingBeard','Haircut','HeadMassage',],
					['CuttingInKitchen','Knitting','PizzaTossing','MoppingFloor','Hammering','WritingOnBoard','BlowingCandles','Typing','Mixing',],
					['HulaHoop','Nunchucks','YoYo','SkateBoarding','TrampolineJumping','IceDancing','SalsaSpin','JumpingJack','JugglingBalls','SoccerJuggling'],
					['BodyWeightSquats','HandStandPushups','HandstandWalking','PullUps','RockClimbingIndoor','RopeClimbing','Swing','TaiChi','WallPushups','BenchPress','CleanAndJerk','PushUps','Lunges',],
					['Diving','Skijet','FrontCrawl','Surfing','BreastStroke','Kayaking','Skiing','Rowing','CliffDiving','Rafting',],
					['StillRings','PommelHorse','ParallelBars','LongJump','HighJump','UnevenBars','JumpRope','BalanceBeam','FloorGymnastics','TennisSwing','CricketBowling','CricketShot',],
					['BabyCrawling','WalkingWithDog','BandMarching','MilitaryParade','SkyDiving','Biking','HorseRiding','HorseRace','SoccerPenalty','FieldHockeyPenalty','FrisbeeCatch',]]
catagories_group = [['Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute'],
					['TrampolineJumping', 'Swing', 'TaiChi', 'Skiing', 'IceDancing', 'SkateBoarding', 'WalkingWithDog', 'Biking', 'HorseRiding', 'MilitaryParade', 'BandMarching', 'GolfSwing'],
					['ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'HeadMassage', 'BlowingCandles', 'Haircut', 'Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'],
					['JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'BenchPress', 'PullUps', 'WallPushups', 'JumpRope'],
					['Archery', 'TableTennisShot', 'Fencing', 'SoccerJuggling', 'VolleyballSpiking', 'Basketball', 'TennisSwing', 'BasketballDunk', 'HammerThrow'],
					['YoYo', 'Nunchucks', 'JugglingBalls', 'WritingOnBoard', 'PizzaTossing', 'BoxingSpeedBag', 'BoxingPunchingBag', 'Punch', 'HulaHoop', 'Bowling', 'SalsaSpin', 'SumoWrestling'],
					['Skijet', 'FrontCrawl', 'Surfing', 'BreastStroke', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'HorseRace'],
					['StillRings', 'PommelHorse', 'ParallelBars', 'UnevenBars', 'BalanceBeam', 'FloorGymnastics', 'Billiards', 'RopeClimbing', 'Diving'],
					['PoleVault', 'FrisbeeCatch', 'JavelinThrow', 'ThrowDiscus', 'HighJump', 'Shotput', 'LongJump', 'BaseballPitch', 'CricketBowling', 'CricketShot', 'SoccerPenalty', 'FieldHockeyPenalty'],
					['RockClimbingIndoor', 'HandstandWalking', 'BabyCrawling', 'Hammering', 'MoppingFloor', 'HandStandPushups', 'PushUps', 'SkyDiving']]
catagories_group_motion = [['Skijet', 'FrontCrawl', 'Surfing', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'Skiing', 'SkyDiving'], ['MilitaryParade', 'FloorGymnastics', 'PoleVault', 'HighJump', 'SkateBoarding', 'Biking', 'Swing', 'RopeClimbing', 'TrampolineJumping', 'HandstandWalking', 'Basketball', 'BasketballDunk', 'PommelHorse', 'ParallelBars', 'BalanceBeam'], ['Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'], ['TableTennisShot', 'JugglingBalls', 'BabyCrawling', 'Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute'], ['IceDancing', 'WalkingWithDog', 'BandMarching', 'GolfSwing', 'FrisbeeCatch', 'BaseballPitch', 'CricketShot', 'FieldHockeyPenalty', 'TennisSwing', 'JavelinThrow', 'ThrowDiscus', 'Shotput', 'LongJump', 'CricketBowling', 'HammerThrow', 'SoccerJuggling', 'VolleyballSpiking', 'SumoWrestling'], ['BreastStroke', 'RockClimbingIndoor', 'StillRings', 'UnevenBars', 'Billiards', 'Diving'], ['Bowling', 'WritingOnBoard', 'PizzaTossing', 'BoxingPunchingBag', 'Punch', 'BenchPress', 'PullUps', 'Hammering', 'MoppingFloor', 'PushUps', 'HandStandPushups'], ['BoxingSpeedBag', 'ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'BlowingCandles', 'Haircut', 'HeadMassage'], ['YoYo', 'Nunchucks', 'SalsaSpin', 'Archery', 'JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'WallPushups', 'JumpRope', 'TaiChi', 'Fencing', 'HulaHoop'], ['HorseRace', 'HorseRiding', 'SoccerPenalty']]

NUM_SUBGROUP = catagories_group_motion.__len__()
test_data = None
test_label = np.array([])
picklelist = os.listdir(os.path.join(RGBDatapath, 'train'))[::1]
picklelist.sort()
data = {}
data_label = {}
for pickle_filename in picklelist:
	RGBData = get_feature(RGBDatapath, pickle_filename)
	FlowData = get_feature(FlowDatapath, pickle_filename)
	## VGG verison BGData & ObjData
	# BGData = get_feature(BGDatapath, pickle_filename.split('.')[0] + "1.pickle")
	# BGData_TMP = get_feature(BGDatapath, pickle_filename.split('.')[0] + "2.pickle")
	# BGData = np.row_stack((BGData, BGData_TMP))
	# ObjData = get_feature(ObjDatapath, pickle_filename.split('.')[0] + "1.pickle")
	# ObjData_TMP = get_feature(ObjDatapath, pickle_filename.split('.')[0] + "2.pickle")
	# ObjData = np.row_stack((ObjData, ObjData_TMP))

	# RGBData = np.append(RGBData[0:15].mean(0), RGBData[16:31].mean(0))
	# FlowData = np.append(FlowData[0:15].mean(0), FlowData[16:31].mean(0))
	# ## load pose feature
	# PoseData = None
	# PoseData_TMP = get_feature(PoseDatapath, pickle_filename)
	# PoseData_TMP[0] /= 320
	# PoseData_TMP[1] /= 240
	# for i in range(PoseData_TMP.shape[0]):
	#     for j in range(PoseData_TMP.shape[1]):
	#         if PoseData is None:
	#             PoseData = PoseData_TMP[i][j]
	#         else:
	#             PoseData = np.append(PoseData, PoseData_TMP[i][j])
	BGData = get_feature(BGDatapath,pickle_filename[:-7]+'.pkl')
	ObjData = get_feature(ObjDatapath,pickle_filename[:-7]+'.pkl').mean(0)
	PoseData = get_feature(PoseDatapath,pickle_filename[:-7]+'.pkl')
	# PoseData = np.append(PoseData[0:2].mean(0),PoseData[3:5].mean(0))
	# RGBData_G = np.array(np.gradient(RGBData))[1]
	# RGBData = np.array([RGBData_G.mean(0), RGBData_G.mean(0)])
	# FlowData_G = np.array(np.gradient(FlowData))[1]
	# FlowData = np.array([FlowData_G.mean(0), FlowData_G.mean(0)])
	# MRData = np.append(RGBData, FlowData/10)
	# test_data = np.append(MRData, ObjData/3)
	test_data = np.append(ObjData/3,PoseData)
	test_data = np.array([np.append(test_data, BGData.sum(0)/10)])
	# test_data = np.array([MRData])
	gt = pickle_filename.split('_')[1]
	## coarse classify
	test_label_group_n = 0
	for group_n in range(NUM_SUBGROUP):
		if gt in catagories_group_motion[group_n]:
			test_label_group_n = group_n*100 + catagories_group_motion[group_n].index(gt)
	test_label = test_label_group_n
	if test_label in data:
		data[test_label] = data[test_label] + test_data
		data_label[test_label] += 1
	else:
		data[test_label] = test_data
		data_label[test_label] = 1
for key in data.keys():
	data[key] /= data_label[key]
save_file = open('action_mean_vector_con.pkl','wb')
pickle.dump(data,save_file)
save_file.close()