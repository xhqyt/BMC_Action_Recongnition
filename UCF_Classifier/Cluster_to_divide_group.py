import numpy as np
import pickle
from sklearn.cluster import KMeans

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
catagories_group_motion = [['HorseRace', 'IceDancing', 'WalkingWithDog', 'HorseRiding', 'MilitaryParade', 'BandMarching', 'GolfSwing'], ['TableTennisShot', 'YoYo', 'Nunchucks', 'JugglingBalls', 'WritingOnBoard', 'PizzaTossing', 'BoxingSpeedBag', 'BoxingPunchingBag', 'Punch', 'SalsaSpin'], ['ApplyEyeMakeup', 'BlowDryHair', 'BrushingTeeth', 'ShavingBeard', 'ApplyLipstick', 'BlowingCandles', 'Haircut', 'Knitting', 'CuttingInKitchen', 'Mixing', 'Typing'], ['PoleVault', 'JavelinThrow', 'ThrowDiscus', 'HighJump', 'Shotput', 'LongJump', 'CricketBowling', 'HammerThrow', 'Bowling'], ['JumpingJack', 'CleanAndJerk', 'Lunges', 'BodyWeightSquats', 'WallPushups', 'JumpRope', 'TrampolineJumping', 'TaiChi', 'HandstandWalking', 'HandStandPushups', 'Fencing', 'HulaHoop'], ['Drumming', 'PlayingCello', 'PlayingDaf', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PlayingDhol', 'PlayingFlute', 'HeadMassage'], ['Skijet', 'FrontCrawl', 'Surfing', 'BreastStroke', 'Rowing', 'Rafting', 'Kayaking', 'CliffDiving', 'Skiing', 'SkateBoarding', 'Biking'], ['SoccerJuggling', 'VolleyballSpiking', 'Basketball', 'BasketballDunk', 'StillRings', 'PommelHorse', 'ParallelBars', 'UnevenBars', 'BalanceBeam', 'Billiards', 'Diving', 'SumoWrestling'], ['FrisbeeCatch', 'BaseballPitch', 'CricketShot', 'SoccerPenalty', 'FieldHockeyPenalty', 'TennisSwing', 'FloorGymnastics'], ['BenchPress', 'PullUps', 'Swing', 'RockClimbingIndoor', 'BabyCrawling', 'Hammering', 'MoppingFloor', 'PushUps', 'SkyDiving', 'Archery', 'RopeClimbing']]

def opendata():
	try:
		with open('Console/action_mean_vector_motion.pkl','rb') as pkl_file:
			return pickle.load(pkl_file)
	except EOFError:
		return None
action_mean_vector = opendata()
data = None
data_label = []
for i in action_mean_vector:
	if data is None:
		data = action_mean_vector[i]
	else:
		data = np.append(data,action_mean_vector[i],0)
	data_label.append(catagories_group_motion[int(i/100)][i%100])
while(1):
	estimator = KMeans(n_clusters=10)
	estimator.fit(data)
	label_pred = estimator.labels_
	label_pred = label_pred.tolist()
	label_num = []
	for i in range(10):
		label_num.append(label_pred.count(i))
	if min(label_num) >= 3 and max(label_num) <= 20:
		break
catagories_group_new = [[],[],[],[],[],[],[],[],[],[]]
for i in range(101):
	catagories_group_new[label_pred[i]].append(data_label[i])
print(catagories_group_new)




