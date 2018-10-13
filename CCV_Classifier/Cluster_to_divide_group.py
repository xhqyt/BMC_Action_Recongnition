import numpy as np
import pickle
from sklearn.cluster import KMeans



catagories_group_save = [['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE', 'NONMUSICPERFORMANCE', 'PARADE'],
                    ['BASKETBALL', 'BASEBALL', 'SOCCER', 'BIKING', 'PLAYGROUND'],
                    ['ICESKATING', 'SKIING', 'SWIMMING', 'BEACH']]
catagories_group = [['BIKING', 'PLAYGROUND', 'PARADE'],
                    ['BASKETBALL', 'BASEBALL', 'SOCCER', 'NONMUSICPERFORMANCE'],
                    ['ICESKATING', 'SKIING','SWIMMING', 'BEACH'],
                    ['GRADUATION', 'BIRTHDAY', 'WEDDINGCEREMONY', 'WEDDINGDANCE']]

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
    data_label.append(catagories_group[int(i/100)][int(i%100)])
while(1):
    estimator = KMeans(n_clusters=4)
    estimator.fit(data)
    label_pred = estimator.labels_
    label_pred = label_pred.tolist()
    label_num = []
    for i in range(4):
        label_num.append(label_pred.count(i))
    if min(label_num) >= 3 and max(label_num) <= 5:
        break
catagories_group_new = [[],[],[],[]]
for i in range(15):
    catagories_group_new[label_pred[i]].append(data_label[i])
print(catagories_group_new)



