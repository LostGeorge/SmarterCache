import numpy as np
import pandas as pd
import random, math, time, sys
from collections import deque
from sklearn.decomposition import PCA


'''
Initial Stuff
'''

file_name = sys.argv[1]

train_factor = random.uniform(0.6, 0.7)

med_q3 = int(sys.argv[2])
damp_factor = med_q3 / 1.5

def avg_fut_rd(vtime):
    return med_q3
    

# TODO: Implement the vtime of the request; probably use the head of each feature vector?
class CacheNet(nn.Module):

    N_TRUE_FEATURES = 14

    def __init__(self, p=0.0):
        super(CacheNet, self).__init__()
        self.in_layer = nn.Linear(self.N_TRUE_FEATURES, 64)
        self.in_drop = nn.Dropout(p=p)
        #self.h1_layer = nn.Linear(1024,256)
        #self.h1_drop = nn.Dropout(p=p)
        #self.h2_layer = nn.Linear(64,40)
        #self.h2_drop = nn.Dropout(p=p)
        self.h3_layer = nn.Linear(64,32)
        self.h3_drop = nn.Dropout(p=p)
        self.h4_layer = nn.Linear(32,10)
        self.h4_drop = nn.Dropout(p=p)
        self.out_layer = nn.Linear(10,1)

    # Head of feature vector is the virtual time (column 0)
    def forward(self, inputs):
        #inputs = inputs[:, 1:]
        inputs = self.in_layer(inputs)
        #inputs = F.relu(self.h1_layer(inputs))
        #inputs = self.h1_drop(inputs)
        #inputs = F.relu(self.h2_layer(inputs))
        #inputs = self.h2_drop(inputs)
        inputs = F.relu(self.h3_layer(inputs))
        inputs = self.h3_drop(inputs)
        inputs = F.relu(self.h4_layer(inputs))
        inputs = self.h4_drop(inputs)
        inputs = self.out_layer(inputs)
        output = inputs

        return output


'''
Data Processing Section
'''
N_FEATURES = 19 #???? idk

def get_next_access_dist(id_ser, dummy_length):

    reverse_data = deque()
    next_access_dist = []
    next_access_time = {}

    for req in id_ser:
        reverse_data.appendleft(req)

    for index, req in enumerate(reverse_data):
        if req in next_access_time:
            next_access_dist.append(index - next_access_time[req])
        else:
            next_access_dist.append(dummy_length)
        next_access_time[req] = index
    
    next_access_dist.reverse()
    return next_access_dist


def create_dist_df(feature_df, samples, dists, start_time, eval=False):

    # Returns logistic virtual distance since t_now (sigmoid)
    def get_logit_dist(next_access_dist, timestamp, delta):

        if next_access_dist != -1:
            return 1/(1 + np.exp(-(next_access_dist - delta
                 - avg_fut_rd(timestamp))/damp_factor))
        else: # return 1
            return 1

    
    train_data = []
    for t in samples:
        
        #delta = int(random.random() * dists[t - start_time])
        delta = int(random.random()**2 * dists[t - start_time])
        
        if delta + t < len(feature_df) + start_time:
            if not eval:
                logit_dist_val = get_logit_dist(dists[t], t, delta)
            else:
                logit_dist_val = 0 # dummy value

            ser = feature_df.loc[t].to_list()
            for i in range(5,10):
                if ser[i] == 0:
                    ser[i] = med_q3*20
            ser += [delta] + [logit_dist_val]
            train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES)) + ['final']))

    return full_df

def gen_data(df):

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]
    
    tc = time.time()
    n_samples = max(500000, int(df_len/5))

    time_samples = np.random.randint(0, df_len, size=n_samples)
    train_dists = get_next_access_dist(df, df_len)
    
    df = create_dist_df(df, time_samples, train_dists, 0)

    td = time.time()
    print('Time to Construct DataFrame: ', str(td-tc))

    return df

'''
Pytorch Integration Section
'''
t1 = time.time()

df = gen_data(pd.read_csv('ranktest/features/' + file_name + '_feat16.csv'))
#normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)
def normalizing_func(x):
    stdev = np.std(x, axis=0)
    ret = np.zeros(x.shape, dtype='float64')
    for i in range(ret.shape[1]): # iterate across columns
        if stdev[i] != 0:
            ret[:, i] = (x[:, i] - np.mean(x[:, i]))/stdev[i]
    return ret

df_feat = df.drop(columns=[0,1,2,3,4,
    'final']).astype('float64').to_numpy()

df_feat = normalizing_func(df_feat)

pca = PCA()
pca.fit(df_feat.numpy())
variances = pca.explained_variance_ratio_
ser = pd.Series(variances)
ser.to_csv('eval/pca/' + file_name + '_pca.csv')
