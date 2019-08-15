import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, time, sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from collections import deque
from shap import DeepExplainer
from shap import summary_plot



'''
Initial Stuff
'''

file_name = sys.argv[1]

train_factor = random.uniform(0.6, 0.7)

med_q3 = int(sys.argv[2])
damp_factor = med_q3 / 1.5

def avg_fut_rd(vtime):
    return med_q3
    


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

def gen_train_eval_data(df):

    df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)
    df_len = df.shape[0]
    #df.insert(loc=2, column='vtime', value=df.index)
    
    tc = time.time()
    n_samples = max(500000, int(df_len * train_factor/5))

    time_samples = np.random.randint(0, int(train_factor * df_len), size=n_samples)
    learn_data = df.iloc[:int(train_factor * df_len)]
    train_dists = get_next_access_dist(learn_data['id'], df_len)
    
    train_df = create_dist_df(learn_data, time_samples, train_dists, 0)

    td = time.time()
    print('Time to Construct Training DataFrame: ', str(td-tc))

    time_samples = list(range(int(train_factor * df_len), df_len))
    eval_data = df.iloc[int(train_factor * df_len):]
    eval_dists = get_next_access_dist(eval_data['id'], df_len)

    eval_df = create_dist_df(eval_data, time_samples, eval_dists, int(train_factor * df_len), eval=True)

    te = time.time()
    print('Time to Construct Evaluation DataFrame: ', str(te-td))

    return train_df, eval_df

'''
Pytorch Integration Section
'''
t1 = time.time()

train_df, eval_df = gen_train_eval_data(pd.read_csv('feat/features/' + file_name + '_feat16.csv'))
#normalizing_func = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)
def normalizing_func(x):
    stdev = np.std(x, axis=0)
    ret = np.zeros(x.shape, dtype='float64')
    for i in range(ret.shape[1]): # iterate across columns
        if stdev[i] != 0:
            ret[:, i] = (x[:, i] - np.mean(x[:, i]))/stdev[i]
    return ret

train_feat = train_df.drop(columns=[0,1,2,3,4,
    'final']).astype('float64').to_numpy()
train_target = train_df[['final']].astype('float64').to_numpy()

train_feat = normalizing_func(train_feat)

train_feat = torch.tensor(train_feat, dtype=torch.float)
train_target = torch.tensor(train_target, dtype=torch.float)

eval_ids = eval_df[1].to_list() # For later
eval_feat = eval_df.drop(columns=[0,1,2,3,4,
    'final']).astype('float64').to_numpy()
eval_feat = normalizing_func(eval_feat)
eval_feat = torch.tensor(eval_feat, dtype=torch.float)

model = CacheNet(p=0.5)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.04)

lambda1 = lambda epoch: 0.99
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#train_feat = torch.randn(len(train_target), 34)

model.train()
print(train_target)

for t in range(400):
    # Forward Pass
    y_pred = model(train_feat)
    y_pred = torch.sigmoid((y_pred - med_q3)/damp_factor)

    # Loss
    loss = criterion(y_pred, train_target)
    if (t % 10 == 0):
        print(t, loss.item())
    if (t % 20 == 0):
        print(y_pred)

    # Backward Pass And Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# Evaluation of Model
with torch.no_grad():

    model.eval()
    # Training Score
    y_pred = model(train_feat)
    y_pred = torch.sigmoid((y_pred - med_q3)/damp_factor)
    print('Training Score:')
    print(criterion(y_pred, train_target))
    print(y_pred)  

    t2 = time.time()
    print('Time to Train: ', str(t2-t1))

background_size = 10000
n_analyses = 250

background = train_feat[np.random.randint(0, len(train_feat),
    size=background_size)]
exp = DeepExplainer(model, background)

analyses = eval_feat[np.random.randint(0, len(eval_feat),
    size=n_analyses)]
shap_values = exp.shap_values(analyses)

df = pd.DataFrame(shap_values,
    columns=['d_1', 'd_2', 'd_3', 'd_4', 'd_5', 'f_64', 'f_256', 'f_1024',
    'f_4096', 'f_16384', 'r_64', 'r_256', 'r_1024', 'delta'])

mins = []
maxes = []
means = []
meds = []
stddevs = []
q1s = []
q3s = []

for i, col in df.iteritems():
    shap_vals = col.to_numpy()
    mins.append(min(shap_vals))
    maxes.append(max(shap_vals))
    means.append(np.mean(shap_vals))
    meds.append(np.median(shap_vals))
    stddevs.append(np.std(shap_vals))
    q1s.append(np.percentile(shap_vals, 25.0))
    q3s.append(np.percentile(shap_vals, 75.0))

df.loc['min'] = mins
df.loc['max'] = maxes
df.loc['mean'] = means
df.loc['median'] = meds
df.loc['std dev'] = stddevs
df.loc['P25'] = q1s
df.loc['P75'] = q3s

df.to_csv('eval/shap/' + file_name + '_shap_results.csv')

t3 = time.time()
print('Time to Evaluate SHAP: ', str(t3-t2))

shap_df = df[:n_analyses]
col_names = shap_df.columns
shap_values = shap_df.to_numpy().astype('float64')

summary_plot(shap_values, features=analyses.numpy(), feature_names=col_names)

extra = ''
if len(sys.argv) > 3:
    extra = sys.argv[3]

plt.savefig('eval/shap/' + file_name + extra + '_IMG.png')
plt.xscale('symlog')
plt.savefig('eval/shap/' + file_name + extra + '_IMG_log.png')

    

