import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random, math, time, sys, heapq

from collections import deque
from PyMimircache import Cachecow
from heapdict import heapdict



'''
Initial I/O Parameters and Train/Test Split
'''

file_name = sys.argv[1]
df = pd.read_csv('feat/features/' + file_name + '_.csv')

df = df.iloc[int(0.05*df.shape[0]):int(0.95*df.shape[0])].reset_index(drop=True)

# Returns array of next access distances for each index in trace
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

train_factor = random.uniform(0.6, 0.7)

train_dists = get_next_access_dist(df.loc[:int(train_factor * len(df)), 'id'], len(df))
eval_dists = get_next_access_dist(df.loc[int(train_factor * len(df)):, 'id'], len(df))

sig_cent = np.percentile(train_dists, 75.0)
damp_factor = sig_cent / 1.5

#sig_cent = int(sys.argv[2])
#damp_factor = sig_cent / 1.5

'''
Neural Network Definition
'''
class CacheNet(nn.Module):

    N_TRUE_FEATURES = 14

    def __init__(self, p=0.0):
        super(CacheNet, self).__init__()
        self.in_layer = nn.Linear(self.N_TRUE_FEATURES, 64)
        self.in_drop = nn.Dropout(p=p)
        self.h3_layer = nn.Linear(64,32)
        self.h3_drop = nn.Dropout(p=p)
        self.h4_layer = nn.Linear(32,10)
        self.h4_drop = nn.Dropout(p=p)
        self.out_layer = nn.Linear(10,1)

    def forward(self, inputs):
        inputs = self.in_layer(inputs)
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
N_FEATURES = 19 

# General Purpose Function to Create DataFrame
def create_dist_df(feature_df, samples, dists, start_time, eval=False):
    
    # Returns logistic virtual distance since t_now (sigmoid)
    def get_logit_dist(next_access_dist, delta):
        return 1/(1 + np.exp(-(next_access_dist - delta
                 - sig_cent)/damp_factor))

    
    train_data = []
    for t in samples:
        delta = 0
        if not eval:
            # Sample from linear decay
            delta = int(random.random()**2 * dists[t - start_time])
        
        # Check if t_now is outside scope of trace
        if delta + t < len(feature_df) + start_time:
            if not eval:
                logit_dist_val = get_logit_dist(dists[t - start_time], delta)
            else:
                logit_dist_val = 0 # dummy value for evaluation

            ser = feature_df.loc[t].to_list()

            # Check previous reuse distances for dummy 0 value (no previous reuse).
            # Update to large value based on sigmoid center.
            for i in range(5,10):
                if ser[i] == 0:
                    ser[i] = sig_cent*20
            ser += [delta] + [logit_dist_val]
            train_data.append(ser)

    full_df = pd.DataFrame(data=train_data,
        columns=(list(range(N_FEATURES)) + ['final']))

    return full_df

# Generates Training and Evaulation Data
def gen_train_eval_data(df, train_dists, eval_dists):

    df_len = df.shape[0]
    
    tc = time.time()

    n_samples = max(500000, int(df_len * train_factor/5))

    time_samples = np.random.randint(0, int(train_factor * df_len), size=n_samples)
    learn_data = df.iloc[:int(train_factor * df_len)]
    
    train_df = create_dist_df(learn_data, time_samples, train_dists, 0)

    td = time.time()
    print('Time to Construct Training DataFrame: ', str(td-tc))

    time_samples = list(range(int(train_factor * df_len), df_len))
    eval_data = df.iloc[int(train_factor * df_len):]

    eval_df = create_dist_df(eval_data, time_samples, eval_dists, int(train_factor * df_len), eval=True)

    # For later purposes of PyMimircache readers
    with open('temporary/temp_trace_' + file_name + '.txt', 'w+') as f:
        for i in eval_data['id']:
            f.write(str(i))
            f.write('\n')

    te = time.time()
    print('Time to Construct Evaluation DataFrame: ', str(te-td))

    return train_df, eval_df

'''
Pytorch Integration Section
'''
t1 = time.time()

train_df, eval_df = gen_train_eval_data(df, train_dists, eval_dists)

# Normalizes np.ndarray-like Feature Matrix to have Mean 0 and 
# Variance 1 over Each Feature
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


model = CacheNet(p=0.5)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.045)

lambda1 = lambda epoch: 0.99
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

model.train()
print(train_target)

# Training Loop
for t in range(400):
    # Forward Pass
    y_pred = model(train_feat)
    y_pred = torch.sigmoid((y_pred - sig_cent)/damp_factor)

    # Loss
    loss = criterion(y_pred, train_target)
    if (t % 20 == 0):
        print(t, loss.item())
    if (t % 40 == 0):
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
    y_pred = torch.sigmoid((y_pred - sig_cent)/damp_factor)
    print('Training Score:')
    print(criterion(y_pred, train_target))
    print(y_pred)  

    t2 = time.time()
    print('Time to Train: ', str(t2-t1))

    # Linear time scan to determine the indices for the n eviction candidates
    def select_indices(eval_lst, n):
        ret = heapdict()
        curr_min = float('-inf')
        for i, fut_dist in enumerate(eval_lst):
            if fut_dist > curr_min or len(ret) < n:
                ret[i] = fut_dist
                if len(ret) > n:
                    ret.popitem()
                curr_min = ret.peekitem()[1]
        return list(ret.keys())

    def eviction_process(cache_dict, sample_size, n_evicts, ts):
        # Ensure that at least one element is evicted
        if sample_size == 0:
            sample_size = 1
        if n_evicts == 0:
            n_evicts = 1
        
        # Randomly sample the cache for eviction candidates
        reqs = random.sample(cache_dict.items(), sample_size)
        id_lst = [req[0] for req in reqs]
        eval_features = np.zeros((sample_size, 14), dtype='float64')

        for i, (ident, ind) in enumerate(reqs):
            eval_i = eval_feat[ind]
            eval_i[-1] = ts - cache_dict[ident] # changing the delta value
            eval_features[i] = eval_i
        
        # Run Model
        eval_features = torch.Tensor(normalizing_func(eval_features))
        eval_values = model(eval_features)
        eval_values = [eval_values[i, 0] for i in range(len(eval_values))]

        # Get indices for eviction candidates
        evict_inds = select_indices(eval_values, n_evicts)

        # For some reason sorting is slightly faster even though it shouldn't be.
        # Decline to do so for elegance purposes.
        #evict_inds = np.argsort(eval_values)[-1*n_evicts:] 
        
        # Do deletions
        for ind in evict_inds:
            del cache_dict[id_lst[ind]]

    cache_sizes = sys.argv[2:]
    cache_sizes = [int(size) for size in cache_sizes]

    # Manually Get Hit Ratios for Model
    length = len(eval_ids)
    hit_ratios = []
    
    for cache_size in cache_sizes:
        cache = {}
        n_hits = 0
        for ts, ident in enumerate(eval_ids):
            if ident in cache:
                n_hits += 1
                cache[ident] = ts # update cache
            else:
                cache[ident] = ts # add to cache
                if len(cache) > cache_size:
                    # Eviction Process
                    eviction_process(cache, int(cache_size/50), int(cache_size/500), ts)

        print(cache_size)

        hit_ratios.append(n_hits / length)    

    t3 = time.time()
    print('Time to Evaluate: ', str(t3-t2))


'''
Comparison and Plotting
'''

# Setup PyMimircache for Comparison
c = Cachecow()
c.open('temporary/temp_trace_' + file_name + '.txt')

comparison_lst = ['Optimal', 'LRU', 'LFU', 'Random', 'SLRU', 'ARC']

# Chaining lol
comparison_hrs = [[c.profiler(alg, cache_size=size, use_general_profiler=True).get_hit_ratio()[-1]
    for size in cache_sizes] for alg in comparison_lst]


comparison_hrs.append(hit_ratios)
comparison_lst.append('SmarterCache')

comparison_df = pd.DataFrame(data=comparison_hrs, index=comparison_lst,
    columns=cache_sizes)
print(file_name)
print(comparison_df)