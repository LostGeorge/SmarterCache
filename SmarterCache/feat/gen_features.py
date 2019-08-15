import numpy as np
import pandas as pd
import time, random, sys
from collections import defaultdict, deque

def generate_features(df, sample=0, hexadecimal=False):

    if sample != 0:
        if hexadecimal:
            char_lst = [str(i) for i in range(10)] + ['a', 'b', 'c', 'd', 'e']
        else:
            char_lst = [str(i) for i in range(10)]
        take = random.sample(char_lst, sample)
        substring_ser = df['id'].apply(lambda x: x[-1:])
        df = df[substring_ser.isin(take)]
        df = df.reset_index(drop=True)
    
    df_len = df.shape[0]

    # Reuse Distance Features (The last 5 of them, not actual reuse, non-unique)
    last_req_dict = {}
    rd_arr = np.zeros((df_len, 5), dtype=np.float64)
    
    # Frequency Features
    freq_arr = np.zeros((df_len, 5), dtype=np.float64)
    freq1_size = 64
    freq2_size = 256
    freq3_size = 1024
    freq4_size = 4096
    freq5_size = 16384
    freq_size_lst = [freq1_size, freq2_size, freq3_size, freq4_size, freq5_size]

    freq1_flg = False
    freq2_flg = False
    freq3_flg = False
    freq4_flg = False
    freq5_flg = False
    freq_flg_lst = [freq1_flg, freq2_flg, freq3_flg, freq4_flg, freq5_flg]

    # Dictionary of id to deque of vtimes of previous access for that id
    freq1_dict = {}
    freq2_dict = {}
    freq3_dict = {}
    freq4_dict = {}
    freq5_dict = {}
    freq_dict_lst = [freq1_dict, freq2_dict, freq3_dict, freq4_dict, freq5_dict]

    # Request Rate Features
    req_rate_arr = np.zeros((df_len, 3), dtype=np.float64)
    req2_flg = False
    req4_flg = False
    req6_flg = False
    req_flg_lst = [req2_flg, req4_flg, req6_flg]


    def next_req_deletion(id_to_remove, freq_dict):
        deq = freq_dict[id_to_remove]
        if len(deq) == 1:
            del freq_dict[id_to_remove]
        else:
            deq.popleft()

    # dict_ind is 'i' in the iteration
    def next_req_insertion(dict_ind, freq_dict, req_id, req_vtime, freq_arr):
        if req_id not in freq_dict:
            freq_dict[req_id] = deque([req_vtime])
        else:
            freq_dict[req_id].append(req_vtime)
        
        freq_arr[req_vtime, dict_ind] = len(freq_dict[req_id])  


    for index, series in df.iterrows():
        id = series['id']

        # Populate Reuse Distance Array
        if id in last_req_dict:
            prev_vtime = last_req_dict[id]
            rd_arr[index, 0] = index - prev_vtime
            rd_arr[index, 1:5] = rd_arr[prev_vtime, 0:4]
        last_req_dict[id] = index

        # Populate Frequency Array
        for i in range(5):
            if index == freq_size_lst[i]:
                freq_flg_lst[i] = True
            
            if freq_flg_lst[i]:
                id_remove = df.iloc[index - freq_size_lst[i], 1]
                next_req_deletion(id_remove, freq_dict_lst[i])
            next_req_insertion(i, freq_dict_lst[i],
                id, index, freq_arr)

        # Populate Request Rate Array
        for i in range(3):
            if index == freq_size_lst[i]/2:
                req_flg_lst[i] = True
            elif index == df_len - freq_size_lst[i]/2:
                req_flg_lst[i] = False
            
            if req_flg_lst[i]:
                time_delta = max(0.001, df.loc[index + freq_size_lst[i]/2]['time']
                    - df.loc[index - freq_size_lst[i]/2]['time'])
                req_rate_arr[index, i] = freq_size_lst[i] / time_delta

    # Convert to float frequencies
    for i in range(5):
        freq_arr[:,i] /= freq_size_lst[i]

    """ df['time'] = df['time'].apply(
        lambda x: x / 1e6
    ) """

    df['access_day'] = df['time'].apply(
        lambda x: int(x % 6.048e5 / 8.64e4))

    df['access_hr'] = df['time'].apply(
        lambda x: int(x % 8.64e4 / 3600))

    df['access_min'] = df['time'].apply(
        lambda x: int(x % 3600 / 60))

    df = df.join(pd.DataFrame(rd_arr, 
        columns=['rd1', 'rd2', 'rd3', 'rd4', 'rd5']))

    df = df.join(pd.DataFrame(freq_arr,
        columns=['freq64', 'freq256', 'freq1024', 'freq4096', 'freq16384']))
    
    df = df.join(pd.DataFrame(req_rate_arr,
        columns=['req_rate64', 'req_rate256', 'req_rate1024']))

    return df

def main():
    
    file_name = sys.argv[1]
    output_add_on = sys.argv[2]
    source_file = 'traces/' + file_name
    target_dest = 'features/' + file_name + '_' + output_add_on + '.csv'

    col_names = ['time', 'id']
    data_types = {'time': 'float64', 'id': 'str'}

    df = pd.read_csv(source_file, sep='\s+', usecols=[0,4], header=0, names=col_names, dtype=data_types
        #,nrows=10000
        )
    
    ta = time.time()
    print('Generating Features for ' + source_file + ':')

    df = generate_features(df, sample=0, hexadecimal=False)
    df.to_csv(target_dest, index=False)

    tb = time.time()
    print('Done. Time Elapsed:')
    print(tb - ta)


if __name__ == '__main__':
    main()