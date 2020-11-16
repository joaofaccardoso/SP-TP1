import pandas as pd
import numpy as np
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import time

pd.options.display.float_format = "{:,.4f}".format

gun_violence_path = 'Datasets/gun_violence/gun-violence-data_01-2013_03-2018.csv'

impt_columns = [
    'incident_id',
    'date',
    'state',
    'city_or_county',
    'n_killed',
    'n_injured'
]

n_samples_test = [
    1000,
    2000,
    4000,
    8000,
    16000,
    32000
]


def read_file():
    return pd.read_csv(gun_violence_path)

def create_pyfhel_obj(ctxt_val = 65537):
    print('Creating Context and KeyGen in a Pyfhel object.')

    HE = Pyfhel()
    HE.contextGen(ctxt_val)

    HE.keyGen()

    return HE
    
def encrypt_dataset(HE, df):
    print('Encrypting Dataset.\n')

    init_time = time.time()
    
    df['n_killed'] = list(map(lambda x : HE.encryptFrac(x),df['n_killed']))
    df['n_injured'] = list(map(lambda x : HE.encryptFrac(x),df['n_injured']))

    final_time = time.time() - init_time

    return [df, final_time]

def homomorphic_ops(HE, df):
    print('Starting Homomorphic Operations.')
    states = df['state'].unique().tolist()
    by_state_stats = []

    init_time = time.time()

    for state in states:
        state_info = df.loc[df['state'] == state].copy()
        
        n_incidents = HE.encryptFrac(float(1/len(state_info)))
        
        killed_sum = np.sum(state_info['n_killed'])
        killed_mean = killed_sum * n_incidents
        
        injured_sum = np.sum(state_info['n_injured'])
        injured_mean = injured_sum * n_incidents
        
        n_incidents = HE.encryptInt(len(state_info))
        
        by_state_stats.append([n_incidents, killed_sum, killed_mean, injured_sum, injured_mean])

    final_time = time.time() - init_time
    
    return [by_state_stats, final_time]



if __name__ == "__main__":
    df = read_file()
    df = df.loc[df['date'].str.contains('2017'), impt_columns].copy()

    n_samples_test.append(len(df))
    operation_times = []

    HE = create_pyfhel_obj()
    [df_test, encrypt_time] = encrypt_dataset(HE, df)

    for num in n_samples_test:
        print('Number of Samples: {:d}'.format(num))
        [stats, ops_time] = homomorphic_ops(HE, df_test)
        print('Done, next please!\n')

        operation_times.append(operation_times)

    frame = {
        'n_samples': n_samples_test,
        'operations_time': operation_times
    }

    times_df = pd.DataFrame(frame)
    print(encrypt_time)
    print(n_samples_test)
    print(operation_times)
