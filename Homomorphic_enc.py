import pandas as pd
import numpy as np
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import time

pd.options.display.float_format = "{:,.4f}".format

gun_violence_path = 'gun-violence-data_01-2013_03-2018.csv'

impt_columns = [
    'incident_id',
    'date',
    'state',
    'n_killed',
    'n_injured'
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
    
    for index, rows in df.iterrows():
        df.loc[index ,'n_killed'] = HE.encryptFrac(df.loc[index ,'n_killed'])
        df.loc[index, 'n_injured'] = HE.encryptFrac(df.loc[index ,'n_injured'])

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
    df = df.loc[(df['date'].str.contains('2017')) | (df['date'].str.contains('2016')), impt_columns].copy()

    n_samples_test = []
    i = len(df)
    while i > 20:
        n_samples_test.append(i)
        i = round(i/2)

    operation_times = []

    HE = create_pyfhel_obj()
    [df_enc, encrypt_time] = encrypt_dataset(HE, df)
    print('Encryption time: {:.4f} seconds'.format(encrypt_time))

    for num in n_samples_test:
        print('Number of Samples: {:d}'.format(num))
        df_test = df_enc[:num].copy()
        states = df_test['state'].unique().tolist()
        [stats, ops_time] = homomorphic_ops(HE, df_test)

        frame = {
            'state': states,
            'n_incidents':[HE.decryptInt(stats[i][0]) for i in range(len(stats))],
            'n_killed':[HE.decryptFrac(stats[i][1]) for i in range(len(stats))],
            'killed_mean':[HE.decryptFrac(stats[i][2]) for i in range(len(stats))],
            'n_injured':[HE.decryptFrac(stats[i][3]) for i in range(len(stats))],
            'injured_mean':[HE.decryptFrac(stats[i][4]) for i in range(len(stats))]
        }

        result_df = pd.DataFrame(frame)
        result_df.to_csv('df'+str(num)+'.csv')

        print('Done! Next please!\n')

        operation_times.append(ops_time)

    print('All done!')

    frame = {
        'n_samples': n_samples_test,
        'operation_times': operation_times
    }

    times_df = pd.DataFrame(frame)
    times_df.to_csv('times_df.csv')
