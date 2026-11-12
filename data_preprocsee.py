# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wfdb as wf
import glob
from biosppy.signals import ecg
from wfdb import processing
import scipy
from scipy import *
from os import path


def extract_data():
    data_files = glob.glob('./mitbih/*.atr')
    data_files = [i[:-4] for i in data_files]
    data_files.sort()
    return data_files

length = 275
records = extract_data()
print('Total files: ', len(records))

good_beats = ['N','L','R','B','A','a','J','S','V','r',
             'F','e','j','n','E','/','f','Q','?']

for file_path in records:
    file_pathpts = file_path.split('/')
    fn = file_pathpts[-1]
    print('Loading file:', file_path)

    # Read in the data
    if path.exists(file_path+".hea"):
        record = wf.rdsamp(file_path)
        annotation = wf.rdann(file_path, 'atr')

        # Print some meta informations
        print('    Sampling frequency used for this record:', record[1].get('fs'))
        print('    Shape of loaded data array:', record[0].shape)
        print('    Number of loaded annotations:', len(annotation.num))

        # Get the ECG values from the file.
        data = record[0].transpose()

        clas = np.array(annotation.symbol)
        rate = np.zeros_like(clas, dtype='float')
        for clasid, clasval in enumerate(clas):
            if (clasval == 'N'):
                rate[clasid] = 1.0 # Normal
            elif (clasval == 'L'):
                rate[clasid] = 2.0 # LBBBB
            elif (clasval == 'R'):
                rate[clasid] = 3.0 # RBBBB
            elif (clasval == 'V'):
                rate[clasid] = 4.0 # Premature Ventricular contraction
            elif (clasval == 'A'):
                rate[clasid] = 5.0 # Atrial Premature beat
            elif (clasval == 'F'):
                rate[clasid] = 6.0 # Fusion ventricular normal beat
            elif (clasval == 'f'):
                rate[clasid] = 7.0 # Fusion of paced and normal beat
            elif (clasval == '/'):
                rate[clasid] = 8.0 # paced beat
            
        rates = np.zeros_like(data[0], dtype='float')
        rates[annotation.sample] = rate

        indices = np.arange(data[0].size, dtype='int')

        # Manipulate both channels
        for channelid, channel in enumerate(data):
            chname = record[1].get('sig_name')[channelid]
            print('    ECG channel type:', chname)

            # Find rpeaks in the ECG data. Most should match with
            # the annotations.
            out = ecg.ecg(signal=channel, sampling_rate=360, show=False)

            # Split into individual heartbeats. For each heartbeat
            # record, append classification.

            beats = []
            for ind, ind_val in enumerate(out['rpeaks']):

                start,end = ind_val-length//2, ind_val+length//2
                if start < 0:
                    start = 0
                diff = length - len(channel[start:end])
                if diff > 0:
                    padding = np.zeros(diff, dtype='float')
                    padded_channel = np.append(padding, channel[start:end])
                    beats.append(padded_channel)
                else:
                    beats.append(channel[start:end])

                # Get the classification value that is on
                # or near the position of the rpeak index.
                from_ind = 0 if ind_val < 10 else ind_val - 10
                to_ind = ind_val + 10
                clasval = rates[from_ind:to_ind].max()

                # Standardize the data
                beats[ind] = ((beats[ind] - np.mean(beats[ind])) / np.std(beats[ind]))

                # Append the classification to the beat data.
                beats[ind] = np.append(beats[ind], clasval)

                # Append the record number to the beat data.
                beats[ind] = np.append(beats[ind], fn[-3:])

            # Save to CSV file.
            #savedata = np.array(beats[:], dtype=np.float)
            savedata = np.array(beats[:], dtype=float)
            outfn = './'+fn+'_'+chname+'.csv'
            print('Generating ', outfn)
            with open(outfn, "wb") as fin:
                np.savetxt(fin, savedata, delimiter=",", fmt='%f')
    else:
        print('    ECG '+file_path+' not complete, skipping')






fig, ax = plt.subplots(2, 1, figsize = (15, 6))
ax[0].plot(data[0, 0:2000])
ax[0].set_ylabel('MLII/mV')
ax[1].plot(data[1, 0:2000])
ax[1].set_ylabel('V5/mV')
ax[1].set_xlabel('time/sample')
ax[0].set_title('Record 234')
fig.savefig('ecg_data_unpreprocessed.jpg', dpi = 400)


# Combine all the pre-processed data:
new_len = 277
alldata = np.empty(shape=[0, new_len])
all_csv = glob.glob('./mitbih/*.csv')
for j in all_csv:
    csvrows = np.loadtxt(j, delimiter=',')
    alldata = np.append(alldata, csvrows, axis=0)
print(alldata.shape)

# Visualize the pre-processed data: 
fig, ax = plt.subplots(3, 3, figsize = (18, 7))
categories = ['N', 'L', 'R', 'V', 'A', 'F', 'f', '/']
for i in range(len(categories)):
    indices_category = np.argwhere(alldata[:, -2] == i + 1)
    ax[i // 3, i % 3].plot(np.arange(275), alldata[indices_category[5], :-2][0])
    ax[i // 3, i % 3].set_title('Class: ' + str(categories[i]) + ', Number: ' + str(i + 1))
ax[2, 2].axis('off')
plt.tight_layout()
fig.savefig('ecg_data_preprocessed.jpg', dpi = 400)


# Remove unidentified beats and save all data:
no_anno = np.where(alldata[:,-2]==0.0)[0]
print(no_anno.shape)
alldata = np.delete(alldata, no_anno,0)
print(alldata.shape)

with open('./all_data.csv', "wb") as fin:
    np.savetxt(fin, alldata, delimiter=",", fmt='%f')



files = extract_data()
i = 0

good_beats =['N','L','R','B','A','a','J','S','V','r',
             'F','e','j','n','E','/','f','Q','?']
list_anns = []

for i in range(len(files)):
    datfile = files[i]
    
    if path.exists(datfile+".hea"):
        record = wf.rdsamp(datfile)
        ann = wf.rdann(datfile, 'atr')
        list_anns.extend(ann.symbol)

dict_anns = {}

for i in list_anns:
    dict_anns[i] = dict_anns.get(i,0)+1
    
dict_anns = {k:v for k,v in dict_anns.items() if k in good_beats}

print(dict_anns.values())

fig = plt.figure(figsize=(12,6))

xlocs, xlabs = plt.xticks()
barplt = plt.bar(list(dict_anns.keys()),dict_anns.values(),width = .6)
xlocs = [j for j in dict_anns.keys()]
ylabs = [j for j in dict_anns.values()]

plt.title('Annotation Distribution for all Classes in the Dataset')
plt.xlabel('Annotations')
plt.ylabel('Counts')

for bar in barplt:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval+1000, yval)
    
plt.show()
fig.savefig('ecg_data_beats_per_class.jpg', dpi = 400)