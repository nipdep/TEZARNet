import os
import random 
import numpy as np
import pandas as pd
# from scipy.signal import resample
from glob import glob 
import scipy.io 
from sklearn.preprocessing import MinMaxScaler

# build PAMAP2 dataset data reader
class PAMAP2Reader(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readPamap2()

    def readFile(self, file_path, cols):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        action_ID = 0

        for l in open(file_path).readlines():
            s = l.strip().split()
            if s[1] != "0":
                if (prev_action != int(s[1])):
                    if not(starting):
                        df = pd.DataFrame(action_seq)
                        intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                        intep_data = intep_df.values 
                        all_data['data'][action_ID] = np.array(intep_data)[500:-500, :]
                        all_data['target'][action_ID] = prev_action
                        action_ID+=1
                    action_seq = []
                else:
                    starting = False
                intm_data = s[3:]
                data_seq = np.array(intm_data)[cols].astype(np.float16)
                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = int(s[1])
                # print(prev_action)
                all_data['collection'].append(data_seq)
        else: 
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def readPamap2Files(self, filelist, cols, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath, cols)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readPamap2(self):
        files = ['subject101.dat', 'subject102.dat','subject103.dat','subject104.dat', 'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat', 'subject109.dat', 'subject110.dat', 'subject111.dat', 'subject112.dat', 'subject113.dat', 'subject114.dat']
            
        label_map = [
            #(0, 'other'),
            (1, 'lying'),
            (2, 'sitting'),
            (3, 'standing'),
            (4, 'walking'),
            (5, 'running'),
            (6, 'cycling'),
            (7, 'Nordic walking'),
            (9, 'watching TV'),
            (10, 'computer work'),
            (11, 'car driving'),
            (12, 'ascending stairs'),
            (13, 'descending stairs'),
            (16, 'vacuum cleaning'),
            (17, 'ironing'),
            (18, 'folding laundry'),
            (19, 'house cleaning'),
            (20, 'playing soccer'),
            (24, 'rope jumping')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        cols = [1,2,3,7,8,9,10,11,12,17,18,19,23,24,25,26,27,28,33,34,35,39,40,41,42,43,44]
        self.cols = cols
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readPamap2Files(files, cols, labelToId)
        # self.data = self.data[:, :, cols]
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def resample(self, signal, freq=10):
        step_size = int(100/freq)
        seq_len, _ = signal.shape 
        resample_indx = np.arange(0, seq_len, step_size)
        resampled_sig = signal[resample_indx, :]
        return resampled_sig

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len*100) # 100Hz compensation 
        overlap_len = int(overlap*100) # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l-seq_len, step=seq_len-overlap_len, dtype=int)[:-1]

            windows = [signal[p:p+seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            # print(">>>>>>>>>>>>>>>  ", np.isnan(d).mean())
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                # print(np.isnan(w).mean(), label, i)
                resample_sig = self.resample(w, resample_freq)
                # print(np.isnan(resample_sig).mean(), label, i)
                all_data.append(resample_sig)
                all_ids.append(i+1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, window_size=5.21, window_overlap=1, resample_freq=10, smoothing=False, normalize=False, seen_ratio=0.2, unseen_ratio=0.8):
        
        def smooth(x, window_len=11, window='hanning'):
            if x.ndim != 1:
                    raise Exception('smooth only accepts 1 dimension arrays.')
            if x.size < window_len:
                    raise Exception("Input vector needs to be bigger than window size.")
            if window_len<3:
                    return x
            if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                    raise Exception("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
            s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
            if window == 'flat': #moving average
                    w=np.ones(window_len,'d')
            else:  
                    w=eval('np.'+window+'(window_len)')
            y=np.convolve(w/w.sum(),s,mode='same')
            return y[window_len:-window_len+1]

        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)

        # build seen dataset 
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # resampling seen and unseen datasets 
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap, resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size, window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)

        if normalize:
            a, b, nft = seen_data.shape 
            intm_sdata = seen_data.reshape((-1, nft))
            intm_udata = unseen_data.reshape((-1, nft))

            scaler = MinMaxScaler()
            norm_sdata = scaler.fit_transform(intm_sdata)
            norm_udata = scaler.transform(intm_udata)

            seen_data = norm_sdata.reshape(seen_data.shape)
            unseen_data = norm_udata.reshape(unseen_data.shape)

        if smoothing:
            seen_data = np.apply_along_axis(smooth, axis=1, arr=seen_data)
            unseen_data = np.apply_along_axis(smooth, axis=1, arr=unseen_data)
        # train-val split
        seen_index = list(range(len(seen_targets)))
        # random.shuffle(seen_index)
        split_point = int((1-seen_ratio)*len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        # print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val, y_seen_train, y_seen_val = seen_data[fst_index,:], seen_data[sec_index,:], seen_targets[fst_index], seen_targets[sec_index]
        

        data = {'train': {
                        'X': X_seen_train,
                        'y': y_seen_train
                        },
                'eval-seen':{
                        'X': X_seen_val,
                        'y': y_seen_val
                        },
                'test': {
                        'X': unseen_data,
                        'y': unseen_targets
                        },
                'seen_classes': seen_classes,
                'unseen_classes': unseen_classes
                }

        return data
        

class KUHARData(object):
    """KU-HAR dataset implementation"""

    def __init__(self, data_dir, n_proc=1, limit_size=300, config=None, filter_classes=[]):
        self.filter_classes = filter_classes
        self.all_df, self.labels_df = self.load_all(data_dir)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df[self.feature_names]
        self.class_names = self.labels_df.label.unique()
    
    def load_data(self, data_dir):
        df = pd.read_csv(data_dir)
        return df

    def label_shift(self, l):
        i = 0
        if self.filter_classes != []:
            for j in self.filter_classes:
                if l > j:
                    i+=1
        return i

    def load_all(self, data_dir):
        main_df = self.load_data(data_dir)
        subdf_list = []
        label_dict = {'ID': [], 'label': []}

        for i, r in main_df.iterrows():
            r = r.values 
            acx, acy, acz, gyx, gyy, gyz, label, _, ID = r[:300], r[300:600], r[600:900], r[900:1200], r[1200:1500], r[1500:1800], r[1800], r[1801], r[1802]
            if int(label) not in self.filter_classes:
                sub_df = pd.DataFrame({'accelX': acx, 'accelY': acy, 'accelZ': acz, 'GyroX': gyx, 'GyroY': gyy, 'GyroZ': gyz}, index=[int(ID),]*300)
                label = label-self.label_shift(label)
                label_dict['label'].append(int(label))
                label_dict['ID'].append(int(ID))
                subdf_list.append(sub_df)
                label_df = pd.DataFrame(label_dict)
                label_df.set_index('ID', inplace=True)

        full_df = pd.concat(subdf_list)
        # full_df.reset_index(inplace=True, drop=True)
        return full_df, label_df

class UTDReader(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readUTD()
    def dataTableOptimizerUpdated(self, mat_file):
        our_data = mat_file['d_iner']
        data = []
        frame_size = len(our_data[0][0])-1
        for each in range(0,frame_size):
            data_flatten = our_data[:,:,each].flatten()
            data_flatten = data_flatten
            data.append(data_flatten)
        return data,frame_size

    def readUTDFiles(self, labelToId):
        data = []
        labels = []
        subjects = []
        
        for p in glob(f'{self.root_path}/*.mat'):
            file_name = p.split('\\')[-1]
            action, subject, time, _ = file_name.split('_')
            mat = scipy.io.loadmat(p)
            np_data = np.array(mat['d_iner'])
        
            data.append(np_data)
            labels.append(int(action.strip('a')))
            subjects.append(subject)
        # print(f"data len : {np.array(data).shape}, data 0 shape : {data[0].shape}")
        return np.array(data), np.array(labels, dtype=int), np.array(subjects)

    def readUTD(self):
           
        label_map = [
            (1, 'swipe left'),
            (2, 'swipe right'),
            (3, 'wave'),
            (4, 'clap'),
            (5, 'throw'),
            (6, 'cross arms'),
            (7, 'basketball shoot'),
            (8, 'draw x'),
            (9, 'draw circle clockwise'),
            (10, 'draw circle counter clockwise'),
            (11, 'draw triangle'),
            (12, 'bowling'),
            (13, 'boxing'),
            (14, 'baseball swing'),
            (15, 'tennis swing'),
            (16, 'arm curl'),
            (17, 'tennis serve'),
            (18, 'two hand push'),
            (19, 'knock'),
            (20, 'catch'),
            (21, 'pick up then throw'),
            (22, 'jogging in place'),
            (23, 'walking in place'),
            (24, 'sit to stand'),
            (25, 'stand to sit'),
            (26, 'lunge'),
            (27, 'squat')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]

        self.data, self.targets, self.all_data = self.readUTDFiles(labelToId)
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel

    def resample(self, signal, freq=10):
        step_size = int(100/freq)
        seq_len, _ = signal.shape 
        resample_indx = np.arange(0, seq_len, step_size)
        resampled_sig = signal[resample_indx, :]
        return resampled_sig

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len*50) # 100Hz compensation 
        overlap_len = int(overlap*50) # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l-seq_len, step=seq_len-overlap_len, dtype=int)[:-1]
            windows = [signal[p:p+seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                resample_sig = self.resample(w, resample_freq)
                all_data.append(resample_sig)
                all_ids.append(i+1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, window_size=5.21, window_overlap=1, resample_freq=20, seen_ratio=0.2, unseen_ratio=0.8):
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)
        
        # build seen dataset 
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]
        print(f"data shape : {self.data.shape}, seen_data shape : {seen_data.shape}")

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

       # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1-seen_ratio)*len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        # print(fst_index)
        # print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val = [seen_data[i] for i in fst_index], [seen_data[j] for j in sec_index]
        y_seen_train, y_seen_val = seen_targets[fst_index], seen_targets[sec_index]
        
        data = {'train': {
                        'X': X_seen_train,
                        'y': y_seen_train
                        },
                'eval-seen':{
                        'X': X_seen_val,
                        'y': y_seen_val
                        },
                'test': {
                        'X': unseen_data,
                        'y': unseen_targets
                        },
                'seen_classes': seen_classes,
                'unseen_classes': unseen_classes
                }

        return data



class DaLiAcReader(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readDaliac()

    def readFile(self, file_path):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        action_ID = 0

        for l in open(file_path).readlines():
            s = l.rstrip('\n').split(',')
            act = int(s[-1])
            if act == 12:
                act = 11
            elif act == 13:
                act = 12

            if (prev_action != act):
                if not(starting):
                    # df = pd.DataFrame(action_seq)
                    # intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                    # intep_data = intep_df.values 
                    intep_data = action_seq
                    all_data['data'][action_ID] = np.array(intep_data)
                    all_data['target'][action_ID] = prev_action
                    action_ID+=1
                action_seq = []
            else:
                starting = False
            data_seq = np.array(s[:-1]).astype(np.float16)
            # data_seq[np.isnan(data_seq)] = 0
            action_seq.append(data_seq)
            prev_action = act
            
            # print(prev_action)
            all_data['collection'].append(data_seq)
        else: 
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def readDaliAcFiles(self, filelist, labelToId):
        data = []
        labels = []
        collection = []
        for i, filename in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readDaliac(self):
        files = [f'dataset_{i}.txt' for i in range(1, 20)]
            
        label_map = [
            (1, 'sitting'),
            (2, 'lying'),
            (3, 'standing'),
            (4, 'washing dishes'),
            (5, 'vacuuming'),
            (6, 'sweeping'),
            (7, 'walking'),
            (8, 'ascending stairs'),
            (9, 'descending stairs'),
            (10, 'treadmill running'),
            (11, 'cycling'),
            (12, 'rope jumping')
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        idToLabel = [x[1] for x in label_map]
        cols = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
                ]
        self.data, self.targets, self.all_data = self.readDaliAcFiles(files, labelToId)
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel

    def dataTableOptimizerUpdated(self, mat_file):
        our_data = mat_file['d_iner']
        data = []
        frame_size = len(our_data[0][0])-1
        for each in range(0,frame_size):
            data_flatten = our_data[:,:,each].flatten()
            data_flatten = data_flatten
            data.append(data_flatten)
        return data,frame_size

    def resample(self, signal, freq=10):
        step_size = int(100/freq)
        seq_len, _ = signal.shape 
        resample_indx = np.arange(0, seq_len, step_size)
        resampled_sig = signal[resample_indx, :]
        return resampled_sig

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len*50) # 100Hz compensation 
        overlap_len = int(overlap*50) # 100Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l-seq_len, step=seq_len-overlap_len, dtype=int)[:-1]

            windows = [signal[p:p+seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                resample_sig = self.resample(w, resample_freq)
                all_data.append(resample_sig)
                all_ids.append(i+1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, window_size=5.21, window_overlap=1, resample_freq=20, seen_ratio=0.2, unseen_ratio=0.8):
        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)
        
        # build seen dataset 
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]
        # print(f"data shape : {self.data.shape}, seen_data shape : {seen_data.shape}")

        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # # resampling seen and unseen datasets 
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap, resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size, window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)
       # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1-seen_ratio)*len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]

        X_seen_train, X_seen_val = seen_data[fst_index, :], seen_data[sec_index, :]
        y_seen_train, y_seen_val = seen_targets[fst_index], seen_targets[sec_index]
        
        data = {'train': {
                        'X': X_seen_train,
                        'y': y_seen_train
                        },
                'eval-seen':{
                        'X': X_seen_val,
                        'y': y_seen_val
                        },
                'test': {
                        'X': unseen_data,
                        'y': unseen_targets
                        },
                'seen_classes': seen_classes,
                'unseen_classes': unseen_classes
                }
                
        return data

class OPPReader(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.readOPP()

    def readFile(self, file_path, active_cols):
        all_data = {"data": {}, "target": {}, 'collection': []}
        prev_action = -1
        starting = True
        # action_seq = []
        accepted_actions = [406516, 406517, 404516, 404517, 406520, 404520, 406505, 404505, 406519, 404519, 406511, 404511, 406508, 404508, 408512, 407521, 405506]
        action_ID = 0
        cols = list(range(37,133))
        for l in open(file_path).readlines():
            s = l.rstrip('\n').split(' ')
            act = int(s[-1])
            if act in accepted_actions:
                if (prev_action != act):
                    if not(starting):
                        df = pd.DataFrame(action_seq)
                        intep_df = df.interpolate(method='linear', limit_direction='both', axis=0)
                        intep_data = intep_df.values[:, cols]

                        if np.isnan(intep_data).mean() == 0:
                            all_data['data'][action_ID] = np.array(intep_data)
                            # print(all_data['data'][action_ID].shape)
                            all_data['target'][action_ID] = prev_action
                            action_ID+=1
                    action_seq = []
                else:
                    starting = False

                data_seq = np.array(s[:-1]).astype(np.float16)                # data_seq[np.isnan(data_seq)] = 0
                action_seq.append(data_seq)
                prev_action = act
                
                # print(prev_action)
                all_data['collection'].append(data_seq)
        else: 
            if len(action_seq) > 1:
                df = pd.DataFrame(action_seq)
                intep_df = df.interpolate(method='linear', limit_direction='backward', axis=0)
                intep_data = intep_df.values[:, cols]
                all_data['data'][action_ID] = np.array(intep_data)
                all_data['target'][action_ID] = prev_action
        return all_data

    def readOPPFiles(self, filelist, labelToId):
        data = []
        labels = []
        collection = []
        accepted_cols = list(range(1, 101))
        for i, fpath in enumerate(filelist):
            print('Reading file %d of %d' % (i+1, len(filelist)))
            # fpath = os.path.join(self.root_path, filename)
            file_data = self.readFile(fpath, accepted_cols)
            # print(np.array(list(file_data['data'].values())).shape)
            data.extend(list(file_data['data'].values()))
            labels.extend(list(file_data['target'].values()))
            collection.extend(file_data['collection'])
        return np.asarray(data), np.asarray(labels, dtype=int), np.array(collection)

    def readOPP(self):
        files = []
        for p in glob(f'{self.root_path}/S*-ADL*.dat'):
            files.append(p)
            # files = [f'dataset_{i}.txt' for i in range(1, 20)]
            
        label_map = [
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch'),
        ]
        labelToId = {x[0]: i for i, x in enumerate(label_map)}
        # print "label2id=",labelToId
        idToLabel = [x[1] for x in label_map]
        # print "id2label=",idToLabel
        # print "cols",cols
        self.data, self.targets, self.all_data = self.readOPPFiles(files, labelToId)
        # print(self.data)
        # nan_perc = np.isnan(self.data).astype(int).mean()
        # print("null value percentage ", nan_perc)
        # f = lambda x: labelToId[x]
        self.targets = np.array([labelToId[i] for i in list(self.targets)])
        self.label_map = label_map
        self.idToLabel = idToLabel
        # return data, idToLabel

    def dataTableOptimizerUpdated(self, mat_file):
        our_data = mat_file['d_iner']
        data = []
        frame_size = len(our_data[0][0])-1
        for each in range(0,frame_size):
            data_flatten = our_data[:,:,each].flatten()
            data_flatten = data_flatten
            data.append(data_flatten)
        return data,frame_size

    def resample(self, signal, freq=10):
        step_size = int(30/freq)
        seq_len, _ = signal.shape 
        resample_indx = np.arange(0, seq_len, step_size)
        resampled_sig = signal[resample_indx, :]
        return resampled_sig

    def windowing(self, signal, window_len, overlap):
        seq_len = int(window_len*30) # 30Hz compensation 
        overlap_len = int(overlap*30) # 30Hz
        l, _ = signal.shape
        if l > seq_len:
            windowing_points = np.arange(start=0, stop=l-seq_len, step=seq_len-overlap_len, dtype=int)[:-1]

            windows = [signal[p:p+seq_len, :] for p in windowing_points]
        else:
            windows = []
        return windows

    def resampling(self, data, targets, window_size, window_overlap, resample_freq):
        assert len(data) == len(targets), "# action data & # action labels are not matching"
        all_data, all_ids, all_labels = [], [], []
        for i, d in enumerate(data):
            # print(">>>>>>>>>>>>>>>  ", np.isnan(d).mean())
            label = targets[i]
            windows = self.windowing(d, window_size, window_overlap)
            for w in windows:
                # print(np.isnan(w).mean(), label, i)
                resample_sig = w#self.resample(w, resample_freq)
                # print(np.isnan(resample_sig).mean(), label, i)
                all_data.append(resample_sig)
                all_ids.append(i+1)
                all_labels.append(label)

        return all_data, all_ids, all_labels

    def generate(self, unseen_classes, window_size=5.21, window_overlap=1, resample_freq=20, seen_ratio=0.2, unseen_ratio=0.8):
        # assert all([i in list(self.label_map.keys()) for i in unseen_classes]), "Unknown Class label!"
        seen_classes = [i for i in range(len(self.idToLabel)) if i not in unseen_classes]
        unseen_mask = np.in1d(self.targets, unseen_classes)
        
        # build seen dataset 
        seen_data = self.data[np.invert(unseen_mask)]
        seen_targets = self.targets[np.invert(unseen_mask)]
        print(f"data shape : {self.data.shape}, seen_data shape : {seen_data.shape}")
        ids, cnts = np.unique(self.targets, return_counts=True)
        print({self.idToLabel[ids[e]]: cnts[e] for e in range(len(ids))})
        
        # build unseen dataset
        unseen_data = self.data[unseen_mask]
        unseen_targets = self.targets[unseen_mask]

        # # resampling seen and unseen datasets 
        seen_data, seen_ids, seen_targets = self.resampling(seen_data, seen_targets, window_size, window_overlap, resample_freq)
        unseen_data, unseen_ids, unseen_targets = self.resampling(unseen_data, unseen_targets, window_size, window_overlap, resample_freq)

        seen_data, seen_targets = np.array(seen_data), np.array(seen_targets)
        unseen_data, unseen_targets = np.array(unseen_data), np.array(unseen_targets)
       # train-val split
        seen_index = list(range(len(seen_targets)))
        random.shuffle(seen_index)
        split_point = int((1-seen_ratio)*len(seen_index))
        fst_index, sec_index = seen_index[:split_point], seen_index[split_point:]
        # print(fst_index)
        # print(type(fst_index), type(sec_index), type(seen_data), type(seen_targets))
        X_seen_train, X_seen_val = seen_data[fst_index, :], seen_data[sec_index, :]
        y_seen_train, y_seen_val = seen_targets[fst_index], seen_targets[sec_index]
        
    
        data = {'train': {
                        'X': X_seen_train,
                        'y': y_seen_train
                        },
                'eval-seen':{
                        'X': X_seen_val,
                        'y': y_seen_val
                        },
                'test': {
                        'X': unseen_data,
                        'y': unseen_targets
                        },
                'seen_classes': seen_classes,
                'unseen_classes': unseen_classes
                }

        return data