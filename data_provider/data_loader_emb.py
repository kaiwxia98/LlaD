import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
import h5py
from utils.polynomial import get_pca_base

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None, 
                 features='M', data_path='ETTh1', num_nodes=7,
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h',
                 model_name="gpt2", external_scaler=None, **kwargs): 
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.freq = freq
        self.timeenc = timeenc
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag

        # Append '.csv' if not present
        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv' 
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.model_name = model_name
        self.external_scaler = external_scaler   
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.external_scaler is not None:        
                self.scaler = self.external_scaler
                data = self.scaler.transform(df_data.values)
                print(f"[{self.flag}] Using EXTERNAL scaler (type: {type(self.scaler).__name__})")
            else:                                       # 正常训练：自己 fit
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
                print(f"[{self.flag}] FIT new scaler")
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    # __getitem__、__len__、inverse_transform 完全不变
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        embeddings = None
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour_PCA(Dataset_ETT_hour):
    def __init__(
        self, rank_ratio=1.0, pca_dim="T", reinit=1, *args,  **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pca_fit(rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, pca_dim="T", reinit=1):
        if self.set_type != 0:
            # Note: we only apply PCA transformation on train data
            self.pca_components = None
            return

        print("Fitting PCA ...")
        label_seq = []
        for i in range(self.__len__()):
            _, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            label_seq.append(label)
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        # Note: get pca projection basis for pytorch based projection
        self.pca_components, self.initializer, self.weights = get_pca_base(
            label_seq, rank_ratio, pca_dim, reinit
        )
        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None, 
                 features='M', data_path='ETTm1', model_name="gpt2",
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None,
                 external_scaler=None, **kwargs):  
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv' 
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.model_name = model_name
        self.external_scaler = external_scaler   
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            if self.external_scaler is not None:        
                self.scaler = self.external_scaler
                data = self.scaler.transform(df_data.values)
            else:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        embeddings = None
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute_PCA(Dataset_ETT_minute):
    def __init__(
        self, rank_ratio=1.0, pca_dim="T", reinit=1, *args,  **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pca_fit(rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, pca_dim="T", reinit=1):
        if self.set_type != 0:
            # Note: we only apply PCA transformation on train data
            self.pca_components = None
            return

        print("Fitting PCA ...")
        label_seq = []
        for i in range(self.__len__()):
            _, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            label_seq.append(label)
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        # Note: get pca projection basis for pytorch based projection
        self.pca_components, self.initializer, self.weights = get_pca_base(
            label_seq, rank_ratio, pca_dim, reinit
        )
        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_Custom(Dataset):
    def __init__(self, root_path="./dataset/", flag='train', size=None,
                 features='M', data_path='ECL',
                 target='OT', scale=True, timeenc=0, freq='h',
                 patch_len=16,percent=100,model_name="gpt2",external_scaler=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        # info
        self.percent = percent
        self.patch_len = patch_len
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv' 
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.model_name = model_name
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom_PCA(Dataset_Custom):
    def __init__(
        self, rank_ratio=1.0, pca_dim="T", reinit=1, *args,  **kwargs
    ):
        data_path = kwargs.get('data_path', '')
        if data_path == 'ILI':
            kwargs['freq'] = 'w'
        elif data_path == 'exchange_rate':
            kwargs['freq'] = 'd'
        super().__init__(*args, **kwargs)
        self.pca_fit(rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, pca_dim="T", reinit=1):
        if self.set_type != 0:
            self.pca_components = None
            return
        label_seq = []
        for i in range(self.__len__()):
            _, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            label_seq.append(label)
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.pca_components, self.initializer, self.weights = get_pca_base(
            label_seq, rank_ratio, pca_dim, reinit
        )


class Dataset_PEMS(Dataset):
    def __init__(self, root_path="./dataset/",
                 flag='train', size=None,
                 features='M', data_path='PEMS03_data.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 percent=100, model_name="gpt2", num_nodes=358,external_scaler=None, **kwargs):
        # size [seq_len, label_len, pred_len]
        self.percent = percent
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.num_nodes = num_nodes

        self.root_path = root_path
        self.data_path = data_path
        if not data_path.endswith('.csv'):
            data_path_file = data_path
            data_path += '.csv'
        else:
            data_path_file = data_path.replace('.csv', '')
        self.data_path = os.path.join(root_path, data_path)
        self.data_path_file = data_path_file

        self.model_name = model_name
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 尝试读取，若无表头则 header=None
        try:
            df_raw = pd.read_csv(self.data_path)
        except:
            df_raw = pd.read_csv(self.data_path, header=None)

        # 如果第一列是时间戳（不常见），需跳过，这里假设全部是数值
        data = df_raw.values.astype(np.float32)  # shape: [T, N]

        # 按 60% / 20% / 20% 划分
        num_train = int(len(data) * 0.6)
        num_test = int(len(data) * 0.2)
        num_vali = len(data) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 训练集可按百分比截取
        if self.set_type == 0:
            border2 = int((border2 - self.seq_len) * self.percent / 100) + self.seq_len

        # 标准化：仅使用训练集部分计算均值和方差
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = data

        # 生成时间特征（因为没有真实日期列，使用等间隔时间戳模拟）
        if self.timeenc == 0:
            # 生成从 0 到 T-1 的时间戳，频率为 freq（例如 't' 表示5分钟）
            date_range = pd.date_range(start='2012-01-01', periods=len(data), freq=self.freq)
            df_stamp = pd.DataFrame()
            df_stamp['date'] = date_range
            df_stamp['year'] = df_stamp.date.dt.year
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            df_stamp['minute'] = df_stamp.date.dt.minute
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            date_range = pd.date_range(start='2012-01-01', periods=len(data), freq=self.freq)
            data_stamp = time_features(date_range, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_PEMS_PCA(Dataset_PEMS):
    def __init__(self, rank_ratio=1.0, pca_dim="T", reinit=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_fit(rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, pca_dim="T", reinit=1):
        if self.set_type != 0:
            self.pca_components = None
            return

        print("Fitting PCA ...")
        label_seq = []
        for i in range(self.__len__()):
            _, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            label_seq.append(label)
        label_seq = np.array(label_seq)
        from utils.polynomial import get_pca_base
        self.pca_components, self.initializer, self.weights = get_pca_base(
            label_seq, rank_ratio, pca_dim, reinit
        )
        print(f"PCA components shape: {self.pca_components.shape}")
