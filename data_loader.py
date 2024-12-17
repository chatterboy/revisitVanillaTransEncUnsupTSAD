import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset


class MSLLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(MSLLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "MSL_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "MSL_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "MSL_test.npy"))
        test_labels = np.load(os.path.join(data_path, "MSL_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))

    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class SMAPLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(SMAPLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "SMAP_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "SMAP_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "SMAP_test.npy"))
        test_labels = np.load(os.path.join(data_path, "SMAP_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))
        
    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class SMDLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(SMDLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "SMD_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "SMD_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "SMD_test.npy"))
        test_labels = np.load(os.path.join(data_path, "SMD_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))
        
    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class PSMLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "PSM_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "PSM_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "PSM_test.npy"))
        test_labels = np.load(os.path.join(data_path, "PSM_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))
        
    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class CICIDSLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(CICIDSLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "CICIDS_train.npy"))
            val = np.load(os.path.join(data_path, "CICIDS_test.npy"))
        else:
            data = np.load(os.path.join(data_path, "CICIDS_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "CICIDS_test.npy"))
        test_label = np.load(os.path.join(data_path, "CICIDS_test_label.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        self.val = val
        self.test = test
        self.test_label = test_label

        self.fake_label = np.zeros((win_size))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_label
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_label
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_label[index:index + self.win_size]

    def get_statistics(self):
        print("train: ", self.train.shape)
        print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_label: ", self.test_label.shape)


class CreditCardLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(CreditCardLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "CreditCard_train.npy"))
            val = np.load(os.path.join(data_path, "CreditCard_test.npy"))
        else:
            data = np.load(os.path.join(data_path, "CreditCard_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "CreditCard_test.npy"))
        test_label = np.load(os.path.join(data_path, "CreditCard_test_label.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        self.val = val
        self.test = test
        self.test_label = test_label

        self.fake_label = np.zeros((win_size))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_label
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_label
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_label[index:index + self.win_size]

    def get_statistics(self):
        print("train: ", self.train.shape)
        print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_label: ", self.test_label.shape)


class SWANSFLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(SWANSFLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "SWAN_SF_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "SWAN_SF_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "SWAN_SF_test.npy"))
        test_labels = np.load(os.path.join(data_path, "SWAN_SF_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))

    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class GECCOLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None):
        super(GECCOLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "GECCO_train.npy"))
        else:
            data = np.load(os.path.join(data_path, "GECCO_train.npy"))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "GECCO_test.npy"))
        test_labels = np.load(os.path.join(data_path, "GECCO_test_labels.npy"))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))

    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]


class UCRLoader(Dataset):
    def __init__(self, data_path, win_size,
                 step=1, split_ratio=None, flag="train",
                 norm=None, data_name=None):
        super(UCRLoader, self).__init__()
        self.win_size = win_size
        self.step = step
        self.flag = flag
        self.split_ratio = split_ratio

        if split_ratio is None:
            train = np.load(os.path.join(data_path, "{}_train.npy".format(data_name)))
        else:
            data = np.load(os.path.join(data_path, "{}_train.npy".format(data_name)))
            train = data[:int(data.shape[0] * split_ratio)]
            val = data[int(data.shape[0] * split_ratio):]
        test = np.load(os.path.join(data_path, "{}_test.npy".format(data_name)))
        test_labels = np.load(os.path.join(data_path, "{}_test_labels.npy").format(data_name))

        if norm == "minmax":
            scaler = MinMaxScaler()
        elif norm == "standard":
            scaler = StandardScaler()
        elif norm is not None:
            raise ValueError("Expected 'minmax'/'standard'/'None', but got {}".format(norm))

        if norm is not None:
            scaler.fit(train)
            train = scaler.transform(train)
            if self.split_ratio is not None:
                val = scaler.transform(val)
            test = scaler.transform(test)

        self.train = train
        if self.split_ratio is not None:
            self.val = val
        self.test = test
        self.test_labels = test_labels

        self.fake_labels = np.zeros((win_size))

    def describe(self):
        print("train: ", self.train.shape)
        if self.split_ratio is not None:
            print("val: ", self.val.shape)
        print("test: ", self.test.shape)
        print("test_labels: ", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return self.train[index:index + self.win_size], self.fake_labels
        elif self.flag == "val":
            return self.val[index:index + self.win_size], self.fake_labels
        elif self.flag == "test":
            return self.test[index:index + self.win_size], self.test_labels[index:index + self.win_size]



if __name__ == "__main__":
    # class MSLLoader(Dataset):
    # def __init__(self, data_path, win_size,
    #              step=1, split_ratio=None, flag='train',
    #              norm=None, norm_level=None):
    
    data_path = "data/MSL"
    win_size = 96
    step = 1
    norm = "standard"
    
    train_loader = MSLLoader(data_path, win_size, step=step, norm=norm)