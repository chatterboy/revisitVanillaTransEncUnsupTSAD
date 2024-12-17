import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, required=True, help="[ABP, \
                     Acceleration, AirTemperature, ECG, \
                     EPG, Gait, NASA, PowerDemand, RESP]")
parser.add_argument("--norm", type=str, default=None, help="[ \
                     (default) None, minmax, standard]")

args = parser.parse_args()

DATASETS_DIR_NAME = "UCR_TimeSeriesAnomalyDatasets2021_splitted"
SUBDATASETS_DIR_NAME = args.dir
DIFFERENT_FORMAT_FILE_NAME_LIST = [
    204, 205, 206, 207, 208,
    225, 226, 242, 243
]

train_list = []
test_list = []
test_labels_list = []

for fname in os.listdir(os.path.join(DATASETS_DIR_NAME, SUBDATASETS_DIR_NAME)):
    if int(fname[:3]) in DIFFERENT_FORMAT_FILE_NAME_LIST:
        with open(os.path.join(DATASETS_DIR_NAME, SUBDATASETS_DIR_NAME, fname)) as f:
            l = f.readlines()
            l = l[0].strip()
            df = pd.DataFrame(l.split(), dtype=float)
    else:
        df = pd.read_csv(os.path.join(DATASETS_DIR_NAME, SUBDATASETS_DIR_NAME, fname),
                         header=None)
    
    data = df.to_numpy()

    token_list = fname.split(".")[0].split("_")
    train_length = int(token_list[4])
    anom_start = int(token_list[5]) - 1
    anom_end = int(token_list[6]) - 1

    data_labels = np.zeros((data.shape[0],), dtype=int)
    data_labels[anom_start:anom_end + 1] = 1

    train = data[:train_length]
    test = data[train_length:]
    test_labels = data_labels[train_length:]

    train_list.append(train)
    test_list.append(test)
    test_labels_list.append(test_labels)

for i in range(len(train_list)):
    train_df = pd.DataFrame(train_list[i])
    test_df = pd.DataFrame(test_list[i])
    print("{}-th sub-dataset\n".format(i + 1))
    print(train_df.describe())
    print(test_df.describe())
    print()

if args.norm is not None:
    if args.norm == "minmax":
        scaler = MinMaxScaler()
        # scaler = MinMaxScaler(feature_range=(-1, 1))  # TODO
    elif args.norm == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError
    for i in range(len(train_list)):
        scaler.fit(train_list[i])
        train_list[i] = scaler.transform(train_list[i])
        test_list[i] = scaler.transform(test_list[i])

train = np.concatenate(train_list, axis=0)
test = np.concatenate(test_list, axis=0)
test_labels = np.concatenate(test_labels_list)

print("dataset name: {}".format(SUBDATASETS_DIR_NAME))
print("train.shape: {}".format(train.shape))
print("test.shape: {}".format(test.shape))
print("test_labels.shape: {}".format(test_labels.shape))
print("# anomalies: {}".format(np.sum(test_labels)))
print("anomaly ratio: {}".format(np.mean(test_labels)))

np.save("{}_train.npy".format(SUBDATASETS_DIR_NAME), train)
np.save("{}_test.npy".format(SUBDATASETS_DIR_NAME), test)
np.save("{}_test_labels.npy".format(SUBDATASETS_DIR_NAME), test_labels)