import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TEST_LABELS_INFO = "labeled_anomalies.csv"
DATA = "data"
TRAIN = "train"
TEST = "test"
TEST_LABELS = "test_labels"

parser = argparse.ArgumentParser()

parser.add_argument("--norm", type=str, default=None, help="[ \
                     (default) None, minmax, standard]")

args = parser.parse_args()

test_labels_info = pd.read_csv(TEST_LABELS_INFO)

train_list = []
test_list = []
test_labels_list = []

for chan_id in set(test_labels_info.loc[test_labels_info["spacecraft"] == "SMAP", "chan_id"].tolist()):
    train_npy = np.load(os.path.join(DATA, TRAIN, "{}.npy".format(chan_id)))
    test_npy = np.load(os.path.join(DATA, TEST, "{}.npy".format(chan_id)))
    if args.norm is not None:
        if args.norm == "minmax":
            scaler = MinMaxScaler()
        elif args.norm == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError
        scaler.fit(train_npy)
        train_npy = scaler.transform(train_npy)
        test_npy = scaler.transform(test_npy)
    train_list.append(train_npy)
    test_list.append(test_npy)
    test_labels_list.append(np.load(os.path.join(TEST_LABELS, "{}.npy".format(chan_id))))

assert len(train_list) == len(test_list) and len(train_list) == len(test_labels_list), "All the lengths of train, test, and test_labels must be same."

train = np.concatenate(train_list, axis=0)
test = np.concatenate(test_list, axis=0)
test_labels = np.concatenate(test_labels_list, axis=None)

print("# sub-datasets {}\ntrain {}\ntest {}\ntest_labels {}\nanomaly ratio in test {}".format(
    len(train_list), train.shape, test.shape, test_labels.shape, np.average(test_labels)
))

np.save("SMAP_train.npy", train)
np.save("SMAP_test.npy", test)
np.save("SMAP_test_labels.npy", test_labels)