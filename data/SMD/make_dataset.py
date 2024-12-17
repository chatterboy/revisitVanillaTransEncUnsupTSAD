import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


parser = argparse.ArgumentParser()

parser.add_argument("--norm", type=str, default=None, help="[ \
                     (default) None, minmax, standard]")

args = parser.parse_args()

train_list = []
test_list = []
test_labels_list = []

for fn in os.listdir(os.path.join("train")):
    train_npy = pd.read_csv(os.path.join("train", fn), sep=",", header=None).to_numpy() 
    test_npy = pd.read_csv(os.path.join("test", fn), sep=",", header=None).to_numpy()
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
    test_labels_list.append(
        pd.read_csv(os.path.join("test_label", fn), header=None).to_numpy()
    )

assert len(train_list) == len(test_list) and len(train_list) == len(test_labels_list), "All the lengths of train, test, and test_labels must be same."

train = np.concatenate(train_list, axis=0)
test = np.concatenate(test_list, axis=0)
test_labels = np.concatenate(test_labels_list, axis=None)

print("# sub-datasets {}\ntrain {}\ntest {}\ntest_labels {}\nanomaly ratio in test {}".format(
    len(train_list), train.shape, test.shape, test_labels.shape, np.average(test_labels)
))

np.save("SMD_train.npy", train)
np.save("SMD_test.npy", test)
np.save("SMD_test_labels.npy", test_labels)