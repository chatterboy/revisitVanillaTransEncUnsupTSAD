import pandas as pd
import numpy as np


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_labels = pd.read_csv("test_label.csv")

for k, v in zip(train.isna().sum().keys().tolist(),
                train.isna().sum().tolist()):
    if v > 0:
        min_val = train[k].describe()["min"]
        train.loc[train[k].isna(), k] = min_val

train = train.drop(columns=["timestamp_(min)"])
test = test.drop(columns=["timestamp_(min)"])
test_labels = test_labels.drop(columns=["timestamp_(min)"])

train = train.to_numpy()
test = test.to_numpy()
test_labels = test_labels.to_numpy()

test_labels = np.squeeze(test_labels, 1)

print("train {}\ntest {}\ntest_labels {}\nanomaly ratio in test {}".format(
    train.shape, test.shape, test_labels.shape, np.average(test_labels)
))

np.save("PSM_train.npy", train)
np.save("PSM_test.npy", test)
np.save("PSM_test_labels.npy", test_labels)