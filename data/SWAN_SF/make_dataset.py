import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DATA_NAME = "swan_sf.csv"
SPLIT_RATIO = 0.5
PPRINT = 5

data = pd.read_csv(DATA_NAME)

train_df = data.iloc[:int(len(data) * SPLIT_RATIO)]
test_df = data.iloc[int(len(data) * SPLIT_RATIO):]

train_df = train_df.drop(columns=["label"])

test_labels_df = test_df[test_df.columns[-1]]
test_df = test_df[test_df.columns[:-1]]

cols_train = train_df.columns.tolist()
cols_test = test_df.columns.tolist()

# PPRINT
print("test_labels_df.describe\n")
print(test_labels_df.describe())
print()
cols_train_df = train_df.columns.tolist()
cols_test_df = test_df.columns.tolist()
print("train_df.describe\n")
for i in range(len(cols_train_df) // PPRINT):
    print(train_df.describe()[cols_train_df[i * PPRINT:(i + 1) * PPRINT]])
print(train_df.describe()[cols_train_df[(len(cols_train_df) // PPRINT) * PPRINT:len(cols_train_df)]])
print()
print("test_df.describe\n")
for i in range(len(cols_test_df) // PPRINT):
    print(test_df.describe()[cols_test_df[i * PPRINT:(i + 1) * PPRINT]])
print(test_df.describe()[cols_test_df[(len(cols_test_df) // PPRINT) * PPRINT:len(cols_test_df)]])
print()
print()

train = train_df.to_numpy()
test = test_df.to_numpy()

# Preprocessing...
# scaler = MinMaxScaler()
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)
for j in range(train.shape[-1]):
    if j in [31, 32, 33, 34, 35]:
    # if j in [31, 32, 33, 34, 35, 37]:
        continue
    scaler = StandardScaler()
    scaler.fit(train[:, j:j + 1])
    train[:, j:j + 1] = scaler.transform(train[:, j:j + 1])
    test[:, j:j + 1] = scaler.transform(test[:, j:j + 1])

train_df = pd.DataFrame(train, columns=cols_train)
test_df = pd.DataFrame(test, columns=cols_test)

# PPRINT
print("test_labels_df.describe\n")
print(test_labels_df.describe())
print()
cols_train_df = train_df.columns.tolist()
cols_test_df = test_df.columns.tolist()
print("train_df.describe\n")
for i in range(len(cols_train_df) // PPRINT):
    print(train_df.describe()[cols_train_df[i * PPRINT:(i + 1) * PPRINT]])
print(train_df.describe()[cols_train_df[(len(cols_train_df) // PPRINT) * PPRINT:len(cols_train_df)]])
print()
print("test_df.describe\n")
for i in range(len(cols_test_df) // PPRINT):
    print(test_df.describe()[cols_test_df[i * PPRINT:(i + 1) * PPRINT]])
print(test_df.describe()[cols_test_df[(len(cols_test_df) // PPRINT) * PPRINT:len(cols_test_df)]])
print()
print()

train = train_df.to_numpy()
test = test_df.to_numpy()
test_labels = test_labels_df.to_numpy()

print("train {} and test {} and test_label {} and anomaly ratio in test {}".format(
    train.shape, test.shape, test_labels.shape, np.average(test_labels)
))

np.save("SWAN_SF_train.npy", train)
np.save("SWAN_SF_test.npy", test)
np.save("SWAN_SF_test_labels.npy", test_labels)