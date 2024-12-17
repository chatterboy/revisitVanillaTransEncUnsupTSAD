import os
import ast
import pandas as pd
import numpy as np


TEST_LABELS_INFO = "labeled_anomalies.csv"
TEST_LABELS = "test_labels"

test_labels_info = pd.read_csv(TEST_LABELS_INFO)

for chan_id in set(test_labels_info.loc[test_labels_info["spacecraft"] == "SMAP", "chan_id"].tolist()):
    df = test_labels_info[test_labels_info["chan_id"] == chan_id]

    if len(df) > 1:  # e.g. the P-2 channel
        keys = df["num_values"].keys().tolist()
        T = df["num_values"][keys[0]].item()
    else:
        T = df["num_values"].item()

    test_labels = np.zeros(T)

    if len(df) > 1:  # e.g. the P-2 channel
        anomalies = []
        for k in keys:
            anomalies += ast.literal_eval(df["anomaly_sequences"][k])
    else:
        anomalies = ast.literal_eval(df["anomaly_sequences"].item())

    print("channel id {} length {} anomalies {}".format(
        chan_id, test_labels.shape, anomalies
    ))

    for s, e in anomalies:
        test_labels[s:e + 1] = 1

    if not os.path.isdir(TEST_LABELS):
        os.mkdir(TEST_LABELS)

    np.save(os.path.join(TEST_LABELS, "{}.npy".format(chan_id)), test_labels)