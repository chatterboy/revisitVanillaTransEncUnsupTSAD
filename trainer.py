import os

import numpy as np
from sklearn.metrics import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from utils import *

from data_loader import *
from models import autoencoder

from metrics.metrics import *


MODEL_DICT = {
    "Autoencoder": autoencoder.Model
}

DATA_DICT = {
    "ABP": UCRLoader,
    "Acceleration": UCRLoader,
    "AirTemperature": UCRLoader,
    "ECG": UCRLoader,
    "EPG": UCRLoader,
    "Gait": UCRLoader,
    "NASA": UCRLoader,
    "PowerDemand": UCRLoader,
    "RESP": UCRLoader,
    "MSL": MSLLoader,
    "SMAP": SMAPLoader,
    "SMD": SMDLoader,
    "PSM": PSMLoader,
    "SWAN_SF": SWANSFLoader,
    "GECCO": GECCOLoader
}

UCR_DATA_NAME = [
    "ABP",  "Acceleration", "AirTemperature",
    "ECG",  "EPG",          "Gait",
    "NASA", "PowerDemand",  "RESP"
]


# TODO:
def get_results(args):
    queue, percentile, scores_for_th, scores, gts = args

    # th = np.percentile(scores_for_th, 100 - anomaly_ratio)
    th = np.percentile(scores_for_th, percentile)

    th_scores = (scores > th).astype(int)

    # PA
    gts, th_scores = point_adjustment(gts, th_scores)

    # P, R, F1
    acc = metrics.accuracy_score(gts, th_scores)
    prec, recall, f1_score, _ = metrics.precision_recall_fscore_support(gts, th_scores, average='binary')

    scores_simple = combine_all_evaluation_scores(th_scores, gts, scores)

    if np.isnan(scores_simple["Affiliation precision"]):
        scores_simple["Affiliation precision"] = 0.0
    if np.isnan(scores_simple["Affiliation recall"]):
        scores_simple["Affiliation recall"] = 0.0
    if scores_simple["Affiliation precision"] == 0 and scores_simple["Affiliation recall"] == 0:
        aff_f1_score = 0.0
    else:
        aff_f1_score = 2 * scores_simple["Affiliation precision"] * scores_simple["Affiliation recall"] / (scores_simple["Affiliation precision"] + scores_simple["Affiliation recall"])

    # print(scores_simple)

    queue.put({
        "acc": acc,
        "prec": prec,
        "recall": recall,
        "f1_score": f1_score,
        "aff_prec": scores_simple["Affiliation precision"],
        "aff_recall": scores_simple["Affiliation recall"],
        "aff_f1_score": aff_f1_score,
        "percentile": percentile
    })


class Trainer:
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = MODEL_DICT[args.model_name](args).to(self.device)
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
            device = torch.device('cuda:{}'.format(self.args.devices))  # TODO: multi gpus도 호환되나?
            print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def get_data_loader(self, flag, step_size, split_ratio=None, data_name=None):
        if data_name in UCR_DATA_NAME:
            dataset = DATA_DICT[self.args.data_name if data_name is None else data_name](
                self.args.data_path,
                self.args.win_size,
                step=step_size,
                split_ratio=split_ratio,
                flag=flag,
                norm=self.args.norm,
                data_name=self.args.data_name
            )
        else:
            dataset = DATA_DICT[self.args.data_name if data_name is None else data_name](
                self.args.data_path,
                self.args.win_size,
                step=step_size,
                split_ratio=split_ratio,
                flag=flag,
                norm=self.args.norm
            )

        if flag == "test":
            shuffle = False
            drop_last = False
        else:
            shuffle = True
            drop_last = False

        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )

        return data_loader

    def get_test_results(self, scores_for_th, scores, gts, percentile):
        th = np.percentile(scores_for_th, percentile)

        preds = (scores > th).astype(int)

        gts, preds = point_adjustment(gts, preds)

        acc = accuracy_score(gts, preds)

        prec, recall, f1_score, _ = precision_recall_fscore_support(
            gts, preds, average="binary"
        )

        eval_scores_dict = combine_all_evaluation_scores(preds, gts, scores)

        aff_prec = eval_scores_dict["Affiliation precision"]
        aff_recall = eval_scores_dict["Affiliation recall"]
        aff_f1_score = 2 * aff_prec * aff_recall / (aff_prec + aff_recall)

        return {
            "acc": acc,
            "prec": prec,
            "recall": recall,
            "f1_score": f1_score,
            "aff_prec": aff_prec,
            "aff_recall": aff_recall,
            "aff_f1_score": aff_f1_score
        }

    def validate(self, criterion):
        self.model.eval()

        val_loader = self.get_data_loader(
            flag="val", step_size=self.args.test_step_size, data_name=self.args.data_name,
            split_ratio=self.args.split_ratio
        )
        val_loader.dataset.describe()

        val_loss = AverageMeter()

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x = x.float().to(self.device)

                preds = self.model(x)

                loss = criterion(preds, x)

                val_loss.update(loss.item())

        return val_loss.avg

    def train_epoch(self, tr_loader, criterion, optimizer):
        self.model.train()

        tr_loss = AverageMeter()

        for i, (x, y) in enumerate(tr_loader):
            # x : (batch size, window length, # vars)
            # y : (batch size, window length)
            x = x.float().to(self.device)
            
            optimizer.zero_grad()
            
            preds = self.model(x)
            
            loss = criterion(preds, x)
            
            tr_loss.update(loss.item())
            
            loss.backward()
            
            optimizer.step()
        
        return tr_loss.avg

    def train(self):
        tr_loader = self.get_data_loader(
            flag="train", step_size=self.args.step_size, data_name=self.args.data_name,
            split_ratio=self.args.split_ratio
        )
        tr_loader.dataset.describe()

        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        criterion = nn.MSELoss()
        # anomaly_criterion = nn.MSELoss(reduction="none")

        for epoch in range(self.args.num_epochs):
            tr_loss = self.train_epoch(tr_loader, criterion, optimizer)

            if self.args.split_ratio is not None:
                val_loss = self.validate(criterion)
                print("Epoch {} tr loss {} val loss {}".format(epoch + 1, tr_loss, val_loss))
            else:
                print("Epoch {} tr loss {}".format(epoch + 1, tr_loss))

        state_dict = {
            "epoch" : epoch + 1,
            "tr_loss": tr_loss,
            "model": self.model.state_dict()
        }

        if self.args.split_ratio is not None:
            state_dict["val_loss"] = val_loss

        torch.save(state_dict, os.path.join(self.args.ckpt, "{}_{}.pth".format(
            self.args.data_name, self.args.model_name
        )))

    def test(self):
        # Use a training set and a (if possible) validation set 
        tr_loader = self.get_data_loader(
            flag="train",
            step_size=self.args.step_size,
            data_name=self.args.data_name
        )
        # Use a test set for evaluation
        test_loader = self.get_data_loader(
            flag="test",
            step_size=self.args.test_step_size,
            data_name=self.args.data_name
        )

        state_dict = torch.load(os.path.join(self.args.ckpt, "{}_{}.pth".format(
            self.args.data_name, self.args.model_name
        )))

        self.model.load_state_dict(state_dict["model"])
        self.model.cuda()
        self.model.eval()

        anomaly_criterion = nn.MSELoss(reduction="none")

        scores_for_th_list = []
        scores_list = []
        gts_list = []

        with torch.no_grad():
            # Collect the scores from a training set and a validation set for
            # threshold selection
            for i, (x, y) in enumerate(tr_loader):
                x = x.float().to(self.device)

                recons = self.model(x)

                scores_for_th_list.append(
                    anomaly_criterion(recons, x).mean(-1).detach().cpu().numpy()
                )

            # Collect the scores and ground truths for evaluation
            for i, (x, y) in enumerate(test_loader):
                x = x.float().to(self.device)

                recons = self.model(x)

                scores_list.append(
                    anomaly_criterion(recons, x).mean(-1).detach().cpu().numpy()
                )
                gts_list.append(y.detach().cpu().numpy())

        scores_for_th = np.concatenate(scores_for_th_list, axis=None)

        scores = np.concatenate(scores_list, axis=None)
        gts = np.concatenate(gts_list, axis=None)

        print("scores_for_th {} scores {} gts {}".format(
            scores_for_th.shape, scores.shape, gts.shape
        ))

        test_results_dict = self.get_test_results(scores_for_th, scores, gts, self.args.percentile)

        print("accuracy ", test_results_dict["acc"])
        print("precision ", test_results_dict["prec"])
        print("recall ", test_results_dict["recall"])
        print("f1 score ", test_results_dict["f1_score"])
        print("affiliation precision ", test_results_dict["aff_prec"])
        print("affiliation recall ", test_results_dict["aff_recall"])
        print("affiliation f1 score ", test_results_dict["aff_f1_score"])