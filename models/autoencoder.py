import torch
import torch.nn as nn

from models.RevIN import RevIN
from models.embedding import DataEmbedding
from models.encoder import TransformerEncoder
from models.decoder import SimpleLinearBlock


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        if args.data_name == "MSL" or args.data_name == "SMAP":
            self.revin = RevIN(num_features=1)
        elif args.data_name == "SWAN_SF":
            self.revin = RevIN(num_features=33)
        else:
            self.revin = RevIN(num_features=args.n_vars)
        self.data_emb = DataEmbedding(args)
        self.encoder = TransformerEncoder(args)
        self.decoder = SimpleLinearBlock(args)
    
    def forward(self, x):
        '''
        x : (B, L, N) input time series segments
        '''
        if self.args.data_name == "MSL" or self.args.data_name == "SMAP":
            x[:, :, :1] = self.revin(x[:, :, :1], "norm")
        elif self.args.data_name == "SWAN_SF":
            temp = self.revin(torch.cat([x[:, :, :31], x[:, :, 36:]], axis=-1), "norm")
            x[:, :, :31] = temp[:, :, :31]
            x[:, :, 36:] = temp[:, :, 31:]
        else:
            x = self.revin(x, "norm")
        x = self.data_emb(x)
        x = self.encoder(x)
        x = self.decoder(x)
        if self.args.data_name == "MSL" or self.args.data_name == "SMAP":
            x[:, :, :1] = self.revin(x[:, :, :1], "denorm")
        elif self.args.data_name == "SWAN_SF":
            temp = self.revin(torch.cat([x[:, :, :31], x[:, :, 36:]], axis=-1), "denorm")
            x[:, :, :31] = temp[:, :, :31]
            x[:, :, 36:] = temp[:, :, 31:]
        else:
            x = self.revin(x, "denorm")
        return x