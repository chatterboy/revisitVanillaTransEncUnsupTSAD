import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class PointEmbedding(nn.Module):
    def __init__(self, args):
        super(PointEmbedding, self).__init__()
        self.linear = nn.Linear(args.n_vars, args.d_model, bias=False)

    def forward(self, x):
        return self.linear(x)


class LocalEmbedding(nn.Module):
    def __init__(self, args):
        super(LocalEmbedding, self).__init__()
        self.conv = nn.Conv1d(
            args.n_vars, args.d_model, args.k_size,
            padding="same", bias=False
        )

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, args):
        super(DataEmbedding, self).__init__()
        if args.data_embed == "point":
            self.model = PointEmbedding(args)
        elif args.data_embed == "local":
            self.model = LocalEmbedding(args)
        else:
            ValueError("Expected 'point' or 'local', but got '{}'".format(args.data_embed))

    def forward(self, x):
        '''
        x : (batch size, window length, # vars)
        return : (batch size, window length, feature size)
        '''
        return self.model(x)


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # TODO: Here, we use [batch size, sequence length, embed dim]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.transpose(0, 1)
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        
        # init.uniform_(self.pe, -0.02, 0.02)

        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.pe, a=math.sqrt(5))

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x + self.pe[:x.size(1), :]
        x = self.dropout(x)
        return x


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, args):
        super(AbsolutePositionEncoding, self).__init__()
        if args.pe_mode == 'fixed':
            self.pe = FixedPositionalEncoding(args.d_model, max_len=args.win_size)
        elif args.pe_mode == 'learnable':
            self.pe = LearnablePositionalEncoding(args.d_model, max_len=args.win_size)
        else:
            raise ValueError('Expected `fixed` or `learnable`, but got `{}`'.format(mode))
    
    def forward(self, x):
        x = self.pe(x)
        return x