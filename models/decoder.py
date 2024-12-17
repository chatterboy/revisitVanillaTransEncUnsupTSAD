import torch.nn as nn


class SimpleLinearBlock(nn.Module):
    def __init__(self, args):
        super(SimpleLinearBlock, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(args.d_model, args.n_vars)

    def forward(self, x):
        '''
        x : (batch size, window length, feature size)
        return : (batch size, window length, # variables)
        '''
        x = self.dropout(x)
        x = self.linear(x)
        return x