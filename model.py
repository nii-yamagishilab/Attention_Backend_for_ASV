import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AttentionAlphaComponent
import copy
import math

def mean(x, dim = 0):
    return torch.mean(x, dim = dim)

class AttentionAggregation(nn.Module):
    def __init__(self, opts):
        super(AttentionAggregation, self).__init__()
        self.head = opts['head']
        self.d_model = opts['d_model']
        self.pooling = opts['pooling']
        self.attention_transformation = nn.MultiheadAttention(self.d_model, self.head)
        if self.pooling == 'mean':
            self.aggregation = mean
        elif self.pooling == 'mha':
            self.aggregation = AttentionAlphaComponent(self.d_model, self.head)
        self.logistic = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def merge(self, x):
        '''
        params:
            x: (N, M, D), N speakers, M utterances per speaker, D dimension
        '''
        assert len(x.size()) == 3
        n_spks, n_utts, dimension = x.size()
        x = x.repeat(1, n_utts, 1) 
        mask = torch.logical_not(torch.eye(n_utts)).repeat(n_spks, 1).view(-1).to(x.device) 
        masked_x = x.view(-1, self.d_model)[mask].contiguous().view(n_spks * n_utts, -1, self.d_model) 
        masked_x = masked_x.transpose(0, 1).contiguous() # n_utts - 1, n_spks * n_utts, dimension
        x, _ = self.attention_transformation(masked_x, masked_x, masked_x, None) # n_utts - 1, n_spks * n_utts, dimension
        x = x + masked_x
        if self.pooling == 'mean':
            aggregation = mean(x) # (n_spks * (n_utts - 1), dimension), n_spks * (n_utts - 1)
        else:
            x = x.permute(1, 2, 0) # n_spks * n_utts, dimension, n_utts - 1
            alpha = self.aggregation(x)
            aggregation = x.view(n_spks * n_utts, self.head, self.d_model // self.head, -1).matmul(alpha.unsqueeze(-1)).view(n_spks * n_utts, self.d_model)
        return aggregation

    def forward(self, x):
        '''
        params:
            x: (N, M, D), N speakers, M utterances per speaker, D dimension
        '''
        n_spks, n_utts, dimension = x.size()
        aggregation = self.merge(x) # obtain n_spks * n_utts centers
        x = x.view(-1, self.d_model)
        cosine_score_matrix = F.normalize(x).matmul(F.normalize(aggregation).T) 
        mask = torch.eye(n_utts, dtype = torch.bool).repeat(n_spks, n_spks).to(x.device) # mask
        ground_truth_matrix = torch.eye(mask.size(0)).to(x.device) 
        scores = cosine_score_matrix.view(-1)[mask.view(-1)].view(-1, 1)
        ground_truth = ground_truth_matrix.view(-1)[mask.view(-1)].view(-1, 1)
        scores = self.logistic(scores)
        scores = self.sigmoid(scores)
        return scores, ground_truth

    def center(self, x):
        x = x.transpose(0, 1).contiguous() # n_utts, 1, dimension
        output, _ = self.attention_transformation(x, x, x, None)
        x += output # n_utts, 1, dimension
        if self.pooling == 'mean':
            aggregation = mean(x) # (n_spks * (n_utts - 1), dimension), n_spks * (n_utts - 1)
        else:
            x = x.permute(1, 2, 0) # 1, dimension, n_utts 
            alpha = self.aggregation(x)
            aggregation = x.view(x.size(0), self.head, self.d_model // self.head, -1).matmul(alpha.unsqueeze(-1)).view(x.size(0), self.d_model)
        return F.normalize(aggregation)
        
    def test(self, center, evaluation):
        cosine_score = F.cosine_similarity(evaluation, center) 
        score = self.logistic(cosine_score.view(-1, 1))
        return cosine_score
