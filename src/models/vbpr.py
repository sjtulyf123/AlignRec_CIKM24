# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
VBPR -- Recommended version
################################################
Reference:
VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback -Ruining He, Julian McAuley. AAAI'16
"""
import numpy as np
import os
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F


class VBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way.
    """
    def __init__(self, config, dataloader):
        super(VBPR, self).__init__(config, dataloader)
        self.use_ln=config['use_ln']
        self.concat_multimodal_input=config['concat_multimodal_input']

        # load parameters info
        self.u_embedding_size = self.i_embedding_size = config['embedding_size']
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))

        if self.concat_multimodal_input:
            # self.v_feat=torch.cat((self.t_feat, self.v_feat), -1)
            self.t_feat=None

        if self.use_ln:
            if self.concat_multimodal_input:
                self.vt_linear = nn.Linear(self.v_feat.shape[1], self.i_embedding_size)
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            else:
                self.vt_linear = nn.Linear(self.t_feat.shape[1]+self.v_feat.shape[1], self.i_embedding_size)
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
                self.t_ln = nn.LayerNorm(self.t_feat.shape[1])
        else:
            if self.v_feat is not None and self.t_feat is not None:
                self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
            elif self.v_feat is not None:
                self.item_raw_features = self.v_feat
            else:
                self.item_raw_features = self.t_feat

            self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)
        self.loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.u_embedding[user, :]

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding[item, :]

    def forward(self, user=None, pos_item=None, neg_item=None, dropout=0.0):
        if self.use_ln:
            v_enc_pos = self.v_ln(self.v_feat[pos_item,:])
            if self.concat_multimodal_input:
                enc_pos = self.vt_linear(v_enc_pos)
            else:
                t_enc_pos = self.t_ln(self.t_feat[pos_item,:])
                enc_pos = torch.cat((t_enc_pos,v_enc_pos),-1)
                enc_pos = self.vt_linear(enc_pos)
            item_embeddings_pos = torch.cat((self.i_embedding[pos_item, :], enc_pos), -1)

            v_enc_neg = self.v_ln(self.v_feat[neg_item,:])
            if self.concat_multimodal_input:
                enc_neg = self.vt_linear(v_enc_neg)
            else:
                t_enc_neg = self.t_ln(self.t_feat[neg_item,:])
                enc_neg = torch.cat((t_enc_neg,v_enc_neg),-1)
                enc_neg = self.vt_linear(enc_neg)
            item_embeddings_neg = torch.cat((self.i_embedding[neg_item, :], enc_neg), -1)

            user_e = F.dropout(self.u_embedding[user,:], dropout)
            item_pos = F.dropout(item_embeddings_pos, dropout)
            item_neg = F.dropout(item_embeddings_neg, dropout)
            return user_e, item_pos, item_neg
        else:
            item_embeddings = self.item_linear(self.item_raw_features)
            item_embeddings = torch.cat((self.i_embedding, item_embeddings), -1)

            user_e = F.dropout(self.u_embedding, dropout)
            item_e = F.dropout(item_embeddings, dropout)
            return user_e, item_e

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        if self.use_ln:
            user_e, pos_e, neg_e = self.forward(user, pos_item, neg_item)
        else:
            user_embeddings, item_embeddings = self.forward()
            user_e = user_embeddings[user, :]
            pos_e = item_embeddings[pos_item, :]
            #neg_e = self.get_item_embedding(neg_item)
            neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_e, pos_e, neg_e)
        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        if self.use_ln:
            v_enc_pos = self.v_ln(self.v_feat)
            if self.concat_multimodal_input:
                enc_pos = self.vt_linear(v_enc_pos)
            else:
                t_enc_pos = self.t_ln(self.t_feat)
                enc_pos = torch.cat((t_enc_pos,v_enc_pos),-1)
                enc_pos = self.vt_linear(enc_pos)
            item_embeddings_pos = torch.cat((self.i_embedding, enc_pos), -1)

            user_e = self.u_embedding[user,:]
            score = torch.matmul(user_e, item_embeddings_pos.transpose(0, 1))
        else:
            user_embeddings, item_embeddings = self.forward()
            user_e = user_embeddings[user, :]
            all_item_e = item_embeddings
            score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score
