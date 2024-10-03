# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN
################################################
Reference:
    https://github.com/demonph10/MGCN
    ACM MM'2023: [Multi-View Graph Convolutional Network for Multimedia Recommendation]
"""

# beit-3 input version of MGCN

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class MGCN_MM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MGCN_MM, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        
        self.use_ln=config['use_ln']
        self.use_cross_att=config['use_cross_att']
        self.use_user_history = config['use_user_history']
        self.add_user_history_after_content_embs = config['add_user_history_after_content_embs']
        self.test_arch1 = config['test_arch1']
        self.test_arch2 = config['test_arch2']
        self.test_arch3 = config['test_arch3']
        
        # max hits length is 100 for baby and 237 for sports
        # min hist length is 3
        self.history_items_per_u=dataset.history_items_per_u
        # if self.use_user_history:
        #     self.hist_matrix, self.valid_flag=self.init_user_hist_info()

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.mm_adj_name="raw_feats" if config['multimodal_data_dir']=='' else config['multimodal_data_dir'].split('/')[-2]
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        
        # mm_adj_file = os.path.join(dataset_path, 'mm_adj_mgcn_{}_{}_{}.pt'.format(self.mm_adj_name, self.knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)


        if self.v_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # if os.path.exists(mm_adj_file):
            #     mm_adj = torch.load(mm_adj_file)
            # else:
            mm_adj = build_sim(self.mm_embedding.weight.detach())
            mm_adj = build_knn_normalized_graph(mm_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                    norm_type='sym')
            # torch.save(mm_adj, mm_adj_file)
            self.mm_original_adj = mm_adj.cuda()

        if self.v_feat is not None:
            if self.use_ln:
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            self.mm_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

        if self.use_cross_att:
            self.mm_prj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh()
            )
            self.content_prj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh()
            )
            self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        else:
            self.softmax = nn.Softmax(dim=-1)

            self.query_common = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh(),
                nn.Linear(self.embedding_dim, 1, bias=False)
            )

            self.gate_mm_prefer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        self.gate_v = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Sigmoid()
            )
        if self.test_arch1:
            # self.mm_emb_mlp = nn.Sequential(
            #     nn.Linear(self.embedding_dim, self.embedding_dim),
            #     nn.BatchNorm1d(self.embedding_dim),
            #     nn.ReLU()
            # )
            # self.mm_emb_mlp = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            # self.W1 = nn.Parameter(torch.eye(self.embedding_dim, self.embedding_dim), requires_grad=False)
            self.id_ln = nn.LayerNorm(self.embedding_dim)
        if self.test_arch2:
            # self.mm_emb_mlp = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            # self.W1 = nn.Parameter(torch.eye(self.embedding_dim, self.embedding_dim), requires_grad=False)
            self.ln1 = nn.LayerNorm(self.embedding_dim)
            self.ln2 = nn.LayerNorm(self.embedding_dim)
            self.gcn_ln = [self.ln1,self.ln2]
        if self.test_arch3:
            self.mm_emb_mlp = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        
        self.concat_emb = config['concat_emb']
        if self.concat_emb:
            self.concat_mlp = nn.Linear(self.embedding_dim*2, self.embedding_dim, bias=False)

        self.side_emb_div = config['side_emb_div']

        self.tau = 0.5

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            if self.use_ln:
                mm_feats = self.mm_trs(self.v_ln(self.mm_embedding.weight))
            else:
                mm_feats = self.mm_trs(self.mm_embedding.weight)

        if self.test_arch1:
            mm_item_embeds = torch.multiply(self.id_ln(self.item_id_embedding.weight), self.gate_v(mm_feats))
            item_embeds = self.id_ln(self.item_id_embedding.weight)
            user_embeds = self.id_ln(self.user_embedding.weight)
        else:
            # Behavior-Guided Purifier
            mm_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(mm_feats))

            # User-Item View
            item_embeds = self.item_id_embedding.weight
            user_embeds = self.user_embedding.weight

        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        if self.test_arch3:
            ego_embeddings = self.mm_emb_mlp(ego_embeddings)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            if self.test_arch2:
                side_embeddings = self.gcn_ln[i](side_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Item-Item View
        if self.sparse:
            for i in range(self.n_layers):
                mm_item_embeds = torch.sparse.mm(self.mm_original_adj, mm_item_embeds)
        else:
            for i in range(self.n_layers):
                mm_item_embeds = torch.mm(self.mm_original_adj, mm_item_embeds)
        mm_user_embeds = torch.sparse.mm(self.R, mm_item_embeds)
        # added by lyf: utilize user history
        if self.use_user_history:
            user_hist_sum = torch.sparse.mm(self.R, mm_feats)
            if not self.add_user_history_after_content_embs:
                mm_user_embeds = mm_user_embeds+user_hist_sum
            else:
                content_embeds[:user_embeds.shape[0]]=content_embeds[:user_embeds.shape[0]]+user_hist_sum

        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)

        # Behavior-Aware Fuser
        if self.use_cross_att:
            # cross att
            content_embeds_new = self.content_prj(content_embeds)
            mm_embeds_new = self.mm_prj(mm_embeds)
            mm_q = self.query(content_embeds_new)
            mm_k = self.key(mm_embeds_new)
            mm_v = self.value(mm_embeds_new)
            score = torch.multiply(mm_q, mm_k)
            attn = F.softmax(score, -1)
            side_embeds = torch.multiply(attn, mm_v)
        else:
            if self.test_arch1:
                # content_embeds = self.mm_emb_mlp(content_embeds)
                # content_embeds = torch.matmul(content_embeds, self.W1)
                side_embeds = mm_embeds/2
                # mm_prefer = self.gate_mm_prefer(content_embeds)
                # side_embeds = torch.multiply(mm_prefer, mm_embeds)
            elif self.test_arch2:
                side_embeds = mm_embeds/2
                # side_embeds = torch.zeros_like(content_embeds)
                # side_embeds = self.mm_emb_mlp(mm_embeds)
                # side_embeds = torch.matmul(mm_embeds, self.W1)
            elif self.test_arch3:
                # content_embeds = self.mm_emb_mlp(content_embeds)
                side_embeds = mm_embeds/2
            else:
                # att_common = self.query_common(mm_embeds)
                # weight_common = self.softmax(att_common)
                # common_embeds = weight_common[:, 0].unsqueeze(dim=1) * mm_embeds
                # sep_mm_embeds = mm_embeds - common_embeds

                # mm_prefer = self.gate_mm_prefer(content_embeds)

                # sep_mm_embeds = torch.multiply(mm_prefer, sep_mm_embeds)

                # side_embeds = (sep_mm_embeds + common_embeds) / 2
                
                if self.side_emb_div!=0:
                    side_embeds = mm_embeds / self.side_emb_div
                else:
                    # above is the same as follows:
                    side_embeds = mm_embeds / 2
            
        if self.concat_emb:
            all_embeds = self.concat_mlp(torch.cat((content_embeds,side_embeds),dim=-1))
        else:
            all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
    
    def bpr_loss_2(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)
        
        # added by lyf
        mf_v_loss = 0.0
        # if self.concat_multimodal_input:
        #     # this loss is similar to freedom
        #     if self.use_ln:
        #         pos_mm_feats = self.mm_trs(self.v_ln(self.mm_embedding.weight[pos_items,:]))
        #         neg_mm_feats = self.mm_trs(self.v_ln(self.mm_embedding.weight[neg_items,:]))
        #         mf_v_loss = self.bpr_loss_2(u_g_embeddings, pos_mm_feats, neg_mm_feats)
        #     else:
        #         mm_feats = self.mm_trs(self.mm_embedding.weight)
        #         mf_v_loss = self.bpr_loss_2(u_g_embeddings, mm_feats[pos_items], mm_feats[neg_items])
        if not_train_ui:
            return batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.reg_weight*mf_v_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # added by lyf
    def init_user_hist_info(self):
        num_users = self.n_users
        num_items = max(len(items) for items in self.history_items_per_u.values())
        hist_matrix = np.zeros((num_users, num_items),dtype=np.int)-1
        valid_flag = np.zeros((num_users, num_items),dtype=np.float)
        for one_user in self.history_items_per_u:
            tmp_tensor = np.array(sorted(list(self.history_items_per_u[one_user])))
            hist_matrix[int(one_user),:len(tmp_tensor)]=tmp_tensor
            valid_flag[int(one_user),:len(tmp_tensor)]=1
        return torch.from_numpy(hist_matrix).to(self.device), torch.from_numpy(valid_flag).to(self.device)