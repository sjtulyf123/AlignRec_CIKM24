import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_scatter import scatter_max
except:
    pass

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class ALIGNREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(ALIGNREC, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.desc = config['desc']
        self.use_ln=config['use_ln']
        self.sim_weight = config['sim_weight']
        self.ui_cosine_loss_weight = config['ui_cosine_loss_weight']
        self.use_cross_att= False
        self.use_user_history = False
        self.add_user_history_after_content_embs = False
        self.reg_loss = EmbLoss()

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # self.mm_adj_name="raw_feats" if config['multimodal_data_dir']=='' else config['multimodal_data_dir'].split('/')[-2]
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        
        mm_adj_file = os.path.join(dataset_path, 'mgcn_zkn_adj_{}_{}_{}.pt'.format(self.knn_k, self.sparse, self.desc))

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)


        if self.v_feat is not None:
            self.mm_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            mm_adj = build_sim(self.mm_embedding.weight.detach())
            mm_adj = build_knn_normalized_graph(mm_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                    norm_type='sym')
            self.mm_original_adj = mm_adj.cuda()

        if self.v_feat is not None:
            if self.use_ln:
                self.v_ln = nn.LayerNorm(self.v_feat.shape[1])
            self.mm_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)

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
        
        self.use_bce=False
        print("use_bce",self.use_bce)
        if self.use_bce:
            self.sigbce_loss=nn.BCEWithLogitsLoss()
        self.side_emb_div = config['side_emb_div'] # set to 0

        self.use_hist_decoder = config['use_hist_decoder'] # False
        if self.use_hist_decoder:
            self.user_topk_hist = self.init_user_hist_info(dataset)
            self.query = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.key = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.value = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            self.hist_ln1 = nn.LayerNorm(self.embedding_dim)

        self.test_arch1 = config['test_arch1'] # False
        if self.test_arch1:
            self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
            nn.init.xavier_normal_(self.predictor.weight)
        self.ui_cosine_loss = config['ui_cosine_loss'] # False

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

    def forward(self, adj, users=None, train=False):
        if self.v_feat is not None:
            if self.use_ln:
                mm_feats = self.mm_trs(self.v_ln(self.mm_embedding.weight))
            else:
                mm_feats = self.mm_trs(self.mm_embedding.weight)

        # Behavior-Guided Purifier
        mm_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(mm_feats))

        # User-Item View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight

        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
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

        mm_user_embeds = torch.sparse.mm(self.R,mm_item_embeds)

        mm_embeds = torch.cat([mm_user_embeds, mm_item_embeds], dim=0)


        if self.side_emb_div!=0:
            all_embeds = content_embeds + mm_embeds/self.side_emb_div
        else: 
            all_embeds = content_embeds + mm_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if self.use_hist_decoder and train:
            hist_seq = mm_item_embeds[self.user_topk_hist[users],:]
            hist_seq = torch.where(torch.unsqueeze(self.user_topk_hist[users],dim=-1)==-1, torch.zeros_like(hist_seq), hist_seq)
            
            score = torch.bmm(self.query(hist_seq), self.key(hist_seq).transpose(1, 2)) / np.sqrt(self.embedding_dim)
            score.masked_fill_((self.user_topk_hist[users]==-1).view(-1, 1, 10), -float('Inf'))

            attn = F.softmax(score, -1)
            context = torch.bmm(attn, self.value(hist_seq))

            hist_hid = self.hist_ln1(hist_seq + context)

        if train:
            if self.use_hist_decoder:
                return all_embeddings_users, all_embeddings_items, mm_embeds, content_embeds, hist_hid[:,-1,:]
            return all_embeddings_users, all_embeddings_items, mm_embeds, content_embeds

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
    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def sim_loss(self, embedding, sim):
        embedding_sim = torch.mm(embedding, embedding.t())
        # embedding_sim = build_sim(embedding)
        sim_loss = self.reg_loss(embedding_sim - sim.detach())
        return sim_loss

    def sim_sigmoid_loss(self, embedding, sim):
        # embedding_sim = build_sim(embedding)
        embedding_sim = torch.mm(embedding, embedding.t())
        logit_emb_sim = torch.reshape(embedding_sim, (-1, 1)) #8.*torch.reshape(embedding_sim, (-1, 1))
        logit_sim =torch.reshape(sim, (-1, 1))
        target = F.sigmoid(logit_sim)
        sim_loss = self.sigbce_loss(logit_emb_sim, target)
        return sim_loss

    def calculate_loss(self, interaction, not_train_ui=False):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        if self.use_hist_decoder:
            ua_embeddings, ia_embeddings, side_embeds, content_embeds, user_hist_seq = self.forward(
                self.norm_adj,users, train=True)
        else:
            ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
                self.norm_adj,users, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(side_embeds_users[users], content_embeds_user[users], 0.2)

        if self.use_hist_decoder:
            cl_loss+=self.InfoNCE(user_hist_seq, content_embeds_user[users], 0.2)
        if self.ui_cosine_loss:
            batch_mf_loss+=(1 - F.cosine_similarity(u_g_embeddings, pos_i_g_embeddings, dim=-1).mean())*self.ui_cosine_loss_weight

        pos_ii_batch_sim_mat = build_sim(self.v_feat[pos_items])
        neg_ii_batch_sim_mat = build_sim(self.v_feat[neg_items])

        if self.use_bce:
            ii_sim_loss = self.sim_sigmoid_loss(side_embeds_items[pos_items], pos_ii_batch_sim_mat) + self.sim_sigmoid_loss(side_embeds_items[neg_items], neg_ii_batch_sim_mat)
        else:
            ii_sim_loss = self.sim_loss(side_embeds_items[pos_items], pos_ii_batch_sim_mat)

        if not_train_ui:
            return batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.sim_weight * ii_sim_loss
        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss + self.sim_weight * ii_sim_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def init_user_hist_info(self, dataloader):
        uid_field = dataloader.dataset.uid_field
        iid_field = dataloader.dataset.iid_field
        time_field = 'timestamp'
        # load avail items for all uid
        uid_freq = dataloader.dataset.df.groupby(uid_field)[iid_field,time_field]
        result_dict = {uid: list(zip(group[iid_field], group[time_field])) for uid, group in uid_freq}
        for k in result_dict:
            result_dict[k] = sorted(result_dict[k], key=lambda tup: tup[1],reverse=True)
        topk = 10
        hist_topk = np.zeros((self.n_users,topk),dtype=np.int)-1
        # empty hist is padded with -1 ''before'' the hist sequence
        for uid in result_dict:
            for i in range(min(topk,len(result_dict[k]))):
                hist_topk[uid][topk-i-1] = result_dict[uid][i][0]
        return torch.from_numpy(hist_topk).to(self.device)
