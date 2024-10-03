# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError
    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # added by lyf: flag of mask image or text
        self.mask_image = config['mask_image']
        self.mask_text = config['mask_text']
        if self.mask_image or self.mask_text:
            self.mask_ratio = config['mask_ratio']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        v_feat, t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            if config['multimodal_data_dir']=='':
                dataset_path = os.path.abspath(config['data_path']+config['dataset'])
            else:
                dataset_path = os.path.join(config['multimodal_data_dir'],config['dataset']+'_'+config['multimodal_data_dir'].split('/')[-2])
            # if file exist?
            # print(dataset_path)
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)

            # replace some feature with masked image or text
            if self.mask_image or self.mask_text:
                assert not (self.mask_image and self.mask_text), "mask_image and mask_text cannot be True simultaneously"
                mask_idx_dataset_path = os.path.abspath(config['data_path']+config['dataset'])
                mask_idx_file_name = 'mask_image_{}_mask_text_{}.npy'.format(self.mask_image,self.mask_text)
                mask_idx_path = os.path.join(mask_idx_dataset_path,mask_idx_file_name)
                if os.path.exists(mask_idx_path):
                    with open(mask_idx_path, 'rb') as f:
                        global_index_to_mask = np.load(f)
                else:
                    global_index_to_mask = torch.randperm(v_feat.shape[0]).numpy()
                    with open(mask_idx_path, 'wb') as f:
                        np.save(f,global_index_to_mask)

                index_to_mask = torch.from_numpy(global_index_to_mask).to(self.device)[:int(self.mask_ratio*v_feat.shape[0])]
                print(index_to_mask)
                if config['multimodal_data_dir']=='' or ('clip' in config['multimodal_data_dir'].split('/')[-2]):
                    # clip or amazon feature: just replace masked modal with 0
                    if self.mask_image:
                        v_feat[index_to_mask,:] = torch.zeros(v_feat.shape[1]).to(self.device)
                    if self.mask_text:
                        t_feat[index_to_mask,:] = torch.zeros(v_feat.shape[1]).to(self.device)
                else:
                    # beit3 feature: replace masked modal (just v_feat) with another file
                    mask_file = os.path.join(os.path.join(config['beit3_mask_data_folder'],
                        config['dataset']+'_'+config['beit3_mask_data_folder'].split('/')[-2]), config['vision_feature_file'])
                    mask_feat = torch.from_numpy(np.load(mask_file, allow_pickle=True)).type(torch.FloatTensor).to(
                        self.device)
                    # in beit3 based model we only use v_feat(i.e.: actually the first cls in beit3)
                    v_feat[index_to_mask,:] = mask_feat[index_to_mask,:]

            self.v_feat = v_feat
            self.t_feat = t_feat

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'
