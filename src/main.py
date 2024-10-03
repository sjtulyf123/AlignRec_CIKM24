# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', type=int, default=0, help='use which gpu id')
    parser.add_argument('--log_file_name', type=str, default='change_learning_rate', help='log file name (already register model and dataset in code)')
    parser.add_argument('--multimodal_data_dir', type=str, default='', help='image and text feat dir')
    parser.add_argument('--use_ln', action='store_true', help='use layer_norm after loading mm features')
    parser.add_argument('--concat_multimodal_input', action='store_true', help='concat multimodal input (or just use one beit3 input)')
    parser.add_argument('--save_model', action='store_true', help='whether save best model for each hyper-parameter combination')

    parser.add_argument('--knn_k_user', type=int, default=20, help='knn k user for damore model')
    parser.add_argument('--add_mmfeat_aftergcn', action='store_true', help='add multimodal feat after items GCN')
    parser.add_argument('--use_user_history', action='store_true', help=' whether use user history')
    parser.add_argument('--add_user_history_after_content_embs', action='store_true', help=' whether add user history to content emb (false is add user history to image_user_emb)')
    parser.add_argument('--use_cross_att', action='store_true', help=' whether use cross attention between id and mm feature')

    # params for mask some feat in experiments
    parser.add_argument('--mask_image', action='store_true', help='whether mask some image in the data')
    parser.add_argument('--mask_text', action='store_true', help='whether mask some text in the data')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio of the whole data')
    parser.add_argument('--beit3_mask_data_folder', type=str, default='', help='image or text masked data dir for beit3')
    # parser.add_argument('--only_use_text_feat', action='store_true', help='whether only use text feat')
    parser.add_argument('--test_arch1', action='store_true', help='1')
    parser.add_argument('--test_arch2', action='store_true', help='2')
    parser.add_argument('--test_arch3', action='store_true', help='3')
    parser.add_argument('--warmup_ui', action='store_true', help='whether train ui-modality loss first')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='warmup epoch')
    parser.add_argument('--concat_emb', action='store_true', help='concat_emb with content emb and side emb')
    parser.add_argument('--side_emb_div', type=int, default=0, help='divide side emb by xx and then add to content emb (default 2 is better)')

    # zkn part feat
    # parser.add_argument('--sim_weight', type=float, default=0.1, help='sim weight')
    parser.add_argument('--use_uu_adj', action='store_true', help='use uu adj')
    parser.add_argument('--add_uu_sim_loss', action='store_true', help='add uu loss')
    parser.add_argument('--use_hist_decoder', action='store_true', help='add hist decoder and a new infonce loss')
    parser.add_argument('--sim_weight_user', type=float, default=0.1, help='sim weight uu')
    parser.add_argument('--ui_cosine_loss', action='store_true', help='add ui cosine sim loss')

    parser.add_argument('--eval_short_seq', action='store_true', help='whether only evaluate short sequence (ablation study)')

    args, _ = parser.parse_known_args()

    config_dict = vars(args)

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


