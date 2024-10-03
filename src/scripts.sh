echo "run scripts"
# torch-scatter

# rub alignrec with our new multimodal features
python main.py --model ALIGNREC --dataset baby --log_file_name alignrec_test --ui_cosine_loss --multimodal_data_dir ../data/beit3_128token_add_title_brand_to_text/
python main.py --model ALIGNREC --dataset sports --log_file_name alignrec_test --ui_cosine_loss --multimodal_data_dir ../data/beit3_128token_add_title_brand_to_text/

# run other methods like vbpr, please do not add --multimodal_data_dir, and then the code will load default multimodal features
# python main.py --model VBPR --dataset baby --log_file_name alignrec_test