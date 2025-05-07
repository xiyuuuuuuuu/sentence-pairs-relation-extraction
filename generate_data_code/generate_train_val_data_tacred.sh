#!/bin/bash



# dev
python generate_data_code/generate_train_val_data_tacred.py \
    --dev/test_dir_path /storage2/data/nlp/corpora/rvacareanu/gpu07-backup/storage/rvacareanu/data/softrules/fsre_dataset/TACRED/ \
    --output_path generated_datasets/ \
    --dev/test dev_episodes

