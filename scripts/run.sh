#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

MARKER=$(date "+%Y%m%d_%H%M") # Add a timestamp to the marker for logging

python /data/nlp/xinyu/nlp_proj/src/sentence_pair.py \
    --finetuning_examples 50000 \
    --do_train \
    --training_path \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/train/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl" \
    --evaluation_paths \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl" \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160291.jsonl" \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160292.jsonl" \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160293.jsonl" \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160294.jsonl" \
    --find_threshold_on_path \
        "/data/nlp/xinyu/nlp_proj/generated_datasets/dev_episodes/val/5_way_1_shots_10K_episodes_3q_seed_160290.jsonl" \
    --append_results_to_file "/data/nlp/xinyu/nlp_proj/results/tacred_1shot.jsonl" \
    >> /data/nlp/xinyu/nlp_proj/log/${MARKER}_seedsall_k1.txt

