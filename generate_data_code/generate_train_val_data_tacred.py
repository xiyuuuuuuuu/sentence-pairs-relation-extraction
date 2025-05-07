from typing import Dict, List, Literal, Any
from tqdm import tqdm
import json
import datasets
from datasets import Dataset
from datasets import load_dataset
import argparse
from entity_marker_with_reg import typed_entity_marker_punct
import hashlib
import os


def read_valdata(dev_test_path: str):
    """
    用 dev_test_path 文件 构造用于 train 和 val 的数据集。
    train 数据集包含所有的 support sentence，和对应的ss_relation。
    val 数据集包含所有的 (test sentence，support sentence) pair 和对应的 ts_relation 和 ss_relation。
    """

    print(f"Read valdata {dev_test_path}")
    val = []         # 用于保存最终输出的样本列表
    train = []
    # load（格式为: [episodes, selections, relations]）
    with open(dev_test_path) as fin:
        val_data = json.load(fin)


    # 遍历每个 episode（三元组并行 zip：episode, selections, relations）
    for episode, selections, relations in zip(val_data[0], val_data[1], val_data[2]):
        has_added = 0
        for ts in episode['meta_test']:  # 遍历目标任务中的测试样本（ts = test sentence）

            # 获取‘meta_train’中所有的 support sentence（ss）作为 train
            episode_ss = [y for x in episode['meta_train'] for y in x]
            preprocess_episode_ss = []
            for ss in episode_ss:
                new_sentence = typed_entity_marker_punct(ss)
                preprocess_episode_ss.append({
                    'sentence': new_sentence,
                    'relation': ss['relation']
                })
            # 加入 train 用于输出数据集
            if not has_added:
                for ss in preprocess_episode_ss:
                    train.append({
                        'sentence': ss['sentence'],
                        'relation': ss['relation']
                    })
                has_added = 1
            
            relations = [s['relation'] for s in episode_ss]

            # 对当前 test sentence 进行预处理（如 lowercase、依赖分析等）
            ts_sentence = typed_entity_marker_punct(ts)
            for ss in preprocess_episode_ss:
                val.append({
                    'ts_sentence': ts_sentence,
                    'ss_sentence': ss['sentence'],
                    'ts_relation': ts['relation'] if ts['relation'] in relations else 'no_relation',
                    'ss_relation': ss['relation']
                })
    # 返回
    return val, train

def write_to_jsonl(data, output_path):
    """
    将数据写入 JSONL 文件。
    每个元素作为一行写入文件。
    """
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Data written to {output_path}")

def generate_data(args):
    """
    用于加载和预处理验证数据集。
    
    工作流程：
    - 读取规则（rules）
    - 加载每一个 dev_path 中的验证数据（read_valdata）
    - 对验证数据进行 tokenize（调用模型自带的 tokenizer）
    - 返回一个列表，每个元素是一个经过 tokenization 的 HuggingFace Dataset
    """
    

    # 合成 input 和 output 的路径
    type_episodes = args['dev/test']
    input_dir_path = args['dev/test_dir_path']
    output_dir_path = args['output_path']
    input_path = input_dir_path + type_episodes + "/"
    output_path = output_dir_path + type_episodes + "/"
    

    # file name prefix
    file_name_prefix = "5_way_1_shots_10K_episodes_3q_seed_16029"

    # all the train data
    all_train_data = []

    # 遍历每个 dev_path（每个 path 是一个 episodic JSON 文件）
    for i in range(5):
        # 构建输入输出完整路径
        input_file_path = input_path + file_name_prefix + str(i) + ".json"
        output_train_file_path = output_path + "/train/"+ file_name_prefix + str(i) + ".jsonl"
        output_val_file_path = output_path + "/val/"+ file_name_prefix + str(i) + ".jsonl"
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path + "/train/"), exist_ok=True)
        os.makedirs(os.path.dirname(output_path + "/val/"), exist_ok=True)
        # 使用 read_valdata() 加载原始验证数据，并附上规则
        val, train = read_valdata(input_file_path)
        # merge train 数据
        all_train_data = all_train_data + train
        # 输出 val 和 train 到 JSON 文件
        write_to_jsonl(val, output_val_file_path)
        write_to_jsonl(train, output_train_file_path)
    
    # output all train data
    all_train_file_path = output_path + "/train/all_train.jsonl"
    write_to_jsonl(all_train_data, all_train_file_path)




# 主程序入口
if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()

    # 加载 dev/test 数据路径
    parser.add_argument("--dev/test_dir_path", type=str, default="/fsre_dataset/TACRED/")
    parser.add_argument("--output_path", type=str, default="/data/nlp/xinyu/TACRED_Dataset/generated_datasets/")
    parser.add_argument("--dev/test", type=str, default="test_episodes")

    # 解析命令行参数为字典
    args = vars(parser.parse_args())
    print(args)

    # 调用主函数，传入参数字典
    # 生成数据
    generate_data(args)





