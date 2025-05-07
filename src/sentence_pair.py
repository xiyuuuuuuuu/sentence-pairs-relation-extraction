import json
import tqdm
import argparse
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any
from typing import Callable
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from sentence_transformers import CrossEncoder
from scipy.special import expit
from sklearn.metrics import f1_score
from collections import Counter
import sys


NO_RELATION = "no_relation"

def tacred_score(key, prediction, verbose=False):
    """
    这是一个用于 TACRED 任务的评分函数，
    输入为：
        key：每个样本的真实标签（gold label）列表
        prediction：每个样本的模型预测标签（预测的 relation）
        verbose：是否打印详细的每类关系的评估信息（True 则会详细打印每个 relation 的精确率、召回率、F1）

    返回：
        精确率（micro 平均）、召回率（micro 平均）、F1（micro 平均）
    """

    # 初始化三个计数器（字典）：
    # correct_by_relation：记录每种关系上预测正确的次数
    # guessed_by_relation：记录每种关系被模型预测为该类的次数（不管对不对）
    # gold_by_relation：记录每种关系在真实数据中出现的次数
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # 遍历所有样本，按下标 row 一一比对真实值和预测值
    for row in range(len(key)):
        gold = key[row]           # 第 row 个样本的真实标签
        guess = prediction[row]   # 第 row 个样本的预测标签

        # 情况 1：真实标签和预测标签都是 no_relation（都没有关系），这类不计入指标
        if gold == NO_RELATION and guess == NO_RELATION:
            pass  # 什么都不做，跳过

        # 情况 2：真实是 no_relation，但模型预测成了某种 relation（模型误报）
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1  # 模型猜了这个关系，+1

        # 情况 3：真实是某个关系，但模型预测是 no_relation（模型漏报）
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1  # 记录真实的关系出现了，但模型没识别

        # 情况 4：真实是某个关系，模型也猜了某个关系（可能对，也可能错）
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1  # 模型猜了这个关系
            gold_by_relation[gold] += 1      # 真实中也有这个关系
            if gold == guess:
                correct_by_relation[guess] += 1  # 如果预测对了，也记录为正确预测

    # 如果启用了 verbose 模式，就打印每种关系的 P/R/F1
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()  # 所有在真实数据中出现过的关系种类
        longest_relation = 0  # 记录最长的关系名称，用于格式对齐输出

        # 先计算最长的关系名字长度
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)

        # 遍历每个关系，打印其评估指标
        for relation in sorted(relations):
            correct = correct_by_relation[relation]  # 正确预测次数
            guessed = guessed_by_relation[relation]  # 模型预测该关系的总次数
            gold    = gold_by_relation[relation]     # 真实中该关系的出现次数

            # 精确率（precision）：预测是该类中有多少是对的
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)

            # 召回率（recall）：该类真实样本中有多少被正确预测了
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)

            # F1 值（调和平均）
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)

            # 打印该关系的精确率、召回率、F1
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))  # 打印关系名称
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))  # 百分比形式显示精确率
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))  # 百分比形式显示召回率
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))  # 百分比形式显示 F1
            sys.stdout.write("  #: %d" % gold)  # 显示真实样本数
            sys.stdout.write("\n")
        print("")

    # 最后计算整体的 micro 平均指标（不按类别，而是按总体个数计算）
    if verbose:
        print("Final Score:")

    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))

    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))

    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    if verbose:
        print("Precision (micro): {:.2%}".format(prec_micro))
        print("   Recall (micro): {:.2%}".format(recall_micro))
        print("       F1 (micro): {:.2%}".format(f1_micro))

    return prec_micro, recall_micro, f1_micro  # 返回最终结果（用于评估比较）


def predict_with_model(model, val):

    gold = []            # 存储真实标签
    pred_scores = []     # 存储模型预测的每条测试句对所有关系的得分
    pred_relations = []  # 存储每条测试句对应的候选关系标签列表

    num_episodes = int(len(val) / 15)

    for i in range(num_episodes):
        start_loc_of_one_episode = i * 15  # 每个 episode 的起始位置
        end_loc_of_one_episode = start_loc_of_one_episode + 15  # 每个 episode 的结束位置
        one_episode = val[start_loc_of_one_episode:end_loc_of_one_episode]  # 获取第 i 个 episode
        
        # to_be_predicted: 获取所有要送入模型的句子对（支持句，测试句）
        to_be_predicted = []  # 存储所有要送入模型的句子对（支持句，测试句）
        for item in one_episode:
            to_be_predicted.append([
                item['ss_sentence'],
                item['ts_sentence']
            ])
        # gold: 获取真实标签  0 1 2 3 4 have the same ts_relation; 5 6 7 8 9 have the same ts_relation; 10 11 12 13 14 have the same ts_relation
        gold.append(one_episode[0]['ts_relation'])
        gold.append(one_episode[5]['ts_relation'])
        gold.append(one_episode[10]['ts_relation'])

        # pred_relations: 获取候选关系标签
        relations = [
            one_episode[0]['ss_relation'],
            one_episode[1]['ss_relation'],
            one_episode[2]['ss_relation'],
            one_episode[3]['ss_relation'],
            one_episode[4]['ss_relation']
        ]
        pred_relations.append(relations)  # 保存候选关系标签集合
        pred_relations.append(relations)  # 保存候选关系标签集合
        pred_relations.append(relations)  # 保存候选关系标签集合

        pred_scores += expit(
            model.predict(to_be_predicted).reshape(3, 5, -1).mean(axis=2)
        ).tolist()

    return gold, pred_scores, pred_relations  



def train_model(model, **kwargs):
    # 获取训练数据的路径
    training_path = kwargs['training_path']
    # 读取训练数据文件（json 格式，按 relation 分类）
    training_data = []
    with open(training_path, "r") as fin:
        for line in fin:
            item = json.loads(line.strip())
            # 每个 item 是一个字典，包含句子、实体位置、关系等信息
            training_data.append({
                'relation': item['relation'],  # 关系
                'sentence': item['sentence'],  # 句子
            })

    # 构造训练样本对
    data = []
    # 按照指定的样本数采样 finetuning_examples 个训练对
    for i in range(kwargs['finetuning_examples']):
        # 随机选择第一个句子 sent_a
        sent_a = random.choice(training_data)
        # 不允许 sent_a 为 no_relation（见下方注释原因），如果抽到则重新抽
        while sent_a['relation'] == 'no_relation':
            sent_a = random.choice(training_data)
        # 随机选择第二个句子 sent_b（可以为任意 relation，包括 no_relation）
        sent_b = random.choice(training_data)
        # 构造输入样本 InputExample，标签为 1（同类关系）或 0（不同关系），texts 字段传入句子字符串
        data.append(InputExample(texts=[sent_a['sentence'], sent_b['sentence']], label=1 if sent_a['relation'] == sent_b['relation'] else 0))
    print("debug")
    # 用 DataLoader 封装训练数据，打乱顺序，每批 64 个样本
    dataloader = DataLoader(data, shuffle=True, batch_size=64)

    # 用模型的 fit 方法进行训练，训练 1 个 epoch，warmup 步数为样本数的 10%
    model.fit(train_dataloader=dataloader,
            epochs=1,
            warmup_steps=len(data) * 0.1)
    
    # 返回训练后的模型
    return model


def compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds, verbose):
    """
    遍历多个阈值（threshold），计算每个阈值下的预测结果与 gold label 的评估指标
    返回每个阈值下的精确率、召回率、F1 分数等评估结果
    """
    results = []  # 用于存储每个 threshold 下的评估结果

    for threshold in thresholds:  # 遍历所有候选的阈值

        #step#1: 保留 pred_scores 超过阈值 并且pred_scores最大的 pred_relations
        pred = [] # 用于存储每个样本的预测关系
        for ps, pr in zip(pred_scores, pred_relations):  
            if np.max(ps) > threshold: # 如果该样本中最高的预测分数超过 threshold，则判为该分数最大对应的关系
                pred.append(pr[np.argmax(ps)])
            else: # 否则，认为模型没有足够置信度支持任一关系，预测为 "no_relation"
                pred.append('no_relation')

        # step#2: 计算f1 - 用 tacred 的标准评估函数，比较 gold 与 pred，返回 [precision, recall, f1]
        scores = [s * 100 for s in tacred_score(gold, pred, verbose=verbose)]  # 将结果转为百分比制

        # step#3: 构造一个结果字典，包含当前 threshold 下的所有评估指标
        results.append({
            'threshold'            : threshold,  # 当前使用的阈值
            'p_tacred'             : scores[0],  # 精确率（precision）
            'r_tacred'             : scores[1],  # 召回率（recall）
            'f1_tacred'            : scores[2],  # F1 分数（F1 score）
            'f1_macro'             : f1_score(gold, pred, average='macro') * 100,  # 宏平均 F1
            'f1_micro'             : f1_score(gold, pred, average='micro') * 100,  # 微平均 F1
            'f1_micro_withoutnorel': f1_score(
                gold, pred,
                average='macro',
                labels=sorted(list(set(gold).difference(["no_relation"])))
            ) * 100,  # 排除 no_relation 的微平均 F1（更能反映有效关系识别能力）
        })

    return results  # 返回每个 threshold 下的评估结果列表


def main(args):
    """
    The main entry point of this file
    This function is reponsible for:
        - creating (and training) the model
        - computing final results over potentially multiple files with a given threshold; if 
        there is no given threshold, find it on a second file
        - append the results to a file
    """
    # step#1: Create the model
    model = CrossEncoder(args['model_name'], max_length=512)
    
    # step#2: train the model
    if args['do_train']:
        model = train_model(model, training_path=args['training_path'], finetuning_examples=args['finetuning_examples'])
    
    # step#3: choose the threshold
    if args['threshold']: # If threshold is passed, use it
        threshold = [args['threshold']]
    else: # If not, then we have to find it on `find_threshold_on_path`
        if args['find_threshold_on_path']:
            print("Finding the best threshold")
            sampled_val = []
            with open(args['find_threshold_on_path'],"r") as fin:
                for line in fin:
                    sampled_val.append(json.loads(line))

            
            # Compute predictions
            gold, pred_scores, pred_relations = predict_with_model(model, sampled_val)
            # Select best threshold 
            threshold_results_list = compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds=np.linspace(0, 1, 101).tolist(), verbose=False)
            best_threshold = max(threshold_results_list, key=lambda x: x['f1_tacred'])['threshold']
            threshold = best_threshold
            print("Best threshold: ", max(threshold_results_list, key=lambda x: x['f1_tacred'])['threshold'], "with score: ", max(threshold_results_list, key=lambda x: x['f1_tacred'])['f1_tacred'])
            print(threshold_results_list)
        else:
            raise ValueError("No threshold nor file to find it is passed. Is everything ok?")
    
    # step#4: evaluate the model
    results = []
    for evaluation_path in args['evaluation_paths']:
        val = []
        with open(evaluation_path) as fin:
            for line in fin:
                # 读取每一行数据，解析为 JSON 格式，并添加到 val 列表中
                val.append(json.loads(line.strip()))

        gold, pred_scores, pred_relations = predict_with_model(model, val)
        print("###############")
        print('Evaluation Path: ', evaluation_path)
        # [0] -> Results for only one thresholds; 
        # [1] -> Get the result portion (it is a Tuple with (1) -> Threshold and (2) -> Reults)
        scores = compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds=[threshold], verbose=True)[0]
        results.append({'evaluation_path': evaluation_path, **scores})
        print(scores)
        print("###############")

    print("Final results")
    df = pd.DataFrame(results)
    print("P:  ", str(df['p_tacred'].mean())                                 + " +- " + str(df['p_tacred'].std()))
    print("R:  ", str(df['r_tacred'].mean())                                 + " +- " + str(df['r_tacred'].std()))
    print("F1: ", str(df['f1_tacred'].mean())                                + " +- " + str(df['f1_tacred'].std()))
    print("F1: (macro) ", str(df['f1_macro'].mean())                         + " +- " + str(df['f1_macro'].std()))
    print("F1: (micro) ", str(df['f1_micro'].mean())                         + " +- " + str(df['f1_micro'].std()))
    print("F1: (micro) (wo norel) ", str(df['f1_micro_withoutnorel'].mean()) + " +- " + str(df['f1_micro_withoutnorel'].std()))

    if args['append_results_to_file']:
        with open(args['append_results_to_file'], 'a+') as fout:
            for line in results:
                _=fout.write(json.dumps({**line, 'args': args}))
                _=fout.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Sentence-Pair Baseline")
    parser.add_argument('--model_name',             type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2', help="What model to use")
    parser.add_argument('--finetuning_examples',    type=int, default=50000, help="How many examples to fine-tune on")
    parser.add_argument('--do_train', action='store_true', required=False, help="Whether to fine-tune the model or not")
    parser.add_argument('--training_path',          type=str, help="What to train on.")
    parser.add_argument('--evaluation_paths',       type=str, nargs='*', help="What to evaluate on")
    parser.add_argument('--threshold',              type=float, default=None, help="The classification threshold")
    parser.add_argument('--seed',                   type=int, default=1, help="The random seed to use")
    parser.add_argument('--find_threshold_on_path', type=str, help="Finds the best threshold by evaluating on this file")
    parser.add_argument('--append_results_to_file', type=str, help="Appends results to this file")
    args = vars(parser.parse_args())
    print(args)

    seed_everything(args['seed'])

    main(args)

