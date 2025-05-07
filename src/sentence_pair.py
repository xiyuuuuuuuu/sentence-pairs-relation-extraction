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
    This is a scoring function for the TACRED task,
    Inputs:
        key: list of true labels (gold labels) for each sample
        prediction: list of model predicted labels (predicted relations) for each sample
        verbose: whether to print detailed evaluation information for each relation (True will print precision, recall, F1 for each relation)

    Returns:
        Precision (micro average), Recall (micro average), F1 (micro average)
    """

    # Initialize three counters (dictionaries):
    # correct_by_relation: counts correct predictions for each relation
    # guessed_by_relation: counts how many times the model predicted each relation (regardless of correctness)
    # gold_by_relation: counts how many times each relation appears in the gold data
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    # Iterate over all samples, comparing gold and predicted labels by index row
    for row in range(len(key)):
        gold = key[row]           # Gold label of sample at row
        guess = prediction[row]   # Predicted label of sample at row

        # Case 1: Both gold and predicted labels are "no_relation" (no relation); this case is excluded from metrics.
        if gold == NO_RELATION and guess == NO_RELATION:
            pass  # Do nothing, skip

        # Case 2: Gold is "no_relation" but model predicted some relation (false positive)
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1  # Model guessed this relation, +1

        # Case 3: Gold is some relation but model predicted "no_relation" (false negative)
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1  # Record that gold relation appeared but model missed it

        # Case 4: Gold is some relation and model predicted some relation (may be correct or incorrect)
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1  # Model guessed this relation
            gold_by_relation[gold] += 1      # Gold relation also appeared
            if gold == guess:
                correct_by_relation[guess] += 1  # If prediction is correct, count as correct

    # If verbose mode is enabled, print precision/recall/F1 for each relation
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()  # All relations that appear in gold data
        longest_relation = 0  # Track longest relation name for formatting output

        # Compute longest relation name length first
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)

        # For each relation, print its evaluation metrics
        for relation in sorted(relations):
            correct = correct_by_relation[relation]  # Correct prediction count
            guessed = guessed_by_relation[relation]  # Total predicted count for relation
            gold    = gold_by_relation[relation]     # Gold count for relation

            # Precision: how many predictions of this class are correct
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)

            # Recall: how many gold samples of this class are correctly predicted
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)

            # F1 score (harmonic mean)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)

            # Print precision, recall, F1 for this relation
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))  # Print relation name
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))  # Show precision as percentage
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))  # Show recall as percentage
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))  # Show F1 as percentage
            sys.stdout.write("  #: %d" % gold)  # Show gold sample count
            sys.stdout.write("\n")
        print("")

    # Finally compute overall micro-average metrics (not by class, but total counts)
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

    return prec_micro, recall_micro, f1_micro  # Return final results (for evaluation/comparison)


def predict_with_model(model, val):

    gold = []            # Store true labels
    pred_scores = []     # Store model predicted scores for all relations per test sentence pair
    pred_relations = []  # Store candidate relation label list for each test sentence

    num_episodes = int(len(val) / 15)

    for i in range(num_episodes):
        start_loc_of_one_episode = i * 15  # Start position of each episode
        end_loc_of_one_episode = start_loc_of_one_episode + 15  # End position of each episode
        one_episode = val[start_loc_of_one_episode:end_loc_of_one_episode]  # Get the i-th episode
        
        # to_be_predicted: get all sentence pairs (support sentence, test sentence) to feed into model
        to_be_predicted = []  # Store all sentence pairs (support sentence, test sentence) to feed into model
        for item in one_episode:
            to_be_predicted.append([
                item['ss_sentence'],
                item['ts_sentence']
            ])
        # gold: get true labels; 0 1 2 3 4 have the same ts_relation; 5 6 7 8 9 have the same ts_relation; 10 11 12 13 14 have the same ts_relation
        gold.append(one_episode[0]['ts_relation'])
        gold.append(one_episode[5]['ts_relation'])
        gold.append(one_episode[10]['ts_relation'])

        # pred_relations: get candidate relation labels
        relations = [
            one_episode[0]['ss_relation'],
            one_episode[1]['ss_relation'],
            one_episode[2]['ss_relation'],
            one_episode[3]['ss_relation'],
            one_episode[4]['ss_relation']
        ]
        pred_relations.append(relations)  # Save candidate relation label set
        pred_relations.append(relations)  # Save candidate relation label set
        pred_relations.append(relations)  # Save candidate relation label set

        pred_scores += expit(
            model.predict(to_be_predicted).reshape(3, 5, -1).mean(axis=2)
        ).tolist()

    return gold, pred_scores, pred_relations  



def train_model(model, **kwargs):
    # Get training data path
    training_path = kwargs['training_path']
    # Read training data file (json format, classified by relation)
    training_data = []
    with open(training_path, "r") as fin:
        for line in fin:
            item = json.loads(line.strip())
            # Each item is a dictionary containing sentence, entity positions, relation, etc.
            training_data.append({
                'relation': item['relation'],  # Relation
                'sentence': item['sentence'],  # Sentence
            })

    # Construct training sample pairs
    data = []
    # Sample finetuning_examples number of training pairs as specified
    for i in range(kwargs['finetuning_examples']):
        # Randomly select first sentence sent_a
        sent_a = random.choice(training_data)
        # Do not allow sent_a to be no_relation (see comment below for reason), if selected then resample
        while sent_a['relation'] == 'no_relation':
            sent_a = random.choice(training_data)
        # Randomly select second sentence sent_b (can be any relation including no_relation)
        sent_b = random.choice(training_data)
        # Construct input sample InputExample, label is 1 (same relation) or 0 (different relation), texts field passes sentence strings
        data.append(InputExample(texts=[sent_a['sentence'], sent_b['sentence']], label=1 if sent_a['relation'] == sent_b['relation'] else 0))
    print("debug")
    # Wrap training data with DataLoader, shuffle order, batch size 64
    dataloader = DataLoader(data, shuffle=True, batch_size=64)

    # Train model using fit method, 1 epoch, warmup steps 10% of sample count
    model.fit(train_dataloader=dataloader,
            epochs=1,
            warmup_steps=len(data) * 0.1)
    
    # Return trained model
    return model


def compute_results_with_thresholds(gold, pred_scores, pred_relations, thresholds, verbose):
    """
    Iterate over multiple thresholds, compute evaluation metrics comparing predicted results with gold labels for each threshold
    Return precision, recall, F1 scores, and other evaluation results for each threshold
    """
    results = []  # Store evaluation results for each threshold

    for threshold in thresholds:  # Iterate over all candidate thresholds

        #step#1: Keep pred_scores exceeding threshold and pred_scores max pred_relations
        pred = [] # Store predicted relation for each sample
        for ps, pr in zip(pred_scores, pred_relations):  
            if np.max(ps) > threshold: # If max predicted score in sample exceeds threshold, predict relation with max score
                pred.append(pr[np.argmax(ps)])
            else: # Otherwise, model confidence is insufficient to support any relation, predict "no_relation"
                pred.append('no_relation')

        # step#2: Calculate f1 - use standard TACRED evaluation function, compare gold and pred, return [precision, recall, f1]
        scores = [s * 100 for s in tacred_score(gold, pred, verbose=verbose)]  # Convert results to percentage

        # step#3: Construct result dictionary containing all evaluation metrics for current threshold
        results.append({
            'threshold'            : threshold,  # Current threshold used
            'p_tacred'             : scores[0],  # Precision
            'r_tacred'             : scores[1],  # Recall
            'f1_tacred'            : scores[2],  # F1 score
            'f1_macro'             : f1_score(gold, pred, average='macro') * 100,  # Macro average F1
            'f1_micro'             : f1_score(gold, pred, average='micro') * 100,  # Micro average F1
            'f1_micro_withoutnorel': f1_score(
                gold, pred,
                average='macro',
                labels=sorted(list(set(gold).difference(["no_relation"])))
            ) * 100,  # Micro average F1 excluding no_relation (better reflects effective relation recognition ability)
        })

    return results  # Return list of evaluation results for each threshold


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
                # Read each line, parse as JSON, and add to val list
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
