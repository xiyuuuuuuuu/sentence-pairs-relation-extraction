# TACRED Relation Extraction via Sentence Pair Classification

This project implements a sentence-pair modeling approach to tackle the relation extraction task on the TACRED dataset, using `CrossEncoder` models from the `sentence-transformers` library.



## 🧠 Task Overview

**Relation extraction** is a fundamental task in text analysis and a cornerstone for building knowledge graphs, which in turn support many downstream applications. Given its importance and the need for efficiency, lightweight baseline models with simple architectures are well-suited for this task. In this project, I aim to explore methods, frameworks, and techniques to enhance such models. Specifically, I focus on improving a **sentence-pair classification** baseline for relation extraction. 

*As part of this course project, I only present a reproduction of the sentence-pair baseline.*



## 📟 Modeling Details

Architecture Choice: 

* Sentence Pair Relation Extraction

For each test sentence,   it is paired with multiple (for TACRED dataset is five) support sentences, and the `CrossEncoder` model is used to:

- Compute the similarity between the test sentence and each support sentence.

Then, code will use the relation of the support sentence with the highest similarity with test sentence as the relation of this test sentence.

**Illustration of the underlying principle：**

<img src="assets/modeling.png" style="zoom:48%;" />





## 📂  Dataset Selection & Preprocessing

Dataset: One-shot TACRED

- We select the **one-shot TACRED** dataset as the training and evaluation resource.

- About the **TACRED**:

  The original TACRED dataset (https://nlp.stanford.edu/projects/tacred/).

  The few-shot TACRED dataset (https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00392/106791/Revisiting-Few-shot-Relation-Classification)).

Preprocessing under Sentence-Pair Framework: 

* Train:

  Only process all the (relation and tokens) units from "meta_train" as the training dataset. 

  (1) Using `typed_entity_marker_punct` to get <u>the sentence with entity markers</u>:

  * Replace the subject and object in the tokens with the entity marker like \# ^ date ^ 2003 #. 
  * Transform the tokens format into the sentence format. and return the formed sentence.

  (2) Using the <u>relation</u> from each unit directly.

  Combining (1) and (2), each episode has 5 (<u>relation</u>, <u>the sentence with entity markers</u>) pairs.

  While training the model, randomly sample two unit, if they have the same relation, give a label of 1. If they have different relation, give a label of 0.

  <u>Meta data:</u>

  ```json
  {
    "sentence": "In high school and at Southern Methodist University , where , already known as # ^ individual ^ Dandy Don # ( a nickname bestowed on him by his brother ) , @ * individual * Meredith @ became an all-American .",
    "relation": "per:alternate_names"
  }
  ```

* Val:

  One episode contains five (support sentence, relation) units and three  (test sentence, relation) units. 
  I combined each  (test sentence, relation) unit with all five (support sentence, relation) units. The relations of the five sentences all are the candidate relation for this test sentence. 
  In this way, for one episode, we generate 15 ((test sentence, ts_relation), (support sentence, ss_relation), "Yes" if  ts_relation==ss_relation else "No") pair. 

  <u>Meta data:</u>

  ```json
  {
    "ts_sentence": "@ * firm * Escada @ , which employs around # ^ number ^ 2,300 # people worldwide , was forced to file for insolvency in mid August .",
    "ss_sentence": "In high school and at Southern Methodist University , where , already known as # ^ individual ^ Dandy Don # ( a nickname bestowed on him by his brother ) , @ * individual * Meredith @ became an all-American .",
    "ts_relation": "no_relation",
    "ss_relation": "per:alternate_names"
  }
  ```




## 📈 Results

under the file path: results/tacred_1shot.jsonl

Summary:

```json
{
	"p_tacred": 5.9 ± 0.72,
	"r_tacred": 12.86 ± 1.52,
	"f1_tacred": 8.09 ± 0.97
}
```



## ⚙️ Environment

All the requirements packages are listed in the file: requirement.txt .

You can create the `python` env with `conda` using the following commands.

```bash
conda create -n sentpair python=3.9 -y
conda activate sentpair

pip install numpy pandas scikit-learn tqdm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Lightning
pip install pytorch-lightning

# Sentence Transformers（包含 CrossEncoder、InputExample 等）
pip install sentence-transformers

# SciPy
pip install scipy

pip install datasets

pip install -U 'accelerate>=0.26.0'
```



## 🖥️ Codebase Overview

```bash
.
├── generate_data_code											# generate train and val dataset code
│   ├── entity_marker_with_reg.py					
│   ├── generate_train_val_data_tacred.py	
│   ├── generate_train_val_data_tacred.sh		# modify the arguements here and run it
│   └── __pycache__
│       ├── entity_marker_with_reg.cpython-313.pyc
│       └── entity_marker_with_reg.cpython-39.pyc
├── generated_datasets
│   └── dev_episodes
│       ├── train
│       │   ├── 5_way_1_shots_10K_episodes_3q_seed_160290.jsonl
│       │   ├── 5_way_1_shots_10K_episodes_3q_seed_160291.jsonl
│       │   ├── 5_way_1_shots_10K_episodes_3q_seed_160292.jsonl
│       │   ├── 5_way_1_shots_10K_episodes_3q_seed_160293.jsonl
│       │   ├── 5_way_1_shots_10K_episodes_3q_seed_160294.jsonl
│       │   └── all_train.jsonl							# dataset used to train
│       └── val
│           ├── 5_way_1_shots_10K_episodes_3q_seed_160290.jsonl	# dataset used to evaluate
│           ├── 5_way_1_shots_10K_episodes_3q_seed_160291.jsonl # dataset used to evaluate
│           ├── 5_way_1_shots_10K_episodes_3q_seed_160292.jsonl # dataset used to evaluate
│           ├── 5_way_1_shots_10K_episodes_3q_seed_160293.jsonl # dataset used to evaluate
│           └── 5_way_1_shots_10K_episodes_3q_seed_160294.jsonl # dataset used to evaluate
├── log
│   ├── 0506.log												# nohup training log
│   └── 20250506_2016_seedsall_k1.txt		# bash file printing training log
├── requirement.txt											# environments
├── results								
│   └── tacred_1shot.jsonl							# results
├── scripts
│   └── run.sh													# bash file to run sentence_pair.py  
└── src
    └── sentence_pair.py								# main code containing training, evaluating model
```



