# TACRED Relation Extraction via Sentence Pair Classification

This project implements a sentence-pair modeling approach to tackle the relation extraction task on the TACRED dataset, using `CrossEncoder` models from the `sentence-transformers` library.

## 🧠 Task Overview

The TACRED dataset is a large-scale relation extraction benchmark with 42 labeled relations between subject and object entities within sentences. Traditional modeling formulates this task as a multi-class classification problem.

In this project, we explore an **episodic, pairwise strategy** inspired by meta-learning:

- For each test sentence (with marked subject and object),
- We compare it with multiple support sentences, each representing a different relation,
- The model evaluates each support–test sentence pair and predicts a similarity score,
- The predicted relation is the one with the highest score, if above a threshold; otherwise, it is labeled `no_relation`.

This reframes relation extraction as a **similarity-based retrieval** or **matching** task.

## 🧪 Modeling Details

We use `CrossEncoder` models (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) and fine-tune them using sentence pairs labeled as same-relation (1) or different-relation (0).

### 🧱 Input Format

Each pair consists of:

- Sentence A: A support example with a known relation
- Sentence B: A test example
- Both sentences are preprocessed with entity markers (e.g., `<e1>`, `<e2>`)

### 📂 Episodic Evaluation

- During inference, each test sentence is paired with all available support sentences from a fixed set of known relations.
- For each test-support pair, the CrossEncoder outputs a similarity score.
- Scores are aggregated (mean-pooling over multiple supports per relation) and the best-matching relation is chosen, conditioned on a learned threshold.

## 📈 Metrics and Evaluation

We compute:

- **Micro-F1 / Macro-F1** with and without the `no_relation` class
- **TACRED-style precision, recall, and F1**
- Optimal threshold selection on a validation split

Example evaluation output includes:

```json
{
  "threshold": 0.71,
  "f1_tacred": 72.30,
  "f1_micro": 71.56,
  "f1_macro": 63.87
}

```

