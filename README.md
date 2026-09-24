# DSRL: Rethinking Cross-Encoder Representations for Conditional Semantic Textual Similarity

## Table of Contents

* [DSRL](#dsrl)

  * [Requirements](#requirements)
  * [Data](#data)
  * [Training](#training)
  * [Results](#results)

## DSRL <a name="dsrl"></a>

This repository contains the implementation of **DSRL (Dual-Space Representation Learning Framework)** for Conditional Semantic Textual Similarity (C-STS).

DSRL is designed for cross-encoder models and explicitly models and optimizes the textual and conditional textual representation spaces. It jointly incorporates a **Uniformity Contrastive Loss (UCL)** to promote uniform textual representations, a **Rank Consistency Loss (RCL)** to shape inter-sample relations among conditional sentence representations, and **MSE** to provide intra-sample supervision in the conditional textual representation space.

This code is based on the official [C-STS](https://github.com/princeton-nlp/c-sts/tree/main) implementation.

### Requirements <a name="requirements"></a>

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### Data <a name="data"></a>

Download the C-STS dataset and place the corresponding files under the `data/` directory. Please refer to the official [C-STS repository](https://github.com/princeton-nlp/c-sts/tree/main) for details on dataset preparation.

### Training <a name="training"></a>

DSRL can be trained by running:

```bash
python single_run.py
```

The hyperparameters are configured according to the settings reported in the paper.

### Results <a name="results"></a>

The `output/` directory contains the final predictions of DSRL based on **SimCSE-base** on the C-STS test set. These predictions can be used to verify the reported test performance.

The official evaluation script can be run as follows:

```bash
python make_test_submission.py your_email@email.com /path/to/your/test_predictions.json
```
