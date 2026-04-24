# Binding shift prediction of SARS-CoV-2
Please download the complete dataset, model, and code in the Code Ocean (https://codeocean.com/capsule/0361003/tree) before running the following codes.
This repository provides a complete pipeline for fine-tuning the ESM-2 protein language model, extracting embeddings for coronavirus sequences (SARS-CoV-2, SARS-CoV, MERS-CoV, and pseudoviruses), evaluating embedding performance, and training a predictive model.

Before running the code, please extract the datasets in the 'data' folder. For the multi-part model archives in the 'predictor' folder, please ensure you have downloaded all parts into the same directory, then simply extract the first file (the 01 file), and the rest will be extracted automatically.

## Prerequisites

Before running the pipeline, please download the **ESM-2 650M model** (https://huggingface.co/facebook/esm2_t33_650M_UR50D) and place it in the main directory of this repository.

------

## 1. ESM-2 Model Fine-Tuning

Run the following scripts in the main directory to fine-tune the pre-trained ESM-2 model using specific viral sequences.

Bash

```
# Fine-tune ESM-2 using MERS-CoV and SARS-CoV sequences
python esm2_finetune3_SpikeS1MERSSARS_epoch200.py

# Fine-tune ESM-2 using SARS-CoV-2 sequences
python esm2_finetune3_SpikeS1SARS2_epoch200.py
```

------

## 2. Sequence Embedding Extraction

Use the following scripts to extract sequence embeddings based on the fine-tuned or wild-type ESM-2 models.

### Pseudovirus Embeddings

Extract pseudovirus sequence embeddings using the model fine-tuned on MERS-CoV and SARS-CoV:

Bash

```
python extract_embedding_S1psudoviruses_esm2SARSMERS.py
```

### SARS-CoV-2 Sequence Embeddings

Extract SARS-CoV-2 embeddings using different versions of the ESM-2 model:

Bash

```
# Based on ESM-2 fine-tuned with MERS-CoV and SARS-CoV
python extract_embedding_SARS2S1_esm2MERSSARS.py

# Based on ESM-2 fine-tuned with SARS-CoV-2
python extract_embedding_SARS2S1_esm2SARS2.py

# Based on the wild-type ESM-2 model
python extract_embedding_SARS2S1_esm2wt.py
```

### MERS-CoV and SARS-CoV Sequence Embeddings

Extract MERS-CoV and SARS-CoV embeddings:

Bash

```
# Based on ESM-2 fine-tuned with MERS-CoV and SARS-CoV
python extract_embedding_SARSMERS_esm2SARSMERS.py

# Based on ESM-2 fine-tuned with SARS-CoV-2 (Part 1 & 2)
python extract_embedding_SARSMERS_esm2SARSS1.py
python extract_embedding_SARSMERS_esm2SARSS1_part2.py
```

------

## 3. Embedding Evaluation (Clustering and PCA)

Embedding evaluations are located in the `embedding` directory, specifically within the following subfolders:

- `embedding_SARS2_esm2MERSSARS`
- `embedding_SARS2_esm2SARS2`
- `embedding_SARSMERS_esm2MERSSARS`

These scripts perform dimensionality reduction (PCA with varying dimensions), clustering, and calculation of clustering metrics. They support different classification labels:

- `label`: 3-class classification
- `label1`: 5-class classification

Navigate to the respective target directory and run the evaluation scripts as needed:

Bash

```
# Example commands for running embedding evaluations
python for_cluster_index_random_withoutPCA2_label.py
python for_cluster_index_random_withoutPCA2_label1.py
python for_cluster_index_random_withPCA2_label.py
python for_cluster_index_random_withPCA2_label1.py
```

------

## 4. Prediction Model (BindingResNet)
Please first copy the `checkpoint-600` files from the `embedding/embedding_SARSMERS_esm2MERSSARS` and `embedding/embedding_SARS2_esm2MERSSARS` directories to the `data` folder. Then, rename them to `checkpoint-600_SARSMERSembedding_SARSMERSesm2.pkl` and `checkpoint-600_SARS2S1embedding_SARSMERSesm2.pkl`, respectively.

All modeling scripts are located in the `predictor` folder. This module handles data loading, model training, and final predictions.

### Step 4.1: Data Loading

Load the respective sequence data for training.

Bash

```
cd predictor

# Load SARS-CoV-2 data
python data_load_SARS2S1_SARSMERSesm2.py

# Load MERS-CoV and SARS-CoV data
python data_load_SARSMERSS1_SARSMERSesm2.py
```

### Step 4.2: Model Training

Train the BindingResNet models.

Bash

```
# Train on SARS-CoV-2 sequences
python bindingResNet_trainer_SARS2S1_SARSMERSesm2.py

# Train on MERS-CoV and SARS-CoV sequences
python bindingResNet_trainer_SARSMERSS1_SARSMERSesm2.py
```

### Step 4.3: Model Prediction

Run the final prediction script to evaluate the trained model.

Bash

```
python bindingResNet_predictor.py
```
