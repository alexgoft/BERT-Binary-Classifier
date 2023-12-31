<h1 align = "center"> BERT Binary Text Classification </h1>
<h2 align="center"> 🚧 Under construction 🚧 </h2>

# Overview
This repository contains a BERT model for binary text classification. 
The model is implemented in PyTorch and uses the Hugging Face transformers.

## Install Dependencies

To install the dependencies, run the following commands:


```bash
conda update -n base -c defaults conda
conda create -n bert python=3.8
conda activate bert
conda install -c "nvidia/label/cuda-12.0.0" cuda
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers pandas matplotlib scikit-learn seaborn nltk pyyaml 
```



## Config File
```yaml
data:
  data_path: "data/assignment_data_en.csv"
  data_class: # The second is the positive class.
    - "not-news"
    - "news"
  class_column: "content_type" # Column name of the class.
  plot_histograms: true  # Plot histograms of train, val and test sets.
  # Percentage of the data_utils left for validation (rest is for test).
  # If 0.7, 70% of the data_utils is for train, 15% for val and 15% for test.
  train_size: 0.7
  val_size: 0.5

  # If not null, split the text into segments of max_seq_length
  # with overlap of overlap_size words/token. Only for train DF.
  split_text: null
#  split_text:
#    overlap_size: 50

model:
  # BERT versions: "bert-base-uncased", "bert-base-cased"
  # Smaller versions: "prajjwal1/bert-tiny", "prajjwal1/bert-mini", "prajjwal1/bert-small"
  model_name: "bert-base-uncased"
  uncased: true  # Tokenizer parameter. Bert uncased or cased (case-sensitive)
  freeze_bert: false  # If True, only train the classifier layers.
  linear_layers_num: 1 # Number of linear layers after the BERT model.
  n_classes: 2  # TODO: Support more than 2 classes.
  max_seq_length: 512  # Max sequence length for BERT model.

train:
  num_epochs: 10
  batch_size: 4
  dropout: 0.4
  early_stopping:
    min_delta: 0 # Minimum change in the monitored quantity to qualify as an improvement.
    patience: 2  # Number of epochs with no improvement after which training will be stopped.
  eps: 1.0e-08
  lr: 1.0e-05
  weight_decay: 0.01
  # Samplers are used to specify how to sample from the dataset.
  # "WeightedRandomSampler" - samples with probability proportional to class weights.
  # "BalancedBatchSampler" - samples batches with equal number of samples from each class.
  # null - samples randomly from the dataset.
  sampler: null
```


## Outputs

During train mode, the output directory is named with the timestamp of the run.
The output directory contains the following files:
- `config.yaml` - The configuration file.
- `model_{val_acc}.pt` - The models with validation accuracy.
- `train_val_loss.png` - Train and validation loss plot.
- Histograms of train, val and test sets (if `plot_histograms` is true).

During test mode, the test metrics are saved in a directory with the same name as the model with the suffix `_metrics`. 
The directory contains the following files:
- `classification_report.txt` - classification report (precision, recall, f1-score, etc)
- `confusion_matrix.png` - confusion matrix
- `roc_curve.png` - ROC curve with AUC score.

## Project Structure
```
BERT-Text-Classifier\
ª
ª           
+---configs
ª       config.yaml
ª       
+---outputs
ª   +---20231012-121624/
ª   ª   ª   config.yaml
ª   ª   +---model_0.49529.pt
ª   ª   +---data_plots/
ª   ª           dataset_histogram.png
ª   ª           dataset_histogram_(multi-class).png
ª   ª           test_histogram.png
ª   ª           train_histogram.png
ª   ª           train_word_count_histogram.png
ª   ª           val_histogram.png
ª   ª   +---model_0_49529_metrics/
ª   ª           classification_report.txt
ª   ª           confusion_matrix.png
ª   ª           roc_curve.png
ª   ª
ª   main.py
ª   model.py
ª   plot_utils.py
ª   test_utils.py
ª   train_utils.py
ª   data_utils.py
ª   config_file.py
```