<h1 align = "center"> BERT Binary Text Classification </h1>
<h2 align="center"> 🚧 Under construction 🚧 </h2>

# Overview


## Install dependencies

To install the dependencies, run the following commands:


```bash
conda update -n base -c defaults conda
conda create -n bert python=3.8
conda activate bert
conda install -c "nvidia/label/cuda-12.0.0" cuda
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers pandas matplotlib scikit-learn seaborn nltk pyyaml 
```

## Project Structure
```
BERT-Binary-Text-Classification/
│
├── data
│   └── data_csv.csv          <- data file with text and labels
├── output                    <- trained models are saved here
│   └── 20231009-094932       <- output folder with timestamp
|       └── config.yaml       <- config file of run
|       └── model_0.45774.pt  <- model checkpoint with best validation accuracy
├── configs                   <- config files for different runs
│       └── config.yaml       <- config file with hyperparameters of
├── config_file.py            <- Configuration class
├── text_dataset.py
├── data_utils.py             <- Data preprocessing and tokenization
├── main.py                   <- main file for training and testing
├── train_utils.py            <- early stopping and samplers are defined here
└── model.py                  <- model architecture
```

## Config file structure
```yaml
general:
  mode: "train"  # 'test' or 'train'
  output_dir: "outputs"
  seed: 42

data:
  data_path: "data/assignment_data_en.csv"
  plot_histograms: true  # Plot histograms of train, val and test sets.
  # Percentage of the data_utils left for validation (rest is for test).
  # If 0.6, 60% of the data_utils is used for training, 20% for validation and 20% for test.
  train_size: 0.6
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
  linear_layers_num: 1  # Number of linear layers after the BERT model.
  # If n_classes > 1, one-hot encoding is used.
  # else integer encoding is used.
  n_classes: 1
  max_seq_length: 512

train:
  num_epochs: 7
  batch_size: 4
  dropout: 0.3
  early_stopping:
    min_delta: 0 # Minimum change in the monitored quantity to qualify as an improvement.
    patience: 3  # Number of epochs with no improvement after which training will be stopped.
  eps: 1.0e-08
  lr: 1.0e-05
  weight_decay: 0.01
  # Samplers are used to specify how to sample from the dataset.
  # "WeightedRandomSampler" - samples with probability proportional to class weights.
  # "BalancedBatchSampler" - samples batches with equal number of samples from each class.
  # None - samples randomly from the dataset.
  sampler: null
test:
  model_path: "outputs/20231007-193004/model_0.43864.pt"
  threshold: 0.5  # Threshold for positive class.
```