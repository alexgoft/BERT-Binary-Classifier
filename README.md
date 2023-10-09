<h1 align = "center"> BERT Binary Text Classification </h1>
<h2 align="center"> 🚧 Under construction 🚧 </h2>
To install the dependencies, run the following commands:

```bash
conda update -n base -c defaults conda
conda create -n bert python=3.8
conda activate bert
conda install -c "nvidia/label/cuda-12.0.0" cuda
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers pandas matplotlib scikit-learn seaborn nltk pyyaml 
```