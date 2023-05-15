# Generative-Adversarial-Federated-Model
This repository contains the implementation of the Generative Adversarial Federated Model (GAFM). The GAFM framework is shown in the following figure.

![pipeline](https://github.com/hyj12345/Generative-Adversarial-Federated-Model/blob/main/pipeline.png)


## Requirements ##
The following packages are required to run the code:

- PyTorch 
- torchvision 
- keras
- numpy 
- pandas 
- sklearn 
- matplotlib

## Data ##

We evaluate GAFM on four publicly available datasets, which can be obtained through the following sources:

- Spabmase: Download from [Spambase](https://archive.ics.uci.edu/ml/datasets/spambase).
- IMDB: Import the data using the following code (details can be found in `GAFM_IMDB/train.py`): `from keras.datasets import imdb` and the `imdb.load_data()` function.  
- Criteo: Download from [Criteo](https://www.kaggle.com/c/criteo-display-ad-challenge).
- ISIC: Download from [ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).

## Usage ##

To train the GAFM model, run `train.py` with the following command:

```linux
sbatch GAFM.sh
```

## Results ##

The model's training progress can be tracked by examining the logs generated during training. Several examples of GAFM training logs are provided in the `training_logs_example.txt`. For more comprehensive results, please refer to the paper.

