# Prior latent distribution comparison for VAE

Code for paper PAPER NAME

# Installation
This a dockerized version of the repo.
Just run the docker build there.

<br> Run loading spacy model: 
```bash
python -m spacy download en
```
<br> Download file from: <a href="https://www.kaggle.com/gpreda/covid19-tweets">kaggle</a>

<br> Run data preparation file. It is located in data_utils/ with command: 

```bash

python data_utils/prepare_data.py -save_dir <PATH TO SAVE DIR> -load_path <PATH TO CSV FILE> -config_path <PATH TO preparation config>

```
Script should create a directory in specified path with required files for training

# Training 
Training is running with command train.py file with commad line interface. Available parameters: 

`-data_path`  The path to the directory, where prepared data and all required files are saved (save_dir for data_utils/prepare_data.py).
<br>`-logdir` Path to dir, where model, logs and checkpoints will be saved.  
<br>`-resume_path` Optional, path to checkpoint to continue training. 
<br>`-model_params_path` Path to model config json file. Example: configs/cauchy_model_config.json. 

<br>`-epochs` Number of epochs to train.
<br>`-batch_size` Batch size to use for training.   
<br>`-lr` Learning rate to use. 
<br>`-loss_reduction` "sum" or "mean" - loss reduction parameter for torch NLLLoss.
<br>`-device` "cuda" or "cpu". Device to use. 
<br>`-min_len`, `-max_len` Min and Max length of sequences. Data will be filtered by length. 

<br>`-weight_kld_k`, `-weight_kld_x` Weights to use for KLD weighting. Check paper for more details. 
<br>`-scheduler_factor` Scheduler facter. Parameter factor in torch.optim.lr_scheduler.ReduceLROnPlateau. 

<br>`-mask_dropout`  Probability of masking token. Each token in training data will be replaced with this probability. 
<br>`-embedding_mask_dropout` Dropout to use on embeddings. 
<br>`-select_ratio` Ratio of tokens to be selected to update weights on. 


