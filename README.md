# Prior latent distribution comparison for VAE

Code for paper PAPER NAME

# Installation
Install requirements from requirements.txt
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


