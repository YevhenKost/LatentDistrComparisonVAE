import os
import sys

# adding current path for importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch, itertools

from archs.vae import VAE
from runners import ClassificationVAERunner
from losses import NLLLoss_criterion
from catalyst.core.callbacks import CriterionCallback, SchedulerCallback
from data_utils.data_loading import DataLoaders, read_sents
from torch.optim import Adam
from callbacks import CustomAccuracyCallback, KLDCallback, \
    VizDecodeCallback, SigmoidWeightKLDCallback, NLLLossCallback
import catalyst, json
from collections import OrderedDict
from encoders import OneHotEncoder
from utils import _fix_seeds


def train(params_grid):

    # fixing random seeds
    _fix_seeds()

    # setting training parameters
    NUM_EPOCHS = params_grid["epochs"]
    BATCH_SIZE = params_grid["batch_size"]
    LR = params_grid["lr"]
    DEVICE = params_grid["device"]

    MAX_LEN = params_grid["max_len"]
    MIN_LEN = params_grid["min_len"]
    LOSS_REDUCTION=params_grid["loss_reduction"]

    dir_path = params_grid["data_path"]
    logdir = params_grid["logdir"]

    # loading embedding vocabularies
    embedding_path = os.path.join(dir_path, "encoding_dict.json")
    enc_dict = json.load(open(embedding_path, "r", encoding="utf-8"))

    preprocess_config_path = os.path.join(dir_path, "config.json")
    preprocess_config = json.load(open(preprocess_config_path, "r"))

    test_drp = preprocess_config["valid_dropout_rate"]
    test_col = f"dropout_tokenized_sents_{test_drp}"

    MASK = preprocess_config["mask_token"]
    PAD = preprocess_config["pad_token"]
    END = preprocess_config["end_seq_token"]
    START = preprocess_config["start_seq_token"]

    # building model arch
    arch_config = json.load(
        open(params_grid["model_params_path"], "r")
    )
    arch_config["embeddding_dropout_rate"] = params_grid["embedding_mask_dropout"]
    arch_config["embedding_config"]["num_embeddings"] = len(enc_dict)
    arch_config["embedding_config"]["padding_idx"] = enc_dict[PAD]
    arch_config["start_idx"] = enc_dict[START]
    arch_config["mask_idx"] = enc_dict[MASK]
    arch_config["rnn_encoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]
    arch_config["rnn_decoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]


    TYPE_DISTR = arch_config["type_distr"]


    # loading encoder
    MASK_DROPOUT = params_grid["mask_dropout"]

    encoder = OneHotEncoder(
        enc_dict=enc_dict,
        max_len=MAX_LEN,
        mask_idx=MASK,
        pad_tok=PAD,
        end_tok=END,
        start_token=START,
        input_dropout=MASK_DROPOUT
    )



    # loading data
    train_dataset_config = {
        "sents": read_sents(os.path.join(dir_path, "train.json"),
                            min_len=MIN_LEN, max_len=MAX_LEN,
                            col="tokenized_sents",verbose=True),
        "encoder": encoder,
        "use_token_dropout": True
    }

    valid_dataset_config = {
        "tokenized_sents": read_sents(os.path.join(dir_path, "valid.json"),
                                      min_len=MIN_LEN, max_len=MAX_LEN,
                                      col="tokenized_sents", verbose=True),
        "tokenized_dropout_sents": read_sents(os.path.join(dir_path, "valid.json"),
                                              min_len=MIN_LEN, max_len=MAX_LEN,
                                              col=test_col),
        "encoder": encoder
    }

    test_dataset_config = {
        "tokenized_sents": read_sents(os.path.join(dir_path, "test.json"),
                                      min_len=MIN_LEN, max_len=MAX_LEN,
                                      col="tokenized_sents", verbose=True),
        "tokenized_dropout_sents": read_sents(os.path.join(dir_path, "test.json"),
                                              min_len=MIN_LEN, max_len=MAX_LEN,
                                              col=test_col),
        "encoder": encoder
    }



    loaders = {
        "train": DataLoaders.get_loader(
            dataset_config=train_dataset_config,
            batch_size=BATCH_SIZE,
            is_train=True,
            select_ratio=params_grid["select_ratio"]

    ),
        "valid": DataLoaders.get_loader(
            dataset_config=valid_dataset_config,
            batch_size=BATCH_SIZE,
            is_train=False,
            select_ratio=1
        ),

        "test": DataLoaders.get_loader(
            dataset_config=test_dataset_config,
            batch_size=BATCH_SIZE,
            is_train=False,
            select_ratio=1
        )
    }




    # init model
    model = VAE(**arch_config)
    model = model.to(DEVICE)

    # init loss
    criterion = NLLLoss_criterion(distr_type=TYPE_DISTR, reduction=LOSS_REDUCTION)
    criterion = criterion.to(DEVICE)

    # setting callbacks in order
    callbacks = OrderedDict({

        "kld_weight": SigmoidWeightKLDCallback(
            k=params_grid["weight_kld_k"],
            x=params_grid["weight_kld_x"]
        ),

        "criterion": CriterionCallback(
            input_key="true_selected_indexes",
            output_key="dist_loss_select",
            prefix="loss",
        ),

        "full_criterion": CriterionCallback(
            input_key="true_unpadded_indexes",
            output_key="dist_loss_unpad",
            prefix="full_loss",
        ),

        "optimizer": catalyst.dl.OptimizerCallback(
                metric_key="loss",
                accumulation_steps=1,
                grad_clip_params=None
            ),

        "nlll": NLLLossCallback(
            input_key="true_unpadded_indexes",
            output_key="unpad_preds",
            reduction=LOSS_REDUCTION,
            prefix="nlll"
        ),

        "pad_accuracy": CustomAccuracyCallback(
            input_key="true_padded_indexes",
            output_key="pred_padded_indexes",
            prefix="pad_accuracy"
        ),

        "unpad_accuracy": CustomAccuracyCallback(
            input_key="true_unpadded_indexes",
            output_key="pred_unpadded_indexes",
            prefix="unpad_accuracy"
        ),

        "mask_accuracy": CustomAccuracyCallback(
            input_key="masked_indexes",
            output_key="masked_preds",
            prefix="mask_accuracy"
        ),

        "kld": KLDCallback(
            distr_type=TYPE_DISTR
        ),

        "viz_decode": VizDecodeCallback(
                input_key="visualize_tokens",
                output_key="pred_scores",
                encoder=encoder,
                max_num=5
            ),

        "scheduler": SchedulerCallback(
            reduced_metric="loss"
        )
    })

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=params_grid["scheduler_factor"],
                                                           patience=params_grid["scheduler_patience"]
                                                           )

    runner = ClassificationVAERunner(device=DEVICE)
    runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            criterion=criterion,
            logdir=logdir,
            num_epochs=NUM_EPOCHS,
            verbose=True,
            load_best_on_end=True,
            callbacks=callbacks,
            scheduler=scheduler,
            resume=params_grid["resume_path"] if params_grid["resume_path"] else None,
            main_metric="nlll"
        )



def run(args):
    params_grid_path = args.params_grid
    general_grid_path = args.gen_grid

    params_grid = json.load(open(params_grid_path, "r"))
    general_grid = json.load(open(general_grid_path, "r"))

    dist_type = general_grid['distr']
    general_grid["model_params_path"] = f"configs/{dist_type}_model_config.json"

    param_keys = list(params_grid.keys())

    params_combinations = itertools.product(*(params_grid[Name] for Name in param_keys))

    os.makedirs(general_grid["save_dir"], exist_ok=True)

    for comb_ in params_combinations:

        save_dir_name = dist_type + "_"

        for k, v in zip(param_keys, comb_):
            save_dir_name += f"__{k}_{v}"
            general_grid[k] = v
        logdir = os.path.join(general_grid["save_dir"], save_dir_name)
        general_grid["logdir"] = logdir
        train(general_grid)



if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument("-params_grid", type=str, default="configs/params_grid.json")
    args.add_argument("-gen_grid", type=str,  default="configs/general_grid.json")

    args = args.parse_args()

    run(args)









