import os
import sys

# adding current path for importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

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


def train(args):

    # fixing random seeds
    _fix_seeds()

    # setting training parameters
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    DEVICE = args.device

    MAX_LEN = args.max_len
    MIN_LEN = args.min_len
    LOSS_REDUCTION=args.loss_reduction

    dir_path = args.data_path
    logdir = args.logdir

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
        open(args.model_params_path, "r")
    )
    arch_config["embeddding_dropout_rate"] = args.embedding_mask_dropout
    arch_config["embedding_config"]["num_embeddings"] = len(enc_dict)
    arch_config["embedding_config"]["padding_idx"] = enc_dict[PAD]
    arch_config["start_idx"] = enc_dict[START]
    arch_config["mask_idx"] = enc_dict[MASK]
    arch_config["rnn_encoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]
    arch_config["rnn_decoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]


    TYPE_DISTR = arch_config["type_distr"]


    # loading encoder
    MASK_DROPOUT = args.mask_dropout

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
            select_ratio=args.select_ratio

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
            k=args.weight_kld_k,
            x=args.weight_kld_x
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.scheduler_factor)

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
            resume=args.resume_path if args.resume_path else None,
            main_metric="nlll"
        )

if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument("-batch_size", type=int, default=128)
    args.add_argument("-lr", type=float, default=1e-3)
    args.add_argument("-loss_reduction", type=str, default="sum")

    args.add_argument("-epochs", type=int, default=10)
    args.add_argument("-device", type=str, default="cuda")
    args.add_argument("-min_len", type=int, default=10)
    args.add_argument("-max_len", type=int, default=30)

    args.add_argument("-weight_kld_x", type=float, default=10)
    args.add_argument("-weight_kld_k", type=float, default=10)

    args.add_argument("-scheduler_factor", type=float, default=0.85)

    args.add_argument("-mask_dropout", type=float, default=0.2)
    args.add_argument("-embedding_mask_dropout", type=float, default=0)
    args.add_argument("-select_ratio", type=float, default=1.0)

    args.add_argument("-data_path", type=str, default="datasets/covidTweets_masking_nopunct_nostops")
    args.add_argument("-logdir", type=str, default="trained_models/model/")
    args.add_argument("-resume_path", type=str, default="")
    args.add_argument("-model_params_path", type=str, default="configs/lognormal_model_config.json")


    args = args.parse_args()
    train(args)