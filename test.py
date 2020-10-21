import os
import sys

# adding current path for importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from archs.vae import VAE
from runners import ClassificationVAERunner
from data_utils.data_loading import DataLoaders, read_sents
from callbacks import CustomAccuracyCallback, KLDCallback, \
    VizDecodeCallback, LinearWeightKLDCallback, NLLLossCallback, MetricsSaverCallback, FullLossCallback
import json
from collections import OrderedDict
from encoders import OneHotEncoder


def test_model(
        dir_path,
        checkpoint_path,
        model_params_path,save_metric_path,
        min_len=10, max_len=60, loss_reduction="sum",
        batch_size=128, device="cuda", lin_weight_x=10
):


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
        open(model_params_path, "r")
    )
    arch_config["embeddding_dropout_rate"] = 0.0
    arch_config["embedding_config"]["num_embeddings"] = len(enc_dict)
    arch_config["embedding_config"]["padding_idx"] = enc_dict[PAD]
    arch_config["start_idx"] = enc_dict[START]
    arch_config["mask_idx"] = enc_dict[MASK]
    arch_config["rnn_encoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]
    arch_config["rnn_decoder_config"]["input_dim"] = arch_config["embedding_config"]["embedding_dim"]


    TYPE_DISTR = arch_config["type_distr"]


    encoder = OneHotEncoder(
        enc_dict=enc_dict,
        max_len=max_len,
        mask_idx=MASK,
        pad_tok=PAD,
        end_tok=END,
        start_token=START,
        input_dropout=0.1
    )



    # loading data

    valid_dataset_config = {
        "tokenized_sents": read_sents(os.path.join(dir_path, "test.json"),
                                      min_len=min_len, max_len=max_len,
                                      col="tokenized_sents", verbose=True),
        "tokenized_dropout_sents": read_sents(os.path.join(dir_path, "test.json"),
                                              min_len=min_len, max_len=max_len,
                                              col=test_col),
        "encoder": encoder
    }



    valid_loader = DataLoaders.get_loader(
            dataset_config=valid_dataset_config,
            batch_size=batch_size,
            is_train=False,
            select_ratio=1
        )





    # init model
    model = VAE(**arch_config)
    model = model.to(device)

    # setting callbacks in order
    callbacks = OrderedDict({

        "full_loss": FullLossCallback(
            input_key="true_selected_indexes",
            output_key="dist_loss_select",
            prefix="full_loss",
            distr_type=TYPE_DISTR
        ),

        "kld_weight": LinearWeightKLDCallback(
            x=lin_weight_x
        ),

        "nlll": NLLLossCallback(
            input_key="true_unpadded_indexes",
            output_key="unpad_preds",
            reduction=loss_reduction,
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
        "saver_metrics": MetricsSaverCallback(
            save_path_json=save_metric_path
        )
    })


    runner = ClassificationVAERunner(device=device)
    runner.infer(
        model=model,
        resume=checkpoint_path,
        callbacks=callbacks,
        loaders={"infer":valid_loader},
        verbose=True
    )

    # return metrics
    metrics = json.load(open(save_metric_path, "r"))
    return metrics


if __name__ == '__main__':
    pathes = [
        "/media/yevhen/Disk 1/Research/ComparisonVAE/models",
        "/media/yevhen/Disk 1/Research/ComparisonVAE/remote_trained_models_1/models",
        '/media/yevhen/Disk 1/Research/ComparisonVAE/remote_trained_models_2/models',
    ]
    dir_path = "/media/yevhen/Disk 1/Research/ComparisonVAE/datasets/covidTweets_ent_masking_nopunct_nostops"

    for p in pathes:
        for distr in os.listdir(p):
            dist_dir = os.path.join(p, distr)

            for model in os.listdir(dist_dir):

                model_dir_path = os.path.join(dist_dir, model)
                checkpoint_path = os.path.join(model_dir_path, "checkpoints/best_full.pth")

                print(distr, model)
                metrics = test_model(
                    dir_path=dir_path,
                    checkpoint_path=checkpoint_path,
                    model_params_path=f"configs/{distr}_model_config.json",
                    save_metric_path=os.path.join(model_dir_path, "mean_metric.json")
                )



