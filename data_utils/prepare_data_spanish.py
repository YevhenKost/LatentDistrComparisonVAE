import os, sys

# adding current path for importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils.preprocessing_utils import PreprocessOHE, apply_dropout, Preprocessing, SpanishTokenizerLemmatizer
import pandas as pd
import os, json, random
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize


def save_prepared(args):
    random.seed(2)

    save_path = args.save_dir
    load_path = args.load_path
    preprocess_config = json.load(open(args.config_path, "r"))

    add_tokens = [

        preprocess_config["start_seq_token"],
        preprocess_config["pad_token"],
        preprocess_config["end_seq_token"],
        preprocess_config["mask_token"]

    ]
    mask_token = preprocess_config["mask_token"]

    drp_rate = preprocess_config["valid_dropout_rate"]
    test_size = preprocess_config["test_size"]
    valid_size = preprocess_config["valid_size_from_train"]

    os.makedirs(save_path, exist_ok=True)


    sents = pd.read_csv(load_path)
    sents = sents["content"].dropna().values.tolist()
    print(f"Total unfiltered loaded: {len(sents)}")

    sents = [sent_tokenize(x) for x in sents]
    sents = sum(sents, [])

    sent_df = pd.DataFrame()
    sent_df["sents"] = sents
    sent_df.to_csv(os.path.join(save_path, "read_sents.csv"))

    print("Start preprocessing...")
    sent_df["sents"] = sent_df["sents"].apply(lambda x: Preprocessing.preprocess(x))
    sent_df = sent_df.dropna().drop_duplicates()

    tok_lem = SpanishTokenizerLemmatizer()

    print("Starting tokenization and lemmatizing...")
    sent_df["tokenized_sents"] = sent_df["sents"].apply(lambda x: tok_lem(x))

    print("Building vocabulary...")
    enc_dict = PreprocessOHE.get_encoding_dict(sent_df["tokenized_sents"].values.tolist(),
                                               preprocess_config,
                                               drop_stopword=preprocess_config["drop_stop_words"],
                                               drop_punct=preprocess_config["drop_punt"],
                                               add_tokens=add_tokens)

    with open(os.path.join(save_path, "encoding_dict.json"), "w") as f:
            json.dump(enc_dict, f)
    with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(preprocess_config, f)


    sent_df["tokenized_sents"] = sent_df["tokenized_sents"].apply(lambda x: PreprocessOHE.filter_tokens_voc(x, enc_dict))
    sent_df = sent_df.dropna(subset=["tokenized_sents"])



    sent_df["len"] = sent_df["tokenized_sents"].apply(lambda x: len(x))

    train_df, test_df = train_test_split(sent_df, random_state=2, test_size=test_size)
    train_df, valid_df = train_test_split(train_df, random_state=2, test_size=valid_size)
    train_df = train_df.reset_index()
    test_df = test_df.reset_index()
    valid_df = valid_df.reset_index()

    test_df[f"dropout_tokenized_sents_{drp_rate}"] = test_df["tokenized_sents"].apply(lambda x: apply_dropout(x, drp_rate, mask_token))
    valid_df[f"dropout_tokenized_sents_{drp_rate}"] = valid_df["tokenized_sents"].apply(lambda x: apply_dropout(x, drp_rate, mask_token))

    test_df[f"dropout_tokenized_sents_{0.5}"] = test_df["tokenized_sents"].apply(
        lambda x: apply_dropout(x, 0.5, mask_token))
    valid_df[f"dropout_tokenized_sents_{0.5}"] = valid_df["tokenized_sents"].apply(
        lambda x: apply_dropout(x, 0.5, mask_token))


    train_df.to_json(os.path.join(save_path, "train.json"))
    test_df.to_json(os.path.join(save_path, "test.json"))
    valid_df.to_json(os.path.join(save_path, "valid.json"))


    train_df.to_csv(os.path.join(save_path, "train.csv"))
    test_df.to_csv(os.path.join(save_path, "test.csv"))
    valid_df.to_csv(os.path.join(save_path, "valid.csv"))


    print("Split length: (train, test)")
    print(len(train_df), len(test_df))
    print(min([len(x) for x in train_df["tokenized_sents"].values.tolist()]), max([len(x) for x in train_df["tokenized_sents"].values.tolist()]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-save_dir", type=str, default="/media/yevhen/Disk 1/Research/ComparisonVAE/datasets/spanishPoetry_masking_nopunct_nostops")
    parser.add_argument("-load_path", type=str, default="/media/yevhen/Disk 1/DataSets/spanishPoetry/poems.csv")
    parser.add_argument("-config_path", type=str, default="../configs/preprocessing_data_config.json")

    args = parser.parse_args()
    save_prepared(args)