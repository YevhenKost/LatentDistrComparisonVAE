from torch.utils.data import Dataset, DataLoader
import torch, os
import pandas as pd

def read_sents(path, col, max_len=None, min_len=None, verbose=False):
    filename, file_extension = os.path.splitext(path)
    if file_extension in [".pkl", ".pickle"]:
        sents = pd.read_pickle(path).dropna(subset=[col])
    if file_extension == ".feather":
        sents = pd.read_feather(path).dropna(subset=[col])
    if file_extension == ".cvs":
        sents = pd.read_csv(path).dropna(subset=[col])
    if file_extension == ".json":
        sents = pd.read_json(path).dropna(subset=[col])

    if max_len:
        sents = sents[sents["len"] <= max_len]
    if min_len:
        sents = sents[sents["len"] >= min_len]
    sents = sents[col].values.tolist()
    if verbose:
        print(f"Num sent: {len(sents)}")
    return sents

class TrainSentDataset(Dataset):
    def __init__(self, sents, encoder, use_token_dropout=True):

        self.encoder = encoder
        self.sents = sents
        self.use_dropout = use_token_dropout

    def __getitem__(self, item):

        sents = self.sents[item]
        encoded_dict = self.encoder.padd_encode(sents, None, self.use_dropout)
        return encoded_dict

    def __len__(self):
        return len(self.sents)


class ValidSentDataset(Dataset):
    def __init__(self, tokenized_sents, tokenized_dropout_sents, encoder):
        self.encoder = encoder
        self.sents = tokenized_sents
        self.dropout_sents = tokenized_dropout_sents


    def __getitem__(self, item):
        sents = self.sents[item]
        dropout_sents = self.dropout_sents[item]
        encoded_dict = self.encoder.padd_encode(dropout_sents, sents, False)
        return encoded_dict

    def __len__(self):
        return len(self.sents)


class DataLoaders:

    @classmethod
    def _collate(cls, batch, select_ratio=0):

        seq_lens = torch.Tensor([x["seq_len"] for x in batch]).long()

        masked_unpadded_indexes = [torch.Tensor(x["masked_unpadded_indexes"]) for x in batch]
        masked_padded_indexes = torch.Tensor([x["masked_padded_indexes"] for x in batch])

        unmasked_unpadded_indexes = [torch.Tensor(x["unmasked_unpadded_indexes"]) for x in batch]
        unmasked_padded_indexes = torch.Tensor([x["unmasked_padded_indexes"] for x in batch])

        unmasked_unpadded_vectors = [torch.Tensor(x["unmasked_unpadded_vectors"]) for x in batch]
        unmasked_padded_vectors = torch.Tensor([x["unmasked_padded_vectors"] for x in batch])

        masked_string_tokens = [x["masked_string_tokens"] for x in batch]
        unmasked_string_tokens = [x["unmasked_string_tokens"] for x in batch]

        masked_indexes = [x["masked_indexes"] for x in batch]
        mask = torch.Tensor(sum([x["unpad_mask"] for x in batch], [])).long()

        # cut on max batch seq len
        max_seq_len = max(seq_lens).item()
        masked_padded_indexes = masked_padded_indexes[:, :max_seq_len]
        unmasked_padded_indexes = unmasked_padded_indexes[:, :max_seq_len]
        unmasked_padded_vectors = unmasked_padded_vectors[:, :max_seq_len, :]

        out_dict = {

            "masked_padded_indexes": masked_padded_indexes,
            "masked_unpadded_indexes": masked_unpadded_indexes,

            "masked_string_tokens": masked_string_tokens,
            "unmasked_string_tokens": unmasked_string_tokens,

            "unmasked_padded_indexes": unmasked_padded_indexes,
            "unmasked_unpadded_indexes": unmasked_unpadded_indexes,

            "unmasked_padded_vectors": unmasked_padded_vectors,
            "unmasked_unpadded_vectors": unmasked_unpadded_vectors,

            "seq_lens": seq_lens,
            "masked_indexes": masked_indexes,

            "mask": mask,
            "select_ratio": select_ratio

        }
        return out_dict

    @classmethod
    def get_loader(cls, dataset_config, batch_size, is_train, select_ratio=0):

        if is_train:
            dataset = TrainSentDataset(**dataset_config)
        else:
            dataset = ValidSentDataset(**dataset_config)

        loader = DataLoader(
            dataset=dataset,
            shuffle=is_train,
            batch_size=batch_size,
            drop_last=False,
            collate_fn=lambda x: cls._collate(x, select_ratio=select_ratio)
        )
        return loader
