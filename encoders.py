import numpy as np
import torch, random

class OneHotEncoder:
    def __init__(self,
                 enc_dict, max_len,
                 pad_tok, mask_idx,
                 end_tok=None,
                 start_token=None,
                 input_dropout=0,
                 dropout_approach="prob"
                 ):

        self.pad_tok = pad_tok
        self.mask_idx = mask_idx
        self.enc_dict = enc_dict
        self.dec_dict = {v:k for k,v in self.enc_dict.items()}

        self.end_token = end_tok
        self.start_token = start_token
        self.n_uniq = len(enc_dict)
        self.max_len = max_len + bool(self.start_token) + bool(self.end_token)

        self.input_dropout = input_dropout
        self.dropout_approach = dropout_approach


    def get_max_len_enc(self):
        return self.max_len
    def get_output_dim(self):
        return self.n_uniq

    def one_hot_encode(self, target):
        target_ = np.zeros((len(target), self.n_uniq))
        for i, t_i in enumerate(target):
            target_[i][t_i] = 1
        return target_

    def _padd(self, encoded_target):
        seq_len = len(encoded_target)
        padded_target = encoded_target.copy()
        padded_target += [self.enc_dict[self.pad_tok]] * (self.max_len - len(padded_target))
        return padded_target, seq_len

    def get_word_id(self, word):
        if word in self.enc_dict:
            return self.enc_dict[word]
        return "NotInDict"

    def _prepare_tokens(self, tokens):
        string_tokens = tokens

        unpadded_indexes = [self.get_word_id(x) for x in string_tokens]
        string_tokens = [x for x in string_tokens if x]
        unpadded_indexes = [x for x in unpadded_indexes if x != "NotInDict"]

        if self.start_token:
            unpadded_indexes = [self.enc_dict[self.start_token]] + unpadded_indexes
            string_tokens = [self.start_token] + string_tokens
        if self.end_token:
            unpadded_indexes.append(self.enc_dict[self.end_token])
            string_tokens = string_tokens + [self.end_token]

        padded_indexes, seq_len = self._padd(unpadded_indexes)

        out_dict = {
            "padded_indexes": padded_indexes,
            "unpadded_indexes": unpadded_indexes,

            "string_tokens": string_tokens,

            "seq_len": seq_len
        }

        return out_dict

    def _apply_prob_token_dropout(self, prepared_dict):
        indexes = list(range(len(prepared_dict["padded_indexes"])))
        probs = np.random.uniform(0,1, len(indexes))
        if_drop = probs > self.input_dropout
        if_drop[len(prepared_dict["unpadded_indexes"]):] = False

        unpad_mask = np.array(if_drop[:len(prepared_dict["unpadded_indexes"])], dtype=int).tolist()

        prepared_dict["string_tokens"] = [x if if_drop[i] else self.mask_idx for i, x in enumerate(prepared_dict["string_tokens"])]
        prepared_dict["padded_indexes"] = [x if if_drop[i] else self.get_word_id(self.mask_idx) for i, x in enumerate(prepared_dict["padded_indexes"])]
        prepared_dict["unpadded_indexes"] = [x if if_drop[i] else self.get_word_id(self.mask_idx) for i, x in enumerate(prepared_dict["unpadded_indexes"])]
        prepared_dict["unpad_mask"] = unpad_mask
        return prepared_dict


    def _apply_fixedsize_token_dropout(self, prepared_dict):

        if isinstance(self.input_dropout, int):
            k=self.input_dropout
        else:
            k = int(self.input_dropout*len(prepared_dict["unpadded_indexes"]))

        indexes_drop = random.choices(
            list(range(len(prepared_dict["unpadded_indexes"]))),
            k=k
        )
        prepared_dict["string_tokens"] = [x if i not in indexes_drop else self.mask_idx for i, x in enumerate(prepared_dict["string_tokens"])]
        prepared_dict["padded_indexes"] = [x if i not in indexes_drop else self.get_word_id(self.mask_idx)
                                           for i, x in enumerate(prepared_dict["padded_indexes"])]
        prepared_dict["unpadded_indexes"] = [x if i not in indexes_drop else self.get_word_id(self.mask_idx)
                                             for i, x in enumerate(prepared_dict["unpadded_indexes"])]
        return prepared_dict

    def _apply_token_dropout(self, dict_):

        if self.dropout_approach == "prob":
            return self._apply_prob_token_dropout(dict_)

        elif self.dropout_approach == "fixed":
            return self._apply_fixedsize_token_dropout(dict_)

    def padd_encode(self, input_tokens, target_tokens=None, use_dropout=False):

        if not target_tokens:
            target_tokens = input_tokens

        masked_dict = self._prepare_tokens(input_tokens)
        unmasked_dict = self._prepare_tokens(target_tokens)

        if use_dropout:
            masked_dict = self._apply_token_dropout(masked_dict)


        masked_indexes = [i for i, x in enumerate(masked_dict["unpadded_indexes"]) if
                         x == self.get_word_id(self.mask_idx)]


        if "unpad_mask" not in masked_dict:
            masked_dict["unpad_mask"] = [0 if i not in masked_indexes else 1 for i in range(len(masked_dict["unpadded_indexes"]))]

        assert masked_dict["seq_len"] == unmasked_dict["seq_len"]


        out_dict = {

            "masked_padded_indexes": masked_dict["padded_indexes"],
            "masked_unpadded_indexes": masked_dict["unpadded_indexes"],

            "masked_string_tokens": masked_dict["string_tokens"],
            "unmasked_string_tokens": unmasked_dict["string_tokens"],

            "unmasked_padded_indexes": unmasked_dict["padded_indexes"],
            "unmasked_unpadded_indexes": unmasked_dict["unpadded_indexes"],

            "unmasked_padded_vectors": self.one_hot_encode(unmasked_dict["padded_indexes"]),
            "unmasked_unpadded_vectors": self.one_hot_encode(unmasked_dict["unpadded_indexes"]),

            "seq_len": masked_dict["seq_len"],
            "masked_indexes": masked_indexes,

            "unpad_mask": masked_dict["unpad_mask"]

        }
        return out_dict


    def decode(self, pred_scores_seqs, need_argmax=True):
        if not isinstance(pred_scores_seqs, torch.Tensor):
            pred_scores_seqs = torch.Tensor(pred_scores_seqs)

        if need_argmax:
            pred_tokens = torch.argmax(pred_scores_seqs, dim=-1)
        else:
            pred_tokens = pred_scores_seqs


        output_seq = [self.dec_dict[x.item()] for x in pred_tokens]
        return output_seq
