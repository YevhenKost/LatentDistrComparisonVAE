import torch
from catalyst import dl
import random

class ClassificationVAERunner(dl.Runner):

    def cut_preds(self, preds, seq_lens):

        cutted_preds = []
        for i, seq_len in enumerate(seq_lens):
            cutted_preds.append(preds[i][:seq_len])
        cutted_preds = torch.cat(cutted_preds)
        return cutted_preds

    def predict_batch(self, batch):
        return self.model(
            unmasked_indexes=batch["unmasked_padded_indexes"],
            masked_indexes=batch["masked_padded_indexes"],
            seq_lens=batch["seq_lens"]
        )

    def select_random_indexes(self, target_indexes, preds, random_ratio):
        if random_ratio == 1:
            return {
                "selected_target_indexes": target_indexes,
                "selected_preds": preds
            }

        len_ = len(preds)

        num_select = max(int(len_ * random_ratio), 1)
        idx = random.choices(list(range(len_)), k=num_select)
        idx = sorted(idx, reverse=True)

        output_dict = {
            "selected_target_indexes": target_indexes[idx],
            "selected_preds": preds[idx]
        }

        return output_dict



    def get_masked_preds(self, preds, targets, targets_vectors, masked_indexes):
        masked_preds = []
        masked_targets = []
        masked_preds_vectors = []
        masked_targets_vectors = []
        for i, indx in enumerate(masked_indexes):
            if indx:
                masked_preds.append(torch.argmax(preds[i][indx], dim=-1))
                masked_preds_vectors.append(preds[i][indx])
                masked_targets.append(targets[i][indx])
                masked_targets_vectors.append(targets_vectors[i][indx])

        masked_out = {
            "masked_preds": torch.cat(masked_preds).view(-1).long(),
            "masked_preds_vectors": torch.cat(masked_preds_vectors).view(1, -1),
            "masked_targets": torch.cat(masked_targets).view(-1).long(),
            "masked_targets_vectors": torch.cat(masked_targets_vectors).view(1, -1)
        }
        return masked_out

    def _handle_batch(self, batch):

        preds, params_dict = self.predict_batch(batch)
        cutted_preds = self.cut_preds(preds, batch["seq_lens"])

        masked_results = self.get_masked_preds(
            preds=preds,
            targets=batch["unmasked_padded_indexes"],
            targets_vectors=batch["unmasked_padded_vectors"],
            masked_indexes=batch["masked_indexes"]
        )



        true_unpadded_vectors = torch.cat(batch["unmasked_unpadded_vectors"]).float()
        true_unpadded_indexes = torch.cat(batch["unmasked_unpadded_indexes"])
        selected_dict = self.select_random_indexes(
            target_indexes=true_unpadded_indexes,
            preds=cutted_preds.float(),
            random_ratio=batch["select_ratio"]
        )

        self.input = {


            "true_unpadded_indexes": true_unpadded_indexes.long(),
            "true_padded_indexes": batch["unmasked_padded_indexes"],
            "true_selected_indexes": selected_dict["selected_target_indexes"],

            "visualize_tokens": {
                "input_string_tokens": batch["masked_string_tokens"],
                "output_string_tokens": batch["unmasked_string_tokens"]
            },
            "true_unpadded_vectors": true_unpadded_vectors,
            "masked_indexes": masked_results["masked_targets"],

        }

        self.output = {

            "dist_loss_select": {
                "reconstructed": selected_dict["selected_preds"],
                "distr_params": params_dict
            },


            "dist_loss_unpad": {
                "reconstructed": cutted_preds.reshape(true_unpadded_vectors.shape).float(),
                "distr_params": params_dict
            },
            "pred_padded_indexes": torch.argmax(preds, dim=-1),
            "pred_unpadded_indexes": torch.argmax(cutted_preds, dim=-1),
            "masked_preds": masked_results["masked_preds"],
            "pred_scores": preds.float(),
            "distr_params": params_dict,
            "unpad_preds": cutted_preds.reshape(true_unpadded_vectors.shape).float()

        }


