from catalyst.core.callbacks.metrics import IBatchMetricCallback, BatchMetricCallback
from catalyst.dl import Callback, CallbackOrder, IRunner
from losses import KLD, NLLLoss_criterion
from sklearn.metrics import accuracy_score
import torch
import random
import json
import numpy as np

class FullLossCallback(IBatchMetricCallback):
    def __init__(self,
                 output_key: str = "distr_params",
                 prefix: str = "norm_kld",
                 input_key="true_padded_indexes",
                 distr_type='N',
                 loss_reduction="sum",
                 tensorboard_callback_name: str = "_full_loss",
                 weight=None
                 ):
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=1
        )

        self.distr_type = distr_type
        self.loss = NLLLoss_criterion(reduction=loss_reduction, weights=weight, distr_type=distr_type)

    def get_criterion(self):
        self._criterion = self.loss

    @property
    def metric_fn(self):
        """Criterion function."""
        return self._criterion

    def on_stage_start(self, runner: IRunner):
        """Checks that the current stage has correct criterion.

        Args:
            runner (IRunner): current runner
        """
        self._criterion = self.loss

class MetricsSaverCallback(Callback):
    def __init__(self,
                 save_path_dir
                 ):
        super().__init__(CallbackOrder.Internal)

        self._save_path = save_path_dir
        self.metrics = {}

    def _reset_stats(self):
        self.metrics = {}

    def on_loader_end(self, runner: "IRunner"):

        mean_metrics = {
            k: np.mean(v) for k,v in self.metrics.items()
        }
        with open(self._save_path, "w") as f:
            json.dump(mean_metrics, f)
        self._reset_stats()

    def on_batch_end(self, runner: "IRunner"):
        metrics = runner.batch_metrics
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append(v.item())

class LinearWeightKLDCallback(Callback):
    def __init__(self,
                 x=2500
                 ):
        super().__init__(CallbackOrder.Internal)
        self.x = x
        self.step = 0

    def on_batch_end(self, runner: "IRunner"):
        self.step += 1
        weight = self.kl_anneal_function()
        # runner.output["dist_loss_dict_masked"]["kld_weight"] = self.kl_anneal_function()
        runner.output["dist_loss_unpad"]["kld_weight"] = weight
        runner.output["dist_loss_select"]["kld_weight"] = weight


    def kl_anneal_function(self):
        return min(1, self.step / self.x)

class SigmoidWeightKLDCallback(Callback):
    def __init__(self,
                 x=2500,
                 k=100
                 ):
        super().__init__(CallbackOrder.Internal)
        self.x = x
        self.k = k
        self.step = 0

    def on_batch_end(self, runner: "IRunner"):
        self.step += 1
        weight = self.kl_anneal_function()
        runner.output["dist_loss_unpad"]["kld_weight"] = weight
        runner.output["dist_loss_select"]["kld_weight"] = weight


    def kl_anneal_function(self):
        return float(1/(1+np.exp(-self.k*(self.step-self.x))))


class KLDCallback(IBatchMetricCallback):
    def __init__(self,
                 output_key: str = "distr_params",
                 prefix: str = "norm_kld",
                 input_key="true_padded_indexes",
                 distr_type='N',
                 tensorboard_callback_name: str = "_norm_kld"
                 ):
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=1
        )

        self.distr_type = distr_type

    def get_criterion(self):
        self._criterion = KLD.get_kld_func(self.distr_type)

    @property
    def metric_fn(self):
        """Criterion function."""
        return self._criterion

    def on_stage_start(self, runner: IRunner):
        """Checks that the current stage has correct criterion.

        Args:
            runner (IRunner): current runner
        """
        self._criterion = KLD.get_kld_func(self.distr_type)

class NLLLossCallback(IBatchMetricCallback):
    def __init__(self,
                 output_key: str = "pred_scores",
                 prefix: str = "nllloss",
                 input_key="true_unpadded_indexes",
                 reduction="sum",
                 weights=None,
                 tensorboard_callback_name: str = "_nllloss"
                 ):
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=1
        )

        self._loss = torch.nn.NLLLoss(reduction=reduction, weight=weights)

    def get_criterion(self):
        self._criterion = self._loss

    @property
    def metric_fn(self):
        """Criterion function."""
        return self._loss

    def on_stage_start(self, runner: IRunner):
        """Checks that the current stage has correct criterion.

        Args:
            runner (IRunner): current runner
        """
        self._criterion = self.get_criterion()

class VizDecodeCallback(Callback):
    def __init__(self,
                 input_key: str = "ordered_indexes",
                 output_key: str = "ordered_indexes",
                 prefix: str = "decoded_text",
                 tensorboard_callback_name: str = "_tensorboard",
                 max_num = 5,
                 encoder=None
                 ):
        super().__init__(CallbackOrder.Internal)

        self.input_tokens = []
        self.output_tokens = []
        self.reconstructed = []

        self.encoder = encoder

        self.input_key = input_key
        self.output_key = output_key

        self.max_num = max_num

    def reset_stats(self):
        self.input_tokens = []
        self.output_tokens = []
        self.reconstructed = []

    def on_batch_end(self, runner: IRunner):

        if len(self.input_tokens) <= self.max_num:
            save_indx = random.choice(list(range(len(runner.input[self.input_key]["input_string_tokens"]))))
            self.input_tokens.append(runner.input[self.input_key]["input_string_tokens"][save_indx])
            self.output_tokens.append(runner.input[self.input_key]["output_string_tokens"][save_indx])
            self.reconstructed.append(runner.output[self.output_key][save_indx])

    def on_loader_start(self, runner: "IRunner"):
        pass
    def on_batch_start(self, runner: "IRunner"):
        pass


    def on_loader_end(self, runner: IRunner):

        to_print_idx = random.randint(0, len(self.input_tokens)-1)

        input_tokens = self.input_tokens[to_print_idx]
        output_tokens = self.output_tokens[to_print_idx]

        recostructed = self.reconstructed[to_print_idx]
        recostructed = self.encoder.decode(recostructed.detach().cpu().numpy())

        print("\nInput Tokens:")
        print(input_tokens)
        print("\nTarget tokens:")
        print(output_tokens)
        print("\nReconstructed tokens: ")
        print(recostructed)

        self.reset_stats()



def custom_accuracy(
    classes_pred: torch.Tensor,
    classes_true: torch.Tensor,
) -> float:


    classes_pred = classes_pred.long().view(-1).cpu().numpy().tolist()
    classes_true = classes_true.long().view(-1).cpu().numpy().tolist()

    return accuracy_score(classes_pred, classes_true)

class CustomAccuracyCallback(BatchMetricCallback):

    def __init__(
        self,
        input_key: str = "true_order",
        output_key: str = "predicted_order",
        prefix: str = "sorting_accuracy",
        multiplier: float = 1.0
    ):

        super().__init__(
            prefix=prefix,
            metric_fn=custom_accuracy,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
        )