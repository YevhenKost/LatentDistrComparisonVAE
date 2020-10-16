import torch.nn as nn
import torch



class KLD:
    @classmethod
    def norm_kld(cls, params_dict, add_args=None):
        mu = params_dict["mean"]
        logvar = params_dict["logstd"]
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    @classmethod
    def cauchy_kld(cls, params_dict, add_args=None):
        mu = params_dict["mean"]
        logvar = params_dict["logstd"]
        var = torch.exp(logvar)

        KLD = - torch.sum((torch.log(((var + 1)**2) + mu**2) - torch.log(4*var)))
        return KLD

    @classmethod
    def get_kld_func(cls, distr_type):
        if distr_type == "N":
            return cls.norm_kld
        if distr_type == "LogN":
            return cls.norm_kld
        if distr_type == "Cauchy":
            return cls.cauchy_kld






class NLLLoss_criterion(nn.Module):
    def __init__(self, reduction="mean", distr_type="N", weights=None, losses_weigths=None):
        super(NLLLoss_criterion, self).__init__()
        self.loss = nn.NLLLoss(reduction=reduction, weight=weights)
        self.losses_weigths = losses_weigths
        self._get_kld(distr_type)

    def _get_kld(self, distr_type):
        self.kld = KLD.get_kld_func(distr_type=distr_type)

    def forward(self, dict_, true_indexes):
        recon_x = dict_["reconstructed"]
        params_dict = dict_["distr_params"]
        kld_weight = dict_["kld_weight"]


        LOSS = self.loss(recon_x.float(), true_indexes.long())
        KLD = self.kld(params_dict)

        if self.losses_weigths:
            TOTAL_LOSS = self.losses_weigths["KLD"]*KLD + self.losses_weigths["LOSS"]*LOSS
        else:
            TOTAL_LOSS = LOSS + kld_weight*KLD

        return TOTAL_LOSS

