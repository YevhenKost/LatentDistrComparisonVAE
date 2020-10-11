import torch.nn as nn
from archs.common_layers import get_relu_sequantial
import torch

class NormalDistrGenerator(nn.Module):
    def __init__(self, input_dim, mean_layers_dim, mean_output_dim, var_layers_dim):
        super(NormalDistrGenerator, self).__init__()

        self.mean_layers = get_relu_sequantial(
            input_dim=input_dim,
            layers=mean_layers_dim,
            output_dim=mean_output_dim,
            last_activation=None
        )

        self.var_layers = get_relu_sequantial(
            input_dim=input_dim,
            layers=var_layers_dim,
            output_dim=mean_output_dim,
            last_activation=None
        )
        self.out_dim = mean_output_dim


    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        z = torch.randn(mean.size()).to(mean.device)
        return z*std + mean

    def get_output_dim(self):
        return self.out_dim

    def forward(self, input):

        mu = self.mean_layers(input)
        var = self.var_layers(input)

        generated_z = self.reparameterize(mu, var)
        return generated_z, {"mean":mu, "std":var}
