import torch.nn as nn
from archs.common_layers import get_relu_sequantial
import torch


class DeterministicGenerator(nn.Module):
    # Build the same way as other generators

    def __init__(self, input_dim, output_dim):
        super(DeterministicGenerator, self).__init__()

        self.den_layer = nn.Linear(input_dim, output_dim)
        self.out_dim = output_dim

    def get_output_dim(self):
        return self.out_dim

    def forward(self, input):

        generated_z = self.den_layer(input)
        return generated_z, {"mean":None, "logstd":None}


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
        return generated_z, {"mean":mu, "logstd":var}


class LogNormalDistrGenerator(nn.Module):
    def __init__(self, input_dim, mean_layers_dim, mean_output_dim, var_layers_dim):
        super(LogNormalDistrGenerator, self).__init__()

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
        normal = torch.randn(mean.size()).to(mean.device)
        normal = normal*std + mean
        z = torch.exp(normal)

        return z

    def get_output_dim(self):
        return self.out_dim

    def forward(self, input):

        mu = self.mean_layers(input)
        var = self.var_layers(input)

        generated_z = self.reparameterize(mu, var)
        return generated_z, {"mean":mu, "logstd":var}


class CauchyDistrGenerator(nn.Module):
    def __init__(self, input_dim, mean_layers_dim, mean_output_dim, var_layers_dim):
        super(CauchyDistrGenerator, self).__init__()

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
        uniform = torch.rand(size=mean.size()).to(mean.device)

        z = mean + std*torch.tan(3.14 * (uniform - 0.5))
        return z

    def get_output_dim(self):
        return self.out_dim

    def forward(self, input):

        mu = self.mean_layers(input)
        var = self.var_layers(input)

        generated_z = self.reparameterize(mu, var)
        return generated_z, {"mean":mu, "logstd":var}