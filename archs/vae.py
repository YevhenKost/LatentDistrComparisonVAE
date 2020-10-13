import torch.nn as nn
import torch
from archs.common_layers import ExtendedEmbeddingLayer, RNNLayer, get_relu_sequantial
from archs.dist_generators import NormalDistrGenerator, LogNormalDistrGenerator, CauchyDistrGenerator


class VAE(nn.Module):
    def __init__(self,
                 embedding_config,
                 rnn_encoder_config,
                 rnn_decoder_config,
                 latent_layers,
                 distr_config,
                 embeddding_dropout_rate,
                 type_distr,
                 final_layers,
                 start_idx, mask_idx
                 ):

        super(VAE, self).__init__()


        self.embedding_layer = ExtendedEmbeddingLayer(**embedding_config)
        self.rnn_encoder = RNNLayer(**rnn_encoder_config)
        self.rnn_decoder = RNNLayer(**rnn_decoder_config)
        self.embedding_dropout = nn.Dropout(embeddding_dropout_rate)

        distr_config["input_dim"] = self.rnn_encoder.get_hidden_shape()
        if type_distr == "N":
            self.latent_generator = NormalDistrGenerator(**distr_config)
        if type_distr == "LogN":
            self.latent_generator = LogNormalDistrGenerator(**distr_config)
        if type_distr == "Cauchy":
            self.latent_generator = CauchyDistrGenerator(**distr_config)


        self.latent2decoder_hidden = get_relu_sequantial(
            input_dim=self.latent_generator.get_output_dim(),
            output_dim=self.rnn_decoder.get_hidden_shape(),
            layers=latent_layers,
            last_activation=None
        )
        self.finalizer = get_relu_sequantial(
            input_dim=self.rnn_decoder.get_output_dim(),
            output_dim=self.embedding_layer.get_num_voc(),
            last_activation=nn.LogSoftmax(dim=-1),
            layers=final_layers
        )

        self.start_idx = start_idx
        self.mask_idx = mask_idx

    def forward(self, unmasked_indexes, masked_indexes, seq_lens):

        unmasked_embedded = self.embedding_layer(unmasked_indexes.long())

        masked_embedded = self.embedding_layer(masked_indexes.long())
        masked_embedded = self.embedding_dropout(masked_embedded)

        # encoding rnn
        encoded_unmasked = self.rnn_encoder(inputs=unmasked_embedded,
                                            seq_lens=seq_lens,
                                            fill_hid=None
                                            )["last_hiddens"]

        # generating latent
        z, distr_params = self.latent_generator(encoded_unmasked)

        decoder_hidden = self.latent2decoder_hidden(z)


        # decoding
        decoder_output = self.rnn_decoder(inputs=masked_embedded,
                                          seq_lens=seq_lens,
                                          fill_hid=decoder_hidden)["rnn_out"]


        logits = self.finalizer(decoder_output)

        return logits, distr_params


    def generate(self, n, len_gen, device):
        batch_size = n
        z = torch.randn([batch_size, self.latent_generator.get_output_dim()]).to(device)
        decoder_hidden = self.latent2decoder_hidden(z)

        seq_gen = torch.Tensor(n, len_gen+1).fill_(self.mask_idx).long().to(device)
        seq_gen[0, :] = self.start_idx

        embedded_masked_seq = self.embedding_layer(seq_gen)
        seq_lens = torch.Tensor([len_gen+1]*n)
        decoder_output = self.rnn_decoder(inputs=embedded_masked_seq,
                                          seq_lens=seq_lens,
                                          fill_hid=decoder_hidden)["rnn_out"]

        logits = self.finalizer(decoder_output)
        indexes = torch.argmax(logits, dim=-1)
        return indexes