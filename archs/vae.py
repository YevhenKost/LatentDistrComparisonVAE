import torch.nn as nn
import torch
from archs.common_layers import ExtendedEmbeddingLayer, RNNLayer, get_relu_sequantial
from archs.dist_generators import NormalDistrGenerator


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
                 start_idx
                 ):

        super(VAE, self).__init__()


        self.embedding_layer = ExtendedEmbeddingLayer(**embedding_config)
        self.rnn_encoder = RNNLayer(**rnn_encoder_config)
        self.rnn_decoder = RNNLayer(**rnn_decoder_config)
        self.embedding_dropout = nn.Dropout(embeddding_dropout_rate)

        distr_config["input_dim"] = self.rnn_encoder.get_hidden_shape()
        if type_distr == "N":
            self.latent_generator = NormalDistrGenerator(**distr_config)

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



    def generate(self, n, max_len, device):
        batch_size = n
        z = torch.randn([batch_size, self.latent_generator.get_output_dim()]).to(device)

        decoder_hidden = self.latent2decoder_hidden(z)

        generated_indexes = torch.zeros(max_len+1, n).long().to(device)

        generated_indexes[0,:] = self.start_idx
        seq_lens = torch.ones(n)

        start_embedded_token = torch.Tensor(n).fill_(self.start_idx).long().to(device)
        start_embedded_token = self.embedding_layer(start_embedded_token)

        generated_embeddings = start_embedded_token


        for i in range(1, max_len+1):

            decoder_output_ = self.rnn_decoder(
                inputs=generated_embeddings,
                seq_lens=seq_lens,
                fill_hid=decoder_hidden
            )

            output_ = decoder_output_["rnn_out"][:, -1, :]
            decoder_hidden = decoder_output_["last_hiddens"]
            seq_lens = seq_lens + 1

            output_ = self.finalizer(output_)

            prev_token = torch.argmax(output_, dim=-1)
            generated_indexes[i, :] = prev_token

            input_ = self.embedding_layer(prev_token)
            generated_embeddings = torch.cat([generated_embeddings, input_], dim=1).view(n, -1,
                                                                                         self.embedding_layer.get_output_dim())

        return generated_indexes.view(n, -1)