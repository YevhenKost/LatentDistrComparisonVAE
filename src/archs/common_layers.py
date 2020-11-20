import torch.nn as nn
import torch
from torch.autograd import Variable
from archs.fix import hotfix_pack_padded_sequence

def get_relu_sequantial(input_dim, layers, output_dim, last_activation):

    seq_layers = nn.ModuleList(
        [nn.Linear(input_dim, layers[0])])
    seq_layers.extend([nn.ReLU()])

    for i in range(len(layers) - 1):
        seq_layers.extend([nn.Linear(layers[i], layers[i + 1])])
        seq_layers.extend([nn.ReLU()])

    seq_layers.extend([nn.Linear(layers[-1], output_dim)])
    if last_activation:
        seq_layers.extend([last_activation])

    seq_layers = nn.Sequential(*seq_layers)
    return seq_layers

class ExtendedEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(ExtendedEmbeddingLayer, self).__init__()

        self.emb = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.out_dim = embedding_dim
        self.num_voc = num_embeddings

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[0], -1).long()
        return self.emb(inputs)
    def get_output_dim(self):
        return self.out_dim
    def get_num_voc(self):
        return self.num_voc


class RNNLayer(nn.Module):
    def __init__(self, rnn_type="GRU", input_dim=2,hidden_dim=2,num_cells=1,dropout=0.1, bidir=True):
        super(RNNLayer, self).__init__()

        if rnn_type == "GRU":
            self.rnn_cell = nn.GRU(input_dim, hidden_dim, num_layers=num_cells,dropout=dropout,
                              batch_first=True, bidirectional=bidir)
        if rnn_type == "LSTM":
            self.rnn_cell = nn.LSTM(input_dim, hidden_dim,num_layers=num_cells,dropout=dropout,
                                  batch_first=True, bidirectional=bidir)
        if rnn_type == "RNN":
            self.rnn_cell = nn.RNN(input_dim, hidden_dim,num_layers=num_cells,dropout=dropout,
                                   batch_first=True, bidirectional=bidir)

        self.rnn_type = rnn_type
        self.bidir = bidir
        self.rnn_input_dim = input_dim
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        self.num_dirs = 1 + self.bidir

    def get_output_dim(self):
        return self.hidden_dim*self.num_dirs
    def get_hidden_shape(self):
        return self.num_cells*self.num_dirs*self.hidden_dim

    def _sort(self, seq_lens, batch):
        sorted_lengths, sorted_idx = torch.sort(seq_lens, descending=True)
        sorted_batch = batch[sorted_idx]
        return sorted_lengths, sorted_batch, sorted_idx

    def _unsort(self, sorted_idx, batch):
        _, reversed_idx = torch.sort(sorted_idx)
        unsorted_batch = batch[reversed_idx]
        return unsorted_batch

    def _get_start_hidden(self, batch_size, device, fill_hid=None):

        if not isinstance(fill_hid, torch.Tensor):
            hidden_el = Variable(torch.zeros(self.num_dirs*self.num_cells, batch_size, self.hidden_dim).to(device))
        else:
            if len(fill_hid.view(-1)) == self.num_dirs*self.num_cells * batch_size * self.hidden_dim:
                hidden_el = fill_hid.view(self.num_dirs*self.num_cells, batch_size, self.hidden_dim)
            else:
                hidden_el = torch.stack([fill_hid]*(self.num_dirs*self.num_cells))
                hidden_el = hidden_el.view(self.num_dirs*self.num_cells, batch_size, self.hidden_dim).to(device)

        if self.rnn_type == "LSTM":
            return (hidden_el, hidden_el)

        return hidden_el

    def forward(self, inputs, seq_lens, fill_hid=None):

        batch_size = inputs.shape[0]
        device = inputs.device

        # sorting inputs and fill_hid
        sorted_lengths, sorted_inputs, sorted_idx = self._sort(seq_lens, inputs)
        if isinstance(fill_hid, torch.Tensor):
            fill_hid = self._sort(seq_lens, fill_hid)[1]

        hidden = self._get_start_hidden(batch_size, device, fill_hid)

        try:
            pack = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_lengths, batch_first=True)
        except:
            pack = hotfix_pack_padded_sequence(sorted_inputs, sorted_inputs, batch_first=True)

        rnn_out, last_hiddens = self.rnn_cell(pack, hidden)
        last_hiddens = last_hiddens.view(batch_size, -1)

        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        # unsorting
        rnn_out = self._unsort(sorted_idx,rnn_out)
        last_hiddens = self._unsort(sorted_idx, last_hiddens)


        out_dict = {

            "rnn_out": rnn_out,
            "last_hiddens": last_hiddens

        }
        return out_dict