import math
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from transformers import BertModel
from models.neural import MultiHeadedAttention, PositionwiseFeedForward, rnn_factory


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class Bert(nn.Module):
    def __init__(self, temp_dir, finetune=False):
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained(temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            output = self.model(x, mask.float(), segs)
            top_vec = output.last_hidden_state
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, mask, segs)
        return top_vec


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):

        pe = torch.zeros(max_len, dim)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        position = torch.arange(0, max_len).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None, add_emb=None):
        emb = emb * math.sqrt(self.dim)
        if add_emb is not None:
            emb = emb + add_emb
        if (step):
            pos = self.pe[:, step][:, None, :]
            emb = emb + pos
        else:
            pos = self.pe[:, :emb.size(1)]
            emb = emb + pos
        emb = self.dropout(emb)
        return emb


class RNNEncoder(nn.Module):
    """ A generic recurrent neural network encoder.
    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn = rnn_factory(rnn_type,
                               input_size=embeddings.embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)

    def forward(self, src, mask):

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()
        lengths = mask.sum(dim=1)

        # Lengths data is wrapped inside a Tensor.
        lengths_list = lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths_list, batch_first=True, enforce_sorted=False)

        memory_bank, encoder_final = self.rnn(packed_emb)

        memory_bank = unpack(memory_bank, batch_first=True)[0]

        return memory_bank, encoder_final


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type='self')
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout,
                 num_inter_layers=1, embeddings=None):
        super(TransformerEncoder, self).__init__()
        self.num_inter_layers = num_inter_layers
        self.hidden_size = d_model
        self.pos_emb = PositionalEncoding(dropout, d_model)
        if embeddings is not None:
            self.embeddings = embeddings
            if self.embeddings.embedding_dim != d_model:
                self.emb_to_hid = nn.Linear(self.embeddings.embedding_dim, d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, mask):
        """ See :obj:`EncoderBase.forward()`"""
        if src.dim() == 2:
            emb = self.embeddings(src)
            if emb.size(-1) != self.hidden_size:
                emb = self.emb_to_hid(emb)
        else:
            emb = src

        x = self.pos_emb(emb)

        for i in range(self.num_inter_layers):
            x = self.transformer[i](i, x, mask)  # all_sents * max_tokens * dim

        output = self.layer_norm(x)

        return output
