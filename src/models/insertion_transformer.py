import numpy as np
import torch
import torch.nn as nn
from others.utils import suggested_ed2_path
from models.decoder_tf import TransformerDecoder


class NegativeDistanceScore(object):
    '''
    functions: get the distance score to compute the loss
    '''
    def __init__(self):

        # pre-compute some values
        self.scores = {}

        self.scores[0.5] = self.compute_score_full(50, 0.5)
        self.scores[1.0] = self.compute_score_full(50, 1.0)
        self.scores[2.0] = self.compute_score_full(50, 2.0)

    def __call__(self, i, L, tau):
        '''
        parameters:
            i: the position to be inserted in the span
            L: the length of this span
            tau: the factor of softmax
        '''
        if (tau is None) or (tau > 1000):
            return 1 / L

        if tau in self.scores:
            if L < self.scores[tau].shape[0]:
                return self.scores[tau][L - 1, i]
        return self.compute_score(L, tau)[i]

    def compute_score(self, L, tau):
        # distance = softmax(-abs([(0 + L) / 2 - i]) / tau)
        s = np.array([-abs(L / 2 - i) / tau for i in range(L)])
        s = np.exp(s - s.max())
        return s / s.sum()

    def compute_score_full(self, L, tau):
        s = -abs(np.arange(0, L - 1)[:, None] / 2 - np.arange(L)[None, :]) / tau
        s = np.tril(s, 0) + np.triu(s - float("inf"), 1)
        s = np.exp(s - s.max(1, keepdims=True))
        return s / s.sum(1, keepdims=True)


neg_scorer = NegativeDistanceScore()


def get_ins_targets(in_tokens, out_tokens, padding_idx, terminal_idx, vocab_size, tau=None):
    '''
    fuctions: compute the distance score for each true-target in the span
    parameters:
    :param in_tokens: [Batch_size, sequence_length]
    :param out_tokens: [Batch_size, sequence_length] the ground truth
    :param padding_idx:
    :param vocab_size:
    :param tau:
    :return: [Batch_size, sequence_length-1, vocab_size] 每个位置插入词典中每个词的得分
    '''
    # try:
    #     from fairseq import libnat
    # except ImportError as e:
    #     import sys
    #
    #     sys.stderr.write("ERROR: missing libnat. run `pip install --editable .`\n")
    #     raise e
    B = in_tokens.size(0)
    T = in_tokens.size(1)
    V = vocab_size

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = suggested_ed2_path(
        in_tokens_list, out_tokens_list, terminal_idx
    )

    insert_labels = [a[:-1] for a in full_labels]

    insert_label_tensors = in_tokens.new_zeros(B * (T - 1) * V).float()

    insert_index, insert_labels = zip(
        *[
            (w + (j + i * (T - 1)) * V, neg_scorer(k, len(label), tau))
            for i, labels in enumerate(insert_labels)
            for j, label in enumerate(labels[1:-1])
            for k, w in enumerate(label)
        ]
    )  # HACK 1:-1
    insert_index, insert_labels = [
        torch.tensor(list(a), device=in_tokens.device)
        for a in [insert_index, insert_labels]
    ]
    insert_label_tensors.scatter_(0, insert_index.long(), insert_labels.float())
    insert_label_tensors = insert_label_tensors.view(B, T - 1, V)

    return insert_label_tensors


def apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, padding_idx):
    '''
    functions: apply the words to be inserted in the output
    :param in_tokens:
    :param in_scores:
    :param word_ins_pred:
    :param word_ins_scores:
    :param padding_idx:
    :return:
    '''
    padding_masks = in_tokens[:, 1:].eq(padding_idx)
    word_ins_scores.masked_fill_(padding_masks, 0.0)
    word_ins_pred.masked_fill_(padding_masks, padding_idx)

    in_coords = new_arange(in_tokens).type_as(in_scores)

    # shift all padding predictions to infinite
    out_coords = (in_coords[:, 1:] - 0.5).masked_fill(
        word_ins_pred.eq(padding_idx), float("inf")
    )
    out_coords = torch.cat([in_coords, out_coords], 1).sort(-1)[1]
    out_tokens = torch.cat([in_tokens, word_ins_pred], 1).gather(1, out_coords)
    out_scores = torch.cat([in_scores, word_ins_scores], 1).gather(1, out_coords)
    return out_tokens, out_scores


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


class InsertionTransformerDecoder(TransformerDecoder):
    def __init__(self, args, embeddings):

        super(InsertionTransformerDecoder, self).__init__(
            args.dec_layers, args.dec_hidden_size,
            args.dec_heads, args.dec_ff_size,
            args.dec_dropout, embeddings
        )
        self.pool_out = nn.Linear(args.dec_hidden_size * 2, args.dec_hidden_size)
        self.label_tau = getattr(args, "tau", None)

    def forward_word_ins(self, enc_out, prev_output_tokens, enc_mask, dec_state, step=None):
        features, states, _ = self.forward(prev_output_tokens, enc_out, dec_state,
                                           memory_masks=enc_mask, step=step, self_attn=False)
        features = self.pool_out(
            torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        )
        return features, states
