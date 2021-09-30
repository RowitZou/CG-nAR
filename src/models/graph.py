import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.neural import GlobalAttention


class GraphEncoder(nn.Module):
    def __init__(self, hid_dim, dropout, embeddings):
        super(GraphEncoder, self).__init__()
        self.embeddings = embeddings
        self.emb_2_hid = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embeddings.embedding_dim, hid_dim),
            nn.ELU()
        )
        self.initial_state = nn.Parameter(torch.empty([1, hid_dim]))
        nn.init.xavier_uniform_(self.initial_state)
        self.rnn_cell = nn.GRUCell(hid_dim, hid_dim)
        self.attn_layer = GlobalAttention(hid_dim, attn_type='general')

    def forward(self, concept, ex_segs):
        """
        emb: [batch_size * sent_num, seq_len, dim]
        ex_segs: [lengths]
        """
        emb = self.embeddings(concept)
        lengths = concept.ne(0).sum(-1)

        emb_hid = self.emb_2_hid(emb)
        emb_hid_list = torch.split(emb_hid, ex_segs)
        emb_hid = pad_sequence(emb_hid_list, batch_first=True, padding_value=0.)

        lengths_list = torch.split(lengths, ex_segs)
        lengths = pad_sequence(lengths_list, batch_first=True, padding_value=1)

        bsz, sent_num, seq_len, dim = emb_hid.size()

        h_list = []
        h = self.initial_state.expand(bsz, dim)
        for idx in range(sent_num):
            x, _ = self.attn_layer(h, emb_hid[:, idx], lengths[:, idx])
            h = self.rnn_cell(x, h)
            h_list.append(h.unsqueeze(1))

        # get the final state at (t-1)-th time step.
        indexs = F.one_hot(torch.tensor(ex_segs).to(h.device)-1).bool()
        final_h = torch.cat(h_list, 1)[indexs]
        return final_h


class GraphGenerator(nn.Module):
    def __init__(self, hid_dim, dropout, embeddings):
        super(GraphGenerator, self).__init__()
        self.embeddings = embeddings
        emb_dim = embeddings.embedding_dim
        self.tail_attn_linear_q = nn.Sequential(
            nn.Linear(hid_dim*2+emb_dim, hid_dim),
            nn.ELU()
        )
        self.tail_attn_linear_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hid_dim),
            nn.ELU()
        )
        self.tail_attn = GlobalAttention(hid_dim)

        self.head_attn_linear_q = nn.Sequential(
            nn.Linear(hid_dim*2, hid_dim),
            nn.ELU()
        )
        self.head_attn_linear_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ELU()
        )
        self.head_attn = GlobalAttention(hid_dim)

    def forward(self, dec_out, history, head, tail, adj_mask, dup_mask):
        """
        args:

        dec_out: bsz * step_num * hid
        history: bsz * hid
        head: bsz * head_num
        tail: bsz * tail_num
        adj_mask: bsz * head_num * tail_num
        dup_mask: bsz * step_num * tail_num

        return:

        prob: bsz * step_num * tail_num
        """
        head_emb = self.embeddings(head)
        tail_emb = self.embeddings(tail)

        bsz, step_num, hid_dim = dec_out.size()
        _, head_num, emb_dim = head_emb.size()
        _, tail_num, _ = tail_emb.size()

        # tail level attention
        tail_attn_q = self.tail_attn_linear_q(
            torch.cat([
                dec_out.unsqueeze(2).expand(bsz, step_num, head_num, -1),
                history.unsqueeze(1).unsqueeze(1).expand(bsz, step_num, head_num, -1),
                head_emb.unsqueeze(1).expand(bsz, step_num, head_num, -1)
            ], -1)
        ).view(bsz, -1, hid_dim)

        tail_attn_k = self.tail_attn_linear_k(tail_emb)

        tail_attn_mask = torch.logical_and(
            adj_mask.unsqueeze(1).expand(bsz, step_num, head_num, tail_num),
            ~dup_mask.unsqueeze(2).expand(bsz, step_num, head_num, tail_num),
        ).view(bsz, -1, tail_num)

        # graph_hid: step_num*head_num, bsz, hid_dim
        graph_hid, tail_attn = self.tail_attn(tail_attn_q, tail_attn_k,
                                              memory_masks=tail_attn_mask)

        graph_hid = graph_hid.transpose(0, 1).reshape(-1, head_num, hid_dim)
        tail_attn = tail_attn.transpose(0, 1).reshape(-1, head_num, tail_num)

        head_attn_q = self.head_attn_linear_q(
            torch.cat([
                dec_out,
                history.unsqueeze(1).expand(bsz, step_num, -1),
            ], -1)
        ).view(-1, hid_dim)

        head_attn_k = self.head_attn_linear_k(graph_hid)

        head_attn_length = head.ne(0).sum(-1).unsqueeze(1).\
            expand(bsz, step_num).reshape(-1)

        _, head_attn = self.head_attn(head_attn_q, head_attn_k,
                                      head_attn_length)

        prob = (tail_attn * head_attn.unsqueeze(-1)).sum(1).\
            view(bsz, step_num, -1)
        return (prob+1e-20).log()
