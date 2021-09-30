import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.categorical import Categorical

from models.encoder import Bert, TransformerEncoder
from models.decoder_tf import TransformerDecoder
from models.insertion_transformer import InsertionTransformerDecoder
from models.insertion_transformer import apply_ins_words, get_ins_targets
from models.generator import Generator
from models.graph import GraphEncoder, GraphGenerator


class Model(nn.Module):
    def __init__(self, args, device, vocab, checkpoint=None):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.vocab_size = len(vocab)
        self.max_step = args.max_dec_step
        self.min_step = args.min_dec_step
        self.max_concept_num = args.max_concept_num
        self.min_concept_num = args.min_concept_num

        # special tokens
        self.start_token = vocab['[unused1]']
        self.end_token = vocab['[unused2]']
        self.pad_token = vocab['[PAD]']
        self.mask_token = vocab['[MASK]']
        self.seg_token = vocab['[SEP]']
        self.cls_token = vocab['[CLS]']
        self.terminal_token = vocab['[unused3]']
        self.concept_unk = 1

        # graph embeddings
        graph_weights = np.load(args.graph_emb_path).astype(np.float32)
        self.graph_embeddings = nn.Embedding.from_pretrained(
            torch.from_numpy(graph_weights), freeze=False, padding_idx=0)

        # sent encoder
        if args.encoder == 'bert':
            self.encoder = Bert(args.bert_dir, args.finetune_bert)
        else:
            if args.vocab_emb_path != '':
                emb_weights = np.load(args.vocab_emb_path)
                embeddings = nn.Embedding.from_pretrained(
                    torch.from_numpy(emb_weights), freeze=False, padding_idx=0)
            else:
                embeddings = nn.Embedding(self.vocab_size, args.enc_hidden_size, padding_idx=0)
                self._set_parameter_embedding(embeddings)
            self.encoder = TransformerEncoder(args.enc_hidden_size, args.enc_ff_size, args.enc_heads,
                                              args.enc_dropout, args.enc_layers, embeddings)

        # concept encoder
        self.concept_encoder = GraphEncoder(args.ce_hidden_size, args.ce_dropout,
                                            self.graph_embeddings)

        # hier encoder
        self.hier_encoder = TransformerEncoder(args.hier_hidden_size, args.hier_ff_size, args.hier_heads,
                                               args.hier_dropout, args.hier_layers)

        # concept decoder
        self.concept_dec_init_token = nn.Parameter(torch.empty([1, args.cd_hidden_size]))
        xavier_uniform_(self.concept_dec_init_token)
        self.concept_decoder = TransformerDecoder(
            args.cd_layers, args.cd_hidden_size, args.cd_heads,
            args.cd_ff_size, args.cd_dropout, self.graph_embeddings
        )

        # decoder
        if args.vocab_emb_path != '':
            emb_weights = np.load(args.vocab_emb_path)
            tgt_embeddings = nn.Embedding.from_pretrained(
                torch.from_numpy(emb_weights), freeze=False, padding_idx=0)
        else:
            tgt_embeddings = nn.Embedding(self.vocab_size, args.dec_hidden_size, padding_idx=0)
            self._set_parameter_embedding(tgt_embeddings)
        self.decoder = InsertionTransformerDecoder(args, tgt_embeddings)

        # generator
        self.concept_generator = GraphGenerator(args.cg_hidden_size, args.cg_dropout,
                                                self.graph_embeddings)

        self.generator = Generator(self.vocab_size, args.dec_hidden_size, self.pad_token)

        self.generator.linear.weight = self.decoder.embeddings.weight

        # graph embedding
        # self.graph_process = Graph_process(args)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
            if args.share_emb:
                self.generator.linear.weight = self.decoder.embeddings.weight
        else:
            # initialize params.
            if args.encoder != "bert":
                for module in self.encoder.modules():
                    self._set_parameter_tf(module)
            for module in self.hier_encoder.modules():
                self._set_parameter_tf(module)
            for module in self.concept_decoder.modules():
                self._set_parameter_tf(module)
            for module in self.decoder.modules():
                self._set_parameter_tf(module)
            for module in self.concept_encoder.modules():
                self._set_parameter_linear(module)
            for module in self.concept_generator.modules():
                self._set_parameter_linear(module)
            for module in self.generator.modules():
                self._set_parameter_linear(module)

            if args.share_emb:
                if args.encoder == 'bert':
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.model.config.hidden_size, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.model.embeddings.word_embeddings.weight)
                    self.decoder.embeddings = tgt_embeddings
                else:
                    tgt_embeddings = nn.Embedding(self.vocab_size, self.encoder.embeddings.embedding_dim, padding_idx=0)
                    tgt_embeddings.weight = copy.deepcopy(self.encoder.embeddings.weight)
                    self.decoder.embeddings = tgt_embeddings
                self.generator.linear.weight = self.decoder.embeddings.weight

        self.to(device)

    def _set_parameter_tf(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_linear(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _set_parameter_embedding(self, emb):
        emb.weight.data.normal_(mean=0.0, std=0.02)

    def _get_concept_tgt(self, tail, tgt):
        """
        tail: bsz * tail_num
        tgt: bsz * tgt_num
        """
        bsz, tail_num = tail.size()
        _, tgt_num = tgt.size()

        one_hot_tgt = torch.logical_and(
            (tail.unsqueeze(1).expand(bsz, tgt_num, tail_num) -
             tgt.unsqueeze(-1).expand(bsz, tgt_num, tail_num)).eq(0),
            (tail.unsqueeze(1).expand(bsz, tgt_num, tail_num) +
             tgt.unsqueeze(-1).expand(bsz, tgt_num, tail_num)).ne(0)
        )

        # Avoid duplicate extraction.
        dup_mask = torch.zeros_like(one_hot_tgt)
        for i in range(one_hot_tgt.size(0)):
            for j in range(1, one_hot_tgt.size(1)):
                if one_hot_tgt[i, j].sum() != 0:
                    dup_mask[i, j] += (dup_mask[i, j-1] + one_hot_tgt[i, j-1])
                else:
                    break
        return one_hot_tgt, dup_mask.bool()

    def _get_concept_tokens(self, concept, tail_ids, txt=None):
        bsz = len(concept)
        concept_token_list = []
        for idx in range(bsz):
            con_list = []
            for item in concept[idx]:
                item = int(item)
                if item != 0 and item != self.concept_unk:
                    item_ids = tail_ids[item]
                    if txt is not None:
                        for item_id in item_ids:
                            if txt[idx].eq(item_id).sum() == 0:
                                item_ids.remove(item_id)
                    con_list.extend(item_ids)
            concept_token_list.append(con_list)

        return concept_token_list

    def _add_concept_subseq(self, target_tokens, prev_target_tokens, concept_tokens):
        target_list = target_tokens.cpu().numpy().tolist()
        prev_list = prev_target_tokens.cpu().numpy().tolist()
        for batch_idx in range(len(target_list)):
            len_concept = len(concept_tokens[batch_idx])
            len_target = len(target_list[batch_idx])
            len_prev = len(prev_list[batch_idx])
            idx_target = 0
            idx_prev = 0
            while(len_concept != 0 and idx_target < len_target and idx_prev < len_prev):
                if target_list[batch_idx][idx_target] == prev_list[batch_idx][idx_prev]:
                    idx_target += 1
                    idx_prev += 1
                    continue
                elif target_list[batch_idx][idx_target] == concept_tokens[batch_idx][0]:
                    idx_target += 1
                    prev_list[batch_idx].insert(idx_prev, concept_tokens[batch_idx].pop(0))
                    len_concept = len(concept_tokens[batch_idx])
                    len_prev = len(prev_list[batch_idx])
                    idx_prev += 1
                else:
                    idx_target += 1
            while len_concept != 0:
                prev_list[batch_idx].insert(idx_prev, concept_tokens[batch_idx].pop(0))
                len_concept = len(concept_tokens[batch_idx])
                idx_prev += 1
            while prev_list[batch_idx][-1] == 0:
                prev_list[batch_idx].pop(-1)
        max_len = len(max(prev_list, key=len))
        prev_tensor = torch.zeros([len(target_list), max_len], dtype=torch.int64)
        for idx in range(len(prev_list)):
            prev_tensor[idx][:len(prev_list[idx])] = torch.tensor(prev_list[idx])
        prev_tensor = prev_tensor.to(self.device)
        return prev_tensor

    def _get_sampled_subseq(self, target_tokens):
        pad = self.pad_token
        bos = self.start_token
        eos = self.end_token

        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(
            target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
        )
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(
            1, keepdim=True
        )

        # do not delete <bos> and <eos> (we assign 0 score for them)
        target_cutoff = (
            2
            + (
                (target_length - 2)
                * target_score.new_zeros(target_score.size(0), 1).uniform_()
            ).long()
        )

        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = (
            target_tokens.gather(1, target_rank)
            .masked_fill_(target_cutoff, pad)
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
        prev_target_tokens = prev_target_tokens[
            :, : prev_target_tokens.ne(pad).sum(1).max()
        ]

        sort_target_tokens, sort_rank = target_tokens.sort(1)

        return prev_target_tokens

    def _concept_decoding(self, batch, memory_bank, history, init_tokens,
                          memory_mask=None, max_length=5, min_length=1, method="sample"):

        head = batch.head
        tail = batch.tail
        adj_mask = batch.adj_matrix

        bsz, mem_len, _ = memory_bank.size()
        _, head_num, tail_num = adj_mask.size()

        dec_states = self.concept_decoder.init_decoder_state(
            batch.doc_src, memory_bank, with_cache=True
        )

        alive_seq = torch.full(
            [bsz, 1],
            self.concept_unk,
            dtype=torch.long,
            device=self.device)

        dup_mask = adj_mask.new_zeros([bsz, 1, tail_num])
        pred_label = torch.tensor([], device=self.device, dtype=torch.long)
        pred_prob = torch.tensor([], device=self.device)

        # record batch id
        batch_idx = torch.arange(bsz, device=self.device)

        # extracted tail concept length
        tail_length = tail.ne(0).sum(1)

        # Structure that holds finished hypotheses.
        results = [[] for _ in range(bsz)]

        for step in range(max_length):

            if step > 0:
                init_tokens = None

            # Decoder forward.
            decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states, _ = self.concept_decoder(decoder_input, memory_bank, dec_states,
                                                          init_tokens=init_tokens, step=step,
                                                          memory_masks=memory_mask)

            # Generator forward.
            log_probs = self.concept_generator(dec_out, history, head, tail, adj_mask, dup_mask)

            if step < min_length:
                if log_probs.requires_grad:
                    mask = torch.zeros_like(log_probs)
                    mask[tail.unsqueeze(1).eq(self.concept_unk)] = -1e20
                    log_probs = log_probs + mask
                else:
                    log_probs[tail.unsqueeze(1).eq(self.concept_unk)] = -1e20

            # greedy selection or sampling
            if method == "sample":
                m = Categorical(logits=log_probs)
                ids = m.sample()
                scores = m.log_prob(ids)
            else:
                scores, ids = log_probs.max(dim=-1)

            # Avoid duplicate extraction
            dup_mask = torch.logical_or(dup_mask, F.one_hot(ids, tail_num).bool())

            # Append last prediction.
            last_pre = torch.cat([tail[i][ids[i]].unsqueeze(0) for i in range(ids.size(0))], 0)
            alive_seq = torch.cat([alive_seq, last_pre], 1)

            # Append last pred label and probability
            pred_label = torch.cat([pred_label, last_pre], -1)
            pred_prob = torch.cat([pred_prob, scores], -1)

            # finished if at stop state
            is_finished = last_pre.eq(self.concept_unk)

            # finished if exceeds max length
            if step + 1 == max_length:
                is_finished.fill_(1)

            # finished if candidate concepts are fewer than min length
            for i in range(len(pred_label)):
                if pred_label.size(-1) >= tail_length[batch_idx[i]] - 1:
                    is_finished[i] = 1

            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                for i in range(is_finished.size(0)):
                    # Store finished hypotheses for this batch.
                    # If the batch reached the end, save the results.
                    if end_condition[i]:
                        if pred_label[i, -1].item() == self.concept_unk:
                            results[batch_idx[i]].append((pred_label[i, :-1], pred_prob[i], pred_label[i]))
                        else:
                            results[batch_idx[i]].append((pred_label[i], pred_prob[i], pred_label[i]))
                non_finished = torch.nonzero(end_condition.eq(0)).view(-1)
                # If all examples are finished, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                alive_seq = alive_seq.index_select(0, non_finished)
                batch_idx = batch_idx.index_select(0, non_finished)
                pred_label = pred_label.index_select(0, non_finished)
                pred_prob = pred_prob.index_select(0, non_finished)
                memory_bank = memory_bank.index_select(0, non_finished)
                memory_mask = memory_mask.index_select(0, non_finished)
                dup_mask = dup_mask.index_select(0, non_finished)
                head = head.index_select(0, non_finished)
                tail = tail.index_select(0, non_finished)
                adj_mask = adj_mask.index_select(0, non_finished)
                history = history.index_select(0, non_finished)
                dec_states.map_batch_fn(
                    lambda state, dim: state.index_select(dim, non_finished))
        return results

    def _fast_translate_batch(self, batch, memory_bank, max_step, memory_mask=None,
                              min_step=0, concept_tokens=None):

        batch_size = memory_bank.size(0)

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=self.device)

        if concept_tokens is None:
            alive_seq = torch.tensor(
                [self.start_token, self.end_token],
                dtype=torch.long, device=self.device).\
                unsqueeze(0).expand(batch_size, 2)
        else:
            concept_tokens = [
                torch.tensor(
                    [self.start_token]+t+[self.end_token],
                    dtype=torch.long, device=self.device
                )
                for t in concept_tokens]

            alive_seq = pad_sequence(concept_tokens, batch_first=True, padding_value=0)

        # Structure that holds finished hypotheses.
        results = [[] for _ in range(batch_size)]

        output_scores = alive_seq.new_zeros(
            *alive_seq.size()).type_as(memory_bank)

        def is_a_loop(x, y, s):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad_token)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad_token)], 1)
            return (x == y).all(1), y, s

        for step in range(max_step):
            # Decoder forward.

            dec_states = self.decoder.init_decoder_state(batch.doc_src, memory_bank)

            dec_out, dec_states = self.decoder.forward_word_ins(memory_bank, alive_seq,
                                                                memory_mask, dec_states)

            # Generator forward.
            log_probs = self.generator(dec_out.view(-1, dec_out.size(-1))).\
                view(dec_out.size(0), -1, self.vocab_size)

            if step < min_step:
                log_probs[:, :, self.terminal_token] = -1e20

            # length penalty
            alpha = self.args.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            log_probs = log_probs / length_penalty

            top_score, top_pred_id = log_probs.max(-1)

            out_seq, out_scores = apply_ins_words(
                alive_seq, output_scores, top_pred_id,
                top_score, self.terminal_token
            )

            # delete some unnecessary paddings
            out_seq.masked_fill_(out_seq.eq(self.terminal_token), self.pad_token)
            cut_off = out_seq.ne(self.pad_token).sum(1).max()
            out_seq = out_seq[:, :cut_off]
            out_scores = out_scores[:, :cut_off]

            is_finished, alive_seq, output_scores = is_a_loop(
                alive_seq,
                out_seq,
                out_scores,
            )

            if step + 1 == max_step:
                is_finished.fill_(1)

            # Save finished hypotheses.
            if is_finished.any():
                for i in range(is_finished.size(0)):
                    if is_finished[i]:
                        b = batch_offset[i]
                        results[b].append(alive_seq[i, 1:])
                non_finished = torch.nonzero(is_finished.eq(0)).view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = alive_seq.index_select(0, non_finished)
                output_scores = output_scores.index_select(0, non_finished)

                # Reorder states.
                if memory_bank is not None:
                    memory_bank = memory_bank.index_select(0, non_finished)
                if memory_mask is not None:
                    memory_mask = memory_mask.index_select(0, non_finished)

        results = [t[0] for t in results]
        return results

    def forward(self, batch):

        # text info
        src_tokens = batch.sent_src
        tgt_tokens = batch.tgt
        src_segs = batch.sent_segs
        mask_src = batch.mask_sent_src
        ex_segs = batch.ex_segs

        # concept info
        context_c = batch.cc_id
        tgt_c = batch.tc_id
        head = batch.head
        tail = batch.tail
        adj_mask = batch.adj_matrix
        tail_ids = batch.tail_ids

        # sent encoding
        if self.args.encoder == "bert":
            sent_out = self.encoder(src_tokens, src_segs, mask_src)
        else:
            sent_out = self.encoder(src_tokens, ~mask_src)

        # hierarchical encoding
        sent_list = torch.split(sent_out[:, 0, :], ex_segs)
        sent_hid = pad_sequence(sent_list, batch_first=True, padding_value=0.)
        sent_mask_list = [mask_src.new_zeros([length]) for length in ex_segs]
        sent_mask = pad_sequence(sent_mask_list, batch_first=True, padding_value=1)

        hier_hid = self.hier_encoder(sent_hid, sent_mask)

        # Concept Predictor
        context_c_hid = self.concept_encoder(context_c, ex_segs)

        # concept decoder input
        cd_input = torch.cat([tgt_c.new_ones([len(batch), 1]), tgt_c], -1)
        cd_init_tokens = self.concept_dec_init_token.expand(len(batch), -1)

        if self.training:

            # Generate concept target
            concept_tgt, dup_mask = self._get_concept_tgt(tail, tgt_c)
            setattr(batch, 'concept_tgt', concept_tgt)

            dec_state = self.concept_decoder.init_decoder_state(src_tokens, hier_hid)
            cd_output, _, _ = self.concept_decoder(cd_input[:, :-1], hier_hid, dec_state,
                                                   cd_init_tokens, memory_masks=sent_mask)
            concept_pred = self.concept_generator(cd_output, context_c_hid, head, tail,
                                                  adj_mask, dup_mask)
            concept_result = None
        else:
            concept_pred = None
            concept_result = self._concept_decoding(batch, hier_hid, context_c_hid, cd_init_tokens,
                                                    sent_mask, self.max_concept_num,
                                                    self.min_concept_num, method="greedy")
            concept_result = [t[0][0].tolist() for t in concept_result]

        # Insertion Transformer
        # Training Stage
        if self.training:

            concept_tokens = self._get_concept_tokens(tgt_c, tail_ids, tgt_tokens)

            prev_output_tokens = self._get_sampled_subseq(batch.tgt)
            prev_output_tokens = self._add_concept_subseq(batch.tgt, prev_output_tokens, concept_tokens)

            dec_state = self.decoder.init_decoder_state(src_tokens, hier_hid)
            ins_out, _ = self.decoder.forward_word_ins(
                hier_hid, prev_output_tokens, sent_mask, dec_state
            )

            ins_pred = self.generator(ins_out.view(-1, ins_out.size(-1))).\
                view(ins_out.size(0), -1, self.vocab_size)

            # generate training labels for insertion
            # word_ins_tgt = [B, T-1, V]
            word_ins_tgt = get_ins_targets(
                prev_output_tokens,
                tgt_tokens,
                self.pad_token,
                self.terminal_token,
                self.vocab_size,
                tau=self.decoder.label_tau,
            ).type_as(ins_out)

            setattr(batch, 'ins_tgt', word_ins_tgt)
            ins_result = None

        # Testing stage
        else:
            concept_tokens = self._get_concept_tokens(concept_result, tail_ids)
            ins_result = self._fast_translate_batch(batch, hier_hid, self.max_step,
                                                    memory_mask=sent_mask,
                                                    min_step=self.min_step,
                                                    concept_tokens=concept_tokens)
            ins_pred = None

        return concept_pred, concept_result, ins_pred, ins_result
