import gc
import glob
import random
import torch

from collections import Counter
from others.logging import logger
from torchtext.vocab import Vocab


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, args, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.args = args
            self.batch_size = len(data)
            session_src = (x[2] for x in data)
            ex_segs = [len(s) for s in session_src]

            sent_src = torch.tensor(self._pad(sum((x[2] for x in data), []), 0))
            sent_segs = torch.tensor(self._pad(sum((x[3] for x in data), []), 0))
            doc_src = torch.tensor(self._pad([x[0] for x in data], 0))
            doc_segs = torch.tensor(self._pad([x[1] for x in data], 0))
            mask_sent_src = ~(sent_src == 0)
            mask_doc_src = ~(doc_src == 0)

            tgt = torch.tensor(self._pad([x[5] for x in data], 0))
            mask_tgt = ~(tgt == 0)

            setattr(self, 'sent_src', sent_src.to(device))
            setattr(self, 'doc_src', doc_src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'sent_segs', sent_segs.to(device))
            setattr(self, 'doc_segs', doc_segs.to(device))
            setattr(self, 'mask_sent_src', mask_sent_src.to(device))
            setattr(self, 'mask_doc_src', mask_doc_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))
            setattr(self, 'ex_segs', ex_segs)

            if args.use_graph:
                sc = [x[7] for x in data]
                sc_id = torch.tensor(self._pad([x[8] for x in data], 0))
                tc = [x[9] for x in data]
                tc_id = torch.tensor(self._pad([x[10] for x in data], 0))
                cc = sum((x[11] for x in data), [])
                cc_id = torch.tensor(self._pad(sum((x[12] for x in data), []), 0))
                head = torch.tensor(self._pad([x[13] for x in data], 0))
                tail = torch.tensor(self._pad([list(set().union(*x[14])) for x in data], 0))
                adj_matrix = self.make_adj_matrix(head, tail, [x[14] for x in data])
                tail_ids = dict()
                tail_ids_list = sum((x[15] for x in data), [])
                for x in tail_ids_list:
                    tail_ids.update(x)

                setattr(self, 'sc', sc)
                setattr(self, 'sc_id', sc_id.to(device))
                setattr(self, 'tc', tc)
                setattr(self, 'tc_id', tc_id.to(device))
                setattr(self, 'cc', cc)
                setattr(self, 'cc_id', cc_id.to(device))
                setattr(self, 'head', head.to(device))
                setattr(self, 'tail', tail.to(device))
                setattr(self, 'tail_ids', tail_ids)
                setattr(self, 'adj_matrix', adj_matrix.to(device))

            original_str = sum((x[4] for x in data), [])
            setattr(self, 'original_str', original_str)
            tgt_txt = [x[6] for x in data]
            setattr(self, 'tgt_txt', tgt_txt)

            if args.copy_attn:
                src_map = self.make_src_map([x[-3] for x in data])
                align = torch.tensor(self._pad([x[-2] for x in data], 0))

                setattr(self, 'src_map', src_map.to(device))
                setattr(self, 'alignment', align.to(device))

                ex_vocabs = [x[-1] for x in data]
                setattr(self, 'src_vocabs', ex_vocabs)

    def __len__(self):
        return self.batch_size

    def make_adj_matrix(self, head, tail, triple):
        bsz, head_num = head.size()
        _, tail_num = tail.size()
        matrix = torch.zeros(bsz, head_num, tail_num).bool()
        for i in range(bsz):
            for j in range(head_num):
                if j < len(triple[i]):
                    for t, e in enumerate(tail[i]):
                        if e.item() in triple[i][j]:
                            matrix[i, j, t] = 1
                else:
                    matrix[i, j, tail[i].eq(1)] = 1
        return matrix

    def make_src_map(self, data):
        src_size = max([len(t) for t in data])
        src_vocab_size = max([max(t) for t in data]) + 1
        src_map = torch.zeros(len(data), src_size, src_vocab_size)
        for i, sent in enumerate(data):
            for j, t in enumerate(sent):
                src_map[i, j, t] = 1
        return src_map


def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "dev", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def batch_size_fn(new, count):
    tgt = new[6]
    global sent_num, max_n_dec_tokens, max_n_enc_tokens
    if count == 1:
        sent_num = 0
        max_n_dec_tokens = 0
        max_n_enc_tokens = 0
    if tgt is not None:
        max_n_dec_tokens = max(max_n_dec_tokens, len(tgt))
    src_elements = count * max_n_dec_tokens + sent_num * max_n_enc_tokens
    if (count > 6):
        return src_elements + 1e3
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, batch_size, batch_ex_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.batch_ex_size = batch_ex_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args, dataset=self.cur_dataset, batch_size=self.batch_size,
                            batch_ex_size=self.batch_ex_size, device=self.device, shuffle=self.shuffle,
                            is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, batch_size, batch_ex_size,
                 device=None, is_test=False, shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.max_ex_num = batch_ex_size
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle
        self.sort_key = lambda x: len(x[1])

        # BERT special tokens
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_seg = '[unused3]'
        self.sep = '[SEP]'
        self.cls = '[CLS]'
        self.tgt_bos_id = 2
        self.tgt_eos_id = 3
        self.tgt_seg_id = 4
        self.cls_id = 101
        self.sep_id = 102

        self._iterations_this_epoch = 0
        self.batch_size_fn = batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):

        src_session = []
        segs_session = []
        txt_session = []

        session = ex["session"]
        dialogue = ex["context"]

        if self.args.use_graph:
            # source_entity target_entity
            sc = ex["source_entity"]['original_entities']
            tc = ex["target_entity"]['original_entities']
            sc_id = ex["source_entity"]['id']
            tc_id = ex["target_entity"]['id']
            # context entity list
            cc = ex['context_entity']['original_entities']
            cc_id = ex['context_entity']['id']

            # triple
            triples = ex["triples"]['id']
            head_list = list()
            tail_list = list()
            tail_id_list = list()
            for head in triples.keys():
                head_list.append(head)
                tails = set()
                tail_ids = dict()
                for tail, tail_id in triples[head].items():
                    tails.add(tail)
                    tail_ids[tail] = tail_id
                tail_list.append(tails)
                tail_id_list.append(tail_ids)

        src_ex = dialogue['src_id']
        segs_ex = dialogue['segs']

        end_id = [src_ex[-1]]
        src_ex = src_ex[:-1][:self.args.max_pos - 1] + end_id
        segs_ex = segs_ex[:self.args.max_pos]

        for sent in session:
            src = sent['src_id']
            segs = sent['segs']
            original_txt = sent['original_txt']
            end_id = [src[-1]]
            src = src[:-1][:self.args.max_pos - 1] + end_id
            segs = segs[:self.args.max_pos]

            src_session.append(src)
            segs_session.append(segs)
            txt_session.append(original_txt)

        tgt = ex["tgt"]["id"][:self.args.max_tgt_len][:-1]+[self.tgt_eos_id]
        tgt_txt = ex["tgt"]["original_txt"]

        if self.args.copy_attn:

            # build dynamic dict
            ex_vocab = Vocab(Counter(src_ex), specials=[0])

            src_map = [ex_vocab.stoi[w] for w in src_ex]

            if tgt is not None:
                align = [0] + [ex_vocab.stoi[w] if w in ex_vocab.stoi.keys() else 0 for w in tgt[1:-1]] + [0]
            else:
                align = None

        if self.args.copy_attn:
            if self.args.use_graph:
                return src_ex, segs_ex, src_session, segs_session, txt_session, tgt, tgt_txt, \
                    sc, sc_id, tc, tc_id, cc, cc_id, head_list, tail_list, tail_id_list, src_map, align, ex_vocab
            else:
                return src_ex, segs_ex, src_session, segs_session, txt_session, \
                    tgt, tgt_txt, src_map, align, ex_vocab
        else:
            if self.args.use_graph:
                return src_ex, segs_ex, src_session, segs_session, txt_session, tgt, tgt_txt, \
                    sc, sc_id, tc, tc_id, cc, cc_id, head_list, tail_list, tail_id_list
            else:
                return src_ex, segs_ex, src_session, segs_session, txt_session, tgt, tgt_txt

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex) == 0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size, max_ex_num=5):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                if len(minibatch) <= 1:
                    minibatch, size_so_far = [], 0
                    continue
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
                if size_so_far > batch_size:
                    minibatch, size_so_far = [], 0
                    continue
            if len(minibatch) >= max_ex_num:
                yield minibatch
                minibatch, size_so_far = [], 0
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = self.batch(buffer, self.batch_size, self.max_ex_num)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b) == 0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(self.args, minibatch, self.device, self.is_test)

                yield batch
            return
