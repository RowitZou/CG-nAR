import json
import pickle
import re
from os.path import join as pjoin
import numpy as np


def word_lemmatizer(sentence):
    return sentence.split()


def filter_sentence(sentence):
    return ''.join(re.findall(r'[\u4e00-\u9fff\s]+', sentence))


def _format_raw_to_json(args, line_src, line_tgt,
                        entity_vocab, idx2entity,
                        adj_matrix, pre_calculated_tail_list):
    line_src = filter_sentence(line_src)
    line_tgt = filter_sentence(line_tgt)
    ex = {}
    ex['context'] = [{'content': line_src.split()}]
    ex['target'] = line_tgt.split()
    ex['source_entity'] = list()
    ex['target_entity'] = list()
    ex['triples'] = {}

    source = ex['context'][-1]['content']
    sentence = ' '.join(source)
    source = word_lemmatizer(sentence)

    target = ex['target']
    sentence = ' '.join(target)
    target = word_lemmatizer(sentence)

    for word in source:
        if word in entity_vocab.keys():
            ex['source_entity'].append(word)
        if len('source_entity') == 0:
            ex['source_entity'].append('[UNK]')

    ex['context_keywords_list'] = [ex['source_entity']]
    for word in target:
        if word in entity_vocab.keys():
            ex['target_entity'].append(word)
        if len('target_entity') == 0:
            ex['target_entity'].append('[UNK]')

    for entity in ex['source_entity']:
        idx = entity_vocab[entity]
        # entity == '[UNK]'
        if idx == 1:
            tail_list = ['[UNK]']
            ex['triples'][entity] = tail_list
            continue
        ex['triples'][entity] = pre_calculated_tail_list[idx]
    return ex


def _pre_calculate_tail_list(adj_matrix, idx2entity):
    pre_calculated_tail_list = {}
    for from_index, row in enumerate(adj_matrix):
        row_adj = np.array(row)
        one_indices = np.where(row_adj == 1)[0]
        pre_calculated_tail_list[from_index+2] = [idx2entity[to_index+2] for to_index in one_indices]
    return pre_calculated_tail_list


def format_raw_to_jsons(args):
    corpus_types = ['train', 'valid', 'test']
    for corpus_type in corpus_types:
        dataset = []
        p_ct = 0

        src_f = open(pjoin(args.raw_path, corpus_type + '/source.txt'), encoding='utf-8')
        tgt_f = open(pjoin(args.raw_path, corpus_type + '/target.txt'), encoding='utf-8')
        # context_key_f = open(pjoin(args.raw_path, corpus_type + '/context_list.txt'), encoding='utf-8')

        entity_vocab = {}
        idx2entity = {}

        with open(args.vertex_file, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                entity_vocab[line[0]] = int(line[1])
                idx2entity[int(line[1])] = line[0]

        with open(args.adj_file, 'rb') as f:
            adj_matrix = pickle.load(f)

        pre_calculated_tail_list = _pre_calculate_tail_list(adj_matrix, idx2entity)
        for line_src in src_f:
            if args.lower:
                line_src = line_src.strip().lower()
                line_tgt = tgt_f.readline().strip().lower()
                # line_context_key = context_key_f.readline().strip().lower()
            else:
                line_src = line_src.strip()
                line_tgt = tgt_f.readline().strip()
                # line_context_key = context_key_f.readline().strip()

            dataset.append(_format_raw_to_json(args, line_src, line_tgt,
                                               entity_vocab, idx2entity,
                                               adj_matrix, pre_calculated_tail_list))

            if len(dataset) >= args.shard_size:
                if corpus_type == 'valid':
                    this_type = 'dev'
                else:
                    this_type = corpus_type
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, this_type, p_ct)
                print("Saving to " + pt_file)
                with open(pt_file, 'w', encoding='utf8') as save:
                    json.dump(dataset, save, ensure_ascii=False)
                    p_ct += 1
                    dataset = []

        if len(dataset) > 0:
            if corpus_type == 'valid':
                this_type = 'dev'
            else:
                this_type = corpus_type
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, this_type, p_ct)
            print("Saving to " + pt_file)
            with open(pt_file, 'w', encoding='utf8') as save:
                json.dump(dataset, save)
