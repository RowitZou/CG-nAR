# -*- coding:utf-8 -*-

import gc
import glob
import json
import os
import re
import random
import torch
from os.path import join as pjoin

from multiprocess import Pool
from others.logging import logger
from transformers import BertTokenizer


def clean(sent):
    needed_char = '''a-zA-Z0-9,.!:'‘’()?\-\$\"'''
    ret = []
    for word in sent:
        cleaned_word = ''.join([char for char in word if re.match("^[" + needed_char + "]*$", char)])
        if len(cleaned_word) > 0:
            ret.append(cleaned_word)
    return ret


class GraphData():
    def __init__(self, args):
        self.entity2idx = {}
        self.idx2entity = {}
        self.args = args

        with open(args.vertex_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split(' ')
                self.entity2idx[line[0]] = int(line[1])
                self.idx2entity[int(line[1])] = line[0]

    def preprocess_src_entities(self, content):
        raw_list = list(set(content))
        idx_list = [self.entity2idx[entity] for entity in raw_list]
        if len(idx_list) == 0:
            idx_list.append(1)
            raw_list.append(self.idx2entity[1])
        return raw_list, idx_list

    def preprocess_tgt_entities(self, content):
        count = 0
        raw_list = sorted(set(content), key=content.index)
        idx_list = [self.entity2idx[entity] for entity in raw_list]
        if len(idx_list) != 0:
            count += 1
        if len(idx_list) == 0 or idx_list[-1] != 1:
            idx_list.append(1)
            raw_list.append(self.idx2entity[1])
        return raw_list, idx_list, count

    def preprocess_context(self, content):
        raw_list = list()
        idx_list = list()
        for item in content:
            item = set(item)
            idx_list_sub = [self.entity2idx[entity] for entity in item
                            if entity in self.entity2idx.keys()]
            raw_list_sub = [entity for entity in item if entity in self.entity2idx.keys()]
            if len(idx_list_sub) == 0:
                idx_list_sub.append(1)
                raw_list_sub.append(self.idx2entity[1])
            idx_list.append(idx_list_sub)
            raw_list.append(raw_list_sub)
        return raw_list, idx_list

    def preprocess_triples(self, content, tgt, tokenizer):
        raw_triple = content
        idx_triple = {}
        if len(raw_triple) == 0:
            raw_triple[self.idx2entity[1]] = []
        for head in raw_triple.keys():
            random.shuffle(raw_triple[head])
            raw_triple[head] = raw_triple[head][:self.args.tail_sample_num]
            raw_triple[head].extend(tgt)
            raw_triple[head].append(self.idx2entity[1])
            head_idx = self.entity2idx[head]
            idx_triple[head_idx] = dict()
            for tail in raw_triple[head]:
                idx_triple[head_idx][self.entity2idx[tail]] = \
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tail))
        return raw_triple, idx_triple


class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.tgt_bos = '[unused1]'
        self.tgt_eos = '[unused2]'
        self.tgt_seg = '[unused3]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.unk_vid = self.tokenizer.vocab[self.unk_token]

    def preprocess_src(self, content):
        if_exceed_length = False

        if len(content) < self.args.min_src_ntokens_per_sent:
            return None
        if len(content) > self.args.max_src_ntokens_per_sent:
            if_exceed_length = True

        original_txt = ' '.join(content)

        if self.args.truncated:
            content = content[:self.args.max_src_ntokens_per_sent]
        content_text = ' '.join(content)
        content_subtokens = self.tokenizer.tokenize(content_text)

        # [CLS] + T0 + T1 + ... + Tn
        src_subtokens = [self.cls_token] + content_subtokens
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        segments_ids = len(src_subtoken_idxs) * [0]

        return src_subtoken_idxs, segments_ids, original_txt, \
            src_subtokens, if_exceed_length

    def preprocess_target(self, content):

        content_subtokens = self.tokenizer.tokenize(' '.join(content))[:self.args.max_tgt_len]
        original_txt = ' '.join(content_subtokens).replace(' ##', '')
        content_subtokens = [self.tgt_bos] + content_subtokens + [self.tgt_eos]
        subtoken_idxs = self.tokenizer.convert_tokens_to_ids(content_subtokens)

        return subtoken_idxs, original_txt, content_subtokens

    def integrate_dial(self, doc):
        src_tokens = [self.cls_token]
        segments_ids = [0]
        segment_id = 0
        for sent in doc:
            tokens = sent["src_tokens"][1:] + [self.sep_token]
            src_tokens.extend(tokens)
            segments_ids.extend([segment_id] * len(tokens))
            segment_id = 1 - segment_id
        src_ids = self.tokenizer.convert_tokens_to_ids(src_tokens)
        return {"src_id": src_ids, "segs": segments_ids}


def format_json_to_data(args, corpus_type=None):
    a_lst = []
    if corpus_type is not None:
        for json_f in glob.glob(pjoin(args.raw_path, '*' + corpus_type + '*.json')):
            real_name = json_f.split('/')[-1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))
    else:
        for json_f in glob.glob(pjoin(args.raw_path, '*.json')):
            real_name = json_f.split('/')[-1]
            corpus_type = real_name.split('.')[1]
            a_lst.append((corpus_type, json_f, args, pjoin(args.save_path, real_name.replace('json', 'pt'))))

    total_statistic = {
        "instances": 0,
        "total_sents": 0.,
        "processed_sents": 0.,
        "max_sents": -1,
        "sents_num": [0] * 11,
        "exceed_length_num": 0,
        "exceed_sents_num": 0,
        "total_src_length": 0.,
        "src_sent_length_num": [0] * 11,
        "src_token_length_num": [0] * 11,
        "total_tgt_length": 0,
        "total_tgt_concept": 0
    }

    pool = Pool(args.n_cpus)
    for statistic in pool.imap(_format_json_to_data, a_lst):
        if statistic is None:
            continue
        total_statistic["instances"] += statistic["instances"]
        total_statistic["total_sents"] += statistic["total_sents"]
        total_statistic["processed_sents"] += statistic["processed_sents"]
        total_statistic["max_sents"] = max(total_statistic["max_sents"], statistic["max_sents"])
        total_statistic["exceed_length_num"] += statistic["exceed_length_num"]
        total_statistic["exceed_sents_num"] += statistic["exceed_sents_num"]
        total_statistic["total_src_length"] += statistic["total_src_length"]
        total_statistic["total_tgt_length"] += statistic["total_tgt_length"]
        total_statistic["total_tgt_concept"] += statistic["total_tgt_concept"]

        for idx in range(len(total_statistic["sents_num"])):
            total_statistic["sents_num"][idx] += statistic["sents_num"][idx]
        for idx in range(len(total_statistic["src_sent_length_num"])):
            total_statistic["src_sent_length_num"][idx] += statistic["src_sent_length_num"][idx]
        for idx in range(len(total_statistic["src_token_length_num"])):
            total_statistic["src_token_length_num"][idx] += statistic["src_token_length_num"][idx]

    pool.close()
    pool.join()

    if total_statistic["instances"] > 0:
        logger.info("Total examples: %d" %
                    total_statistic["instances"])
        logger.info("Average sentence number per dial: %f" %
                    (total_statistic["total_sents"] / total_statistic["instances"]))
        logger.info("Processed average sentence number per dial: %f" %
                    (total_statistic["processed_sents"] / total_statistic["instances"]))
        logger.info("Total sentences: %d" %
                    total_statistic["total_sents"])
        logger.info("Processed sentences: %d" %
                    total_statistic["processed_sents"])
        logger.info("Exceeded max sentence number dials: %d" %
                    total_statistic["exceed_sents_num"])
        logger.info("Max dial sentences: %d" %
                    total_statistic["max_sents"])
        for idx, num in enumerate(total_statistic["sents_num"]):
            logger.info("dial sentences %d ~ %d: %d, %.2f%%" %
                        (idx * 20, (idx+1) * 20, num, (num / total_statistic["instances"])))
        logger.info("Exceed length sentences number: %d" %
                    total_statistic["exceed_length_num"])
        logger.info("Average src sentence length: %f" %
                    (total_statistic["total_src_length"] / total_statistic["total_sents"]))
        for idx, num in enumerate(total_statistic["src_sent_length_num"]):
            logger.info("Sent length %d ~ %d: %d, %.2f%%" %
                        (idx * 10, (idx+1) * 10, num, (num / total_statistic["total_sents"])))
        logger.info("Average src token length: %f" %
                    (total_statistic["total_src_length"] / total_statistic["instances"]))
        for idx, num in enumerate(total_statistic["src_token_length_num"]):
            logger.info("token num %d ~ %d: %d, %.2f%%" %
                        (idx * 300, (idx+1) * 300, num, (num / total_statistic["instances"])))
        logger.info("Average tgt length: %f" %
                    (total_statistic["total_tgt_length"] / total_statistic["instances"]))
        logger.info("Tgt concept ratio: %f" %
                    (total_statistic["total_tgt_concept"] / total_statistic["instances"]))


def _format_json_to_data(params):
    _, json_file, args, save_file = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)
    graph_processor = GraphData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))

    datasets = []
    exceed_length_num = 0
    exceed_sents_num = 0
    total_src_length = 0.
    total_tgt_length = 0.
    src_length_sent_num = [0] * 11
    src_length_token_num = [0] * 11
    max_sents = 0
    sents_num = [0] * 11
    dial_sents = 0.
    processed_sents = 0.
    tgt_concept = 0.

    for dial in jobs:
        dial_b_data = []
        dial_token_num = 0
        for index, sent in enumerate(dial['context']):
            sent = clean(sent['content'])
            b_data = bert.preprocess_src(sent)
            if (b_data is None):
                continue
            src_subtoken_idxs, segments_ids, original_txt, \
                src_subtokens, exceed_length = b_data

            b_data_dict = {"index": index,
                           "src_id": src_subtoken_idxs,
                           "segs": segments_ids,
                           "original_txt": original_txt,
                           "src_tokens": src_subtokens}

            src_length_sent_num[min(len(src_subtoken_idxs) // 10, 10)] += 1
            dial_token_num += len(src_subtoken_idxs)
            total_src_length += len(src_subtoken_idxs)
            dial_b_data.append(b_data_dict)
            if exceed_length:
                exceed_length_num += 1
            if len(dial_b_data) >= args.max_sents:
                exceed_sents_num += 1
                if args.truncated:
                    break
        dial_example = {"session": dial_b_data}
        dial_integrated = bert.integrate_dial(dial_b_data)
        dial_example["context"] = dial_integrated

        # target data process
        cleaned_sents = clean(dial['target'])
        target_b_data = bert.preprocess_target(cleaned_sents)
        subtoken_idxs, original_txt, content_subtokens = target_b_data
        total_tgt_length += len(subtoken_idxs)
        b_data_dict = {"id": subtoken_idxs,
                       "original_txt": original_txt,
                       "content_tokens": content_subtokens}
        dial_example["tgt"] = b_data_dict

        # graph data process

        source_entities = dial['source_entity']
        raw_se, idx_se = graph_processor.preprocess_src_entities(source_entities)
        dial_example["source_entity"] = {
            "id": idx_se,
            "original_entities": raw_se
        }

        target_entities = dial['target_entity']
        raw_te, idx_te, e_num = graph_processor.preprocess_tgt_entities(target_entities)
        dial_example["target_entity"] = {
            "id": idx_te,
            "original_entities": raw_te
        }
        tgt_concept += e_num

        triples = dial['triples']
        raw_triple, idx_tri = graph_processor.preprocess_triples(triples, target_entities, bert.tokenizer)
        dial_example["triples"] = {
            "id": idx_tri,
            "original_triples": raw_triple
        }

        context_list = dial['context_keywords_list']
        raw_c_list, idx_c_list = graph_processor.preprocess_context(context_list)
        dial_example["context_entity"] = {
            "id": idx_c_list[-len(dial_b_data):],
            "original_entities": raw_c_list[-len(dial_b_data):]
        }

        if len(dial_b_data) >= args.min_sents:
            datasets.append(dial_example)
            sents_num[min(len(dial_b_data) // 20, 10)] += 1
            src_length_token_num[min(dial_token_num // 300, 10)] += 1
            max_sents = max(max_sents, len(dial_b_data))
            dial_sents += len(dial['context'])
            processed_sents += len(dial_b_data)

    statistic = {
        "instances": len(datasets),
        "total_sents": dial_sents,
        "processed_sents": processed_sents,
        "max_sents": max_sents,
        "sents_num": sents_num,
        "exceed_length_num": exceed_length_num,
        "exceed_sents_num": exceed_sents_num,
        "total_src_length": total_src_length,
        "src_sent_length_num": src_length_sent_num,
        "src_token_length_num": src_length_token_num,
        "total_tgt_length": total_tgt_length,
        "total_tgt_concept": tgt_concept
    }
    logger.info('Processed instances %d' % len(datasets))
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()
    return statistic
