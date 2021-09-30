import json
import pickle

from os.path import join as pjoin

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def word_lemmatizer(sentence):
    # get pos tags
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

    return lemmas_sent


def get_location_in_list(x, target):
    step = -1
    items = list()
    for i in range(x.count(target)):
        y = x[step + 1:].index(target)
        step = step + y + 1
        items.append(step)
    return items


def _format_raw_to_json(args, line_src, line_tgt,
                        line_context_key,
                        entity_vocab, idx2entity,
                        adj_matrix):
    ex = {}
    ex['context'] = list(map(lambda x: {'content': x.split()}, line_src.split('|||')))
    ex['target'] = line_tgt.split('|||')[0].split()
    ex['source_entity'] = list()
    ex['target_entity'] = list()
    ex['triples'] = {}
    ex['context_keywords_list'] = [line.split(' ') if len(line) > 0 else ['[UNK]']
                                   for line in line_context_key.split(',')]

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
    for word in target:
        if word in entity_vocab.keys():
            ex['target_entity'].append(word)
        if len('target_entity') == 0:
            ex['target_entity'].append('[UNK]')

    for entity in ex['source_entity']:
        idx = entity_vocab[entity]
        # entity == [UNK]
        if idx == 1:
            tail_list = ['[UNK]']
            ex['triples'][entity] = tail_list
            continue
        tail_list = get_location_in_list(adj_matrix[idx-2], 1)
        # considering [UNK][pad]
        tail_list = [idx2entity[tail+2] for tail in tail_list]
        ex['triples'][entity] = tail_list
    return ex


def format_raw_to_jsons(args):
    corpus_types = ['train', 'valid', 'test']
    for corpus_type in corpus_types:
        dataset = []
        p_ct = 0

        src_f = open(pjoin(args.raw_path, corpus_type + '/source.txt'), encoding='utf-8')
        tgt_f = open(pjoin(args.raw_path, corpus_type + '/target.txt'), encoding='utf-8')
        context_key_f = open(pjoin(args.raw_path, corpus_type + '/context_list.txt'), encoding='utf-8')

        entity_vocab = {}
        idx2entity = {}

        with open(args.vertex_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(' ')
                entity_vocab[line[0]] = int(line[1])
                idx2entity[int(line[1])] = line[0]

        with open(args.adj_file, 'rb') as f:
            adj_matrix = pickle.load(f)

        for line_src in src_f:
            if args.lower:
                line_src = line_src.strip().lower()
                line_tgt = tgt_f.readline().strip().lower()
                line_context_key = context_key_f.readline().strip().lower()
            else:
                line_src = line_src.strip()
                line_tgt = tgt_f.readline().strip()
                line_context_key = context_key_f.readline().strip()

            dataset.append(_format_raw_to_json(args, line_src, line_tgt,
                                               line_context_key,
                                               entity_vocab, idx2entity,
                                               adj_matrix))

            if len(dataset) >= args.shard_size:
                if corpus_type == 'valid':
                    this_type = 'dev'
                else:
                    this_type = corpus_type
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, this_type, p_ct)
                print("Saving to " + pt_file)
                with open(pt_file, 'w') as save:
                    json.dump(dataset, save)
                    p_ct += 1
                    dataset = []

        if len(dataset) > 0:
            if corpus_type == 'valid':
                this_type = 'dev'
            else:
                this_type = corpus_type
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, this_type, p_ct)
            print("Saving to " + pt_file)
            with open(pt_file, 'w') as save:
                json.dump(dataset, save)
