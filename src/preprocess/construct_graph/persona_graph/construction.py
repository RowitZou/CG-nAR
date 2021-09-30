from collections import Counter
import numpy as np
import sys

max_vertex = 10000
min_outdegrees = 0
min_pairs_num = 2
Threshold = 0

word_dict = {}
co_occurrence = {}

occurrence_p = {}
co_occurrence_p = {}

co_occurrence_num = 0.0

graph = {}

use_softmax = False

contexts = []
keywords = []
vocab = []

def get_median(s_list):
    length = len(s_list)
    s_list.sort()
    if length % 2 == 1:
        return s_list[length//2]
    else:
        return (s_list[length//2]+s_list[length//2-1])/2

def get_one_third_digit(s_list):
    length = len(s_list)
    s_list.sort()
    return s_list[length//3]

def get_co_data(type):
    context_file = './tx_data/' + type + '/raw_context.txt'
    keywords_file = './tx_data/' + type + '/keywords.txt'
    keywords_vocab_file = './tx_data/' + type + '/keywords_vocab.txt'
    with open(context_file, 'r') as f:
        contexts.extend(f.readlines())
    with open(keywords_file, 'r') as f:
        keywords.extend(f.readlines())
    with open(keywords_vocab_file, 'r') as f:
        vocab = f.readlines()
    for word in vocab:
        word = word.strip()
        if len(word) > 1 and word not in co_occurrence.keys():
            word_dict[word] = 0.0
            co_occurrence[word] = {}

def construct(write_filename):
    prev_list = []
    PMI_list = []
    length1 = len(contexts)
    # length2 = len(keywords)
    # vocab_size = len(word_dict)

    for index in range(0, length1 - 1):
        context = contexts[index + 1]
        keyword = keywords[index]
        context_list = context.strip().split(' ')
        keyword_list = keyword.strip().split(' ')
        for word in keyword_list:
            assert word in word_dict, '{} is not found'.format(word)
            word_dict[word] += 1.0
        for prev_word in prev_list:
            for word in keyword_list:
                if word not in co_occurrence[prev_word]:
                    co_occurrence[prev_word][word] = 1.0
                else:
                    co_occurrence[prev_word][word] += 1.0
        if keyword not in context:
            prev_list = []
        else:
            prev_list = keyword_list

    occurrence_sum = sum(word_dict.values())

    for key, value in word_dict.items():
        occurrence_p[key] = value / occurrence_sum

    # min_PMI = 0.5609278391326677
    # max_PMI = 9.287417147155608
    # Threshold = (min_PMI + max_PMI) / 2

    exp_sum = 0.0

    for key, value in co_occurrence.items():
        exp_sum += np.exp(len(value) / 10)

    # print('sum = {}'.format(exp_sum))

    for key, value in co_occurrence.items():
        co_occurrence_p[key] = {}
        co_occurrence_sum = sum(value.values())
        size = len(value)
        if size < min_outdegrees:
            continue
        ratio = np.exp(size / 10) / exp_sum
        if use_softmax:
            for next_key, next_value in co_occurrence[key].items():
                co_occurrence_p[key][next_key] = next_value / co_occurrence_sum
                co_occurrence_p[key][next_key] = np.log(ratio * co_occurrence_p[key][next_key] / occurrence_p[next_key])
        else:
            for next_key, next_value in co_occurrence[key].items():
                if next_value < min_pairs_num:
                    continue
                co_occurrence_p[key][next_key] = next_value / co_occurrence_sum
                co_occurrence_p[key][next_key] = np.log(co_occurrence_p[key][next_key] / occurrence_p[next_key])
        if len(co_occurrence_p[key]) == 0:
            continue
        else:
            graph[key] = []
        sort_items = list(sorted(co_occurrence_p[key].items(), key=lambda x: x[1], reverse=True))
        max_num = max_vertex if size > max_vertex else size
        for sort_key, sort_value in sort_items[:max_num]:
            PMI_list.append(sort_value)
            if sort_value > Threshold:
                graph[key].append(sort_key)
            # Threshold += sort_value
            # all_sum += 1.0

    # print('PMI-median: {}'.format(get_median(PMI_list)))
    # print('PMI-one-third: {}'.format(get_one_third_digit(PMI_list)))
    # Threshold = Threshold / all_sum
    # print('Threshold= {}'.format(Threshold))

    write_in_graph = {}

    for key, value in graph.items():
        write_in = True
        for word in value:
            if word in graph.keys() and len(graph[word]) != 0:
                if write_in:
                    write_in_graph[key] = []
                    write_in = False
                write_in_graph[key].append(word)

    with open(write_filename, 'w') as target_file:
        for key, value in write_in_graph.items():
            write_in = True
            for word in value:
                if word in write_in_graph.keys() and len(write_in_graph[word]) != 0:
                    if write_in:
                        target_file.write(key + ':')
                        target_file.write('\n')
                        write_in = False
                    target_file.write(word)
                    target_file.write('\n')
            target_file.write('\n')

if __name__ == '__main__':
    types = ['train', 'test', 'valid']
    for type in types:
        get_co_data(type)
    construct(sys.argv[1])