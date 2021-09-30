# encoding=utf-8

import argparse
import random
from others.logging import init_logger
from preprocess import raw_to_json_persona as json_builder_persona
from preprocess import raw_to_json_weibo as json_builder_weibo
from preprocess import json_to_data_persona as data_builder_persona
from preprocess import json_to_data_weibo as data_builder_weibo


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default='persona', type=str, choices=['persona', 'weibo'])
    parser.add_argument("-type", default='train', type=str, choices=['train', 'dev', 'test'])
    parser.add_argument("-mode", default='json_to_data', type=str, choices=['raw_to_json', 'json_to_data'])
    parser.add_argument("-raw_path", default='json_data/persona/', type=str)
    parser.add_argument("-save_path", default='torch_data/persona', type=str)
    parser.add_argument("-n_cpus", default=4, type=int)
    parser.add_argument("-random_seed", default=666, type=int)
    parser.add_argument('-log_file', default='logs/raw_to_json.log')
    parser.add_argument('-seed', default=666, type=int)

    # raw_to_json args
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-adj_file", default='graph_data/persona/adj_matrix.txt', type=str)
    parser.add_argument("-vertex_file", default='graph_data/persona/vertex.txt', type=str)

    # json_to_data args
    parser.add_argument("-tokenizer", default='bert-base-uncased')
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('-min_sents', default=1, type=int)
    parser.add_argument('-max_sents', default=40, type=int)
    parser.add_argument('-max_tgt_len', default=100, type=int)
    parser.add_argument('-tail_sample_num', default=50, type=int)
    parser.add_argument("-truncated", nargs='?', const=True, default=False)

    args = parser.parse_args()
    init_logger(args.log_file)
    random.seed(args.seed)

    if(args.mode == 'raw_to_json'):
        if args.dataset == 'persona':
            json_builder_persona.format_raw_to_jsons(args)
        else:
            json_builder_weibo.format_raw_to_jsons(args)
    else:
        if args.dataset == 'persona':
            data_builder_persona.format_json_to_data(args, args.type)
        else:
            data_builder_weibo.format_json_to_data(args, args.type)
