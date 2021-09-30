#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import glob
import os
import time
import torch

from transformers import BertTokenizer
from models import data_loader
from models.data_loader import load_dataset
from models.model import Model as graph_model
from models.seq2seq import Model as seq2seq_model
from models.model_predictor import build_predictor as graph_build_predictor
from models.seq2seq_predictor import build_predictor as seq2seq_build_predictor
from others.logging import logger

model_flags = ['encoder', 'decoder', 'enc_heads', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_heads', 'dec_layers', 'dec_hidden_size', 'dec_ff_size']


def validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        best_dev_steps = -1
        best_dev_results = (0, 0, 0)
        patient = 100
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            if (args.test_start_from != -1 and step < args.test_start_from):
                xent_lst.append((1e6, cp))
                continue
            logger.info("Step %d: processing %s" % (i, cp))
            result_dev = _validate(args, device_id, cp, step)
            if result_dev[0] > best_dev_results[0]:
                best_dev_results = result_dev
                best_dev_steps = step
                patient = 100
            else:
                patient -= 1
            logger.info("Current step: %d" % step)
            logger.info("Dev results: bleu-dist1-dist2: %.4f, %.4f, %.4f" %
                        (result_dev[0], result_dev[1], result_dev[2]))
            logger.info("Best step: %d" % best_dev_steps)
            logger.info("Best dev results: bleu-dist1-dist2: %.4f, %.4f, %.4f\n\n" %
                        (best_dev_results[0], best_dev_results[1], best_dev_results[2]))

            if patient == 0:
                break

    else:
        best_dev_results = (0, 0, 0)
        best_dev_steps = -1
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    result_dev = _validate(args, device_id, cp, step)
                    if result_dev[0] > best_dev_results[0]:
                        best_dev_results = result_dev
                        best_dev_steps = step

                    logger.info("Current step: %d" % step)
                    logger.info("Dev results: bleu-dist1-dist2: %f, %f, %f" %
                                (result_dev[0], result_dev[1], result_dev[2]))
                    logger.info("Best step: %d" % best_dev_steps)
                    logger.info("Best dev results: bleu-dist1-dist2: %f, %f, %f\n\n" %
                                (best_dev_results[0], best_dev_results[1], best_dev_results[2]))

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def _validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    if args.use_graph:
        model = graph_model(args, device, tokenizer.vocab, checkpoint)
    else:
        model = seq2seq_model(args, device, tokenizer.vocab, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'dev', shuffle=False),
                                        args.test_batch_size, args.test_batch_ex_size, device,
                                        shuffle=False, is_test=True)

    if args.use_graph:
        predictor = graph_build_predictor(args, tokenizer, model, logger)
    else:
        predictor = seq2seq_build_predictor(args, tokenizer, model, logger)
    result = predictor.validate(valid_iter, step)
    return result
