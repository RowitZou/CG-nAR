#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import os
import random
import signal
import torch
import distributed
import numpy as np

from transformers import BertTokenizer
from models import data_loader
from models.data_loader import load_dataset
from models.optimizers import build_optim, build_optim_bert, build_optim_other
from models.model import Model as graph_model
from models.model_trainer import build_trainer as graph_build_trainer
from models.seq2seq import Model as seq2seq_model
from models.seq2seq_trainer import build_trainer as seq2seq_build_trainer
from others.logging import logger, init_logger

model_flags = ['encoder', 'decoder', 'enc_heads', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_heads', 'dec_layers', 'dec_hidden_size', 'dec_ff_size']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_multi(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """

    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks, args.port)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def train(args, device_id):
    if (args.world_size > 1):
        train_multi(args)
    else:
        train_single(args, device_id)


def train_single(args, device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True),
                                      args.batch_size, args.batch_ex_size, device,
                                      shuffle=True, is_test=False)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    if args.use_graph:
        model = graph_model(args, device, tokenizer.vocab, checkpoint)
    else:
        model = seq2seq_model(args, device, tokenizer.vocab, checkpoint)

    if args.train_from_ignore_optim:
        checkpoint = None
    if args.sep_optim and args.encoder == 'bert':
        optim_bert = build_optim_bert(args, model, checkpoint)
        optim_other = build_optim_other(args, model, checkpoint)
        optim = [optim_bert, optim_other]
    else:
        optim = [build_optim(args, model, checkpoint, args.warmup)]

    logger.info(model)

    if args.use_graph:
        trainer = graph_build_trainer(args, device_id, model, optim, tokenizer)
    else:
        trainer = seq2seq_build_trainer(args, device_id, model, optim, tokenizer)

    trainer.train(train_iter_fct, args.train_steps)
