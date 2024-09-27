# here put the import lib
import os
import argparse
import torch

from generators.generator import Generator, Seq2SeqGenerator
from generators.bert_generator import BertGenerator
from trainers.sequence_trainer import SeqTrainer
from utils.utils import set_seed
from utils.logger import Logger
from utils.argument import *


parser = argparse.ArgumentParser()
parser = get_main_arguments(parser)
parser = get_model_arguments(parser)
parser = get_train_arguments(parser)

torch.autograd.set_detect_anomaly(True)

args = parser.parse_args()
set_seed(args.seed) # fix the random seed
args.output_dir = os.path.join(args.output_dir, args.dataset)
args.pretrain_dir = os.path.join(args.output_dir, args.pretrain_dir)
args.output_dir = os.path.join(args.output_dir, args.model_name)
args.output_dir = os.path.join(args.output_dir, args.check_path)    # if check_path is none, then without check_path
args.llm_emb_path = os.path.join("data/"+args.dataset+"/handled/", "{}.pkl".format(args.llm_emb_file))


def main():

    log_manager = Logger(args)  # initialize the log manager
    logger, writer = log_manager.get_logger()    # get the logger
    args.now_str = log_manager.get_now_str()

    device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")


    os.makedirs(args.output_dir, exist_ok=True)

    # generator is used to manage dataset
    if args.model_name in ['gru4rec', "llmemb_gru4rec"]:
        generator = Generator(args, logger, device)
    elif args.model_name in ['bert4rec', "llmemb_bert4rec"]:
        generator = BertGenerator(args, logger, device)
    elif args.model_name in ['sasrec_seq', "llmemb_sasrec"]:
        generator = Seq2SeqGenerator(args, logger, device)
    else:
        raise ValueError

    trainer = SeqTrainer(args, logger, writer, device, generator)

    if args.do_test:
        trainer.test()
    elif args.do_emb:
        trainer.save_item_emb()
        trainer.save_user_emb()
    elif args.do_group:
        trainer.test_group()
    else:
        trainer.train()

    log_manager.end_log()   # delete the logger threads



if __name__ == "__main__":

    main()



