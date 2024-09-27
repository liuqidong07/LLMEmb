# here put the import lib
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec


class SASRecPLUS(SASRec_seq):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.insert(llm_item_emb, -1, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        return item_seq_emb
    

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    


class Bert4RecPLUS(Bert4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.concatenate([llm_item_emb, np.zeros((1, llm_item_emb.shape[1]))], axis=0)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))
        # self.pad_embedding = nn.Parameter(torch.zeros(self.hidden_size).normal_(0, 0.01))

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        item_seq_emb[log_seqs==self.mask_token] = self.mask_embedding
        # item_seq_emb[log_seqs==0] = self.pad_embedding

        return item_seq_emb
    

    def log2feats(self, log_seqs, positions):
        '''Get the representation of given sequence'''
        seqs = self._get_embedding(log_seqs)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    


class GRU4RecPLUS(GRU4Rec):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        self.hidden_size = args.hidden_size
        llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
        llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        llm_item_emb = np.insert(llm_item_emb, -1, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        self.filter_init_modules = ["item_emb"]
        self._init_weights()

    
    def _get_embedding(self, log_seqs):

        item_seq_emb = self.item_emb(log_seqs)
        item_seq_emb = self.adapter(item_seq_emb)

        return item_seq_emb
    

    def log2feats(self, log_seqs):
        '''Get the representation of given sequence'''
        seqs = self.item_emb(log_seqs)
        seqs = self.adapter(seqs)

        log_feats = self.backbone(seqs, log_seqs)

        return log_feats
    




