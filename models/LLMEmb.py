# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
from models.Adapter import SASRecPLUS, Bert4RecPLUS, GRU4RecPLUS
from models.utils import Contrastive_Loss2



class LLMEmbSASRec(SASRecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)
        
        srs_item_emb = pickle.load(open("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), "rb"))
        srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()

    
    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss
    


class LLMEmbBert4Rec(Bert4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        srs_item_emb = pickle.load(open("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), "rb"))
        srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()

    
    def forward(self, seq, pos, neg, positions, **kwargs):

        loss =  super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss
    


class LLMEmbGRU4Rec(GRU4RecPLUS):

    def __init__(self, user_num, item_num, device, args):

        super().__init__(user_num, item_num, device, args)

        srs_item_emb = pickle.load(open("./data/{}/handled/itm_emb_sasrec.pkl".format(args.dataset), "rb"))
        srs_item_emb = np.insert(srs_item_emb, 0, values=np.zeros((1, srs_item_emb.shape[1])), axis=0)
        self.srs_emb = nn.Embedding.from_pretrained(torch.Tensor(srs_item_emb))
        self.srs_emb.weight.requires_grad = False

        self.align_loss_func = Contrastive_Loss2(args.tau)
        self.alpha = args.alpha

        self.filter_init_modules.append("srs_emb")
        self._init_weights()


    def forward(self, seq, pos, neg, positions, **kwargs):

        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # get align loss
        indices = (pos != 0)    # do not calculate the padding units
        srs_embs = self.srs_emb(pos[indices])
        llm_embs = self._get_embedding(pos[indices])
        align_loss = self.align_loss_func(srs_embs, llm_embs)

        loss += self.alpha * align_loss

        return loss




