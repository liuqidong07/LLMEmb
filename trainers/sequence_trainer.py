# here put the import lib
import os
import time
import pickle
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from models.SASRec import SASRec_seq
from models.Bert4Rec import Bert4Rec
from models.GRU4Rec import GRU4Rec
from models.LLMEmb import LLMEmbSASRec, LLMEmbBert4Rec, LLMEmbGRU4Rec
from utils.utils import metric_report, metric_len_report, record_csv, metric_pop_report
from utils.utils import metric_len_5group, metric_pop_5group


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)


    def _create_model(self):
        '''create your model'''
        if self.args.model_name == 'sasrec_seq':
            self.model = SASRec_seq(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == 'bert4rec':
            self.model = Bert4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == 'gru4rec':
            self.model = GRU4Rec(self.user_num, self.item_num, self.device, self.args)
        
        ## Our Method -- LLMEmb
        elif self.args.model_name == "llmemb_sasrec":
            self.model = LLMEmbSASRec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "llmemb_bert4rec":
            self.model = LLMEmbBert4Rec(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "llmemb_gru4rec":
            self.model = LLMEmbGRU4Rec(self.user_num, self.item_num, self.device, self.args)
        
        else:
            raise ValueError
        
        self.model.to(self.device)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            inputs = self._prepare_train_inputs(batch)
            loss = self.model(**inputs)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            self.model.load_state_dict(model_state_dict['state_dict'])
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('\t %s: %.5f' % (k, v))

        if test:
            self.logger.info("User Group Performance:")
            for k, v in res_len_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
            self.logger.info("Item Group Performance:")
            for k, v in res_pop_dict.items():
                if not test:
                    self.writer.add_scalar('Test/{}'.format(k), v, epoch)
                self.logger.info('\t %s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict, **res_pop_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
    


    def save_user_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        test_loader = self.test_loader

        self.model.eval()
        user_emb = torch.empty(0).to(self.device)
        desc = 'Running'

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            
            with torch.no_grad():

                per_user_emb = self.model.get_user_emb(**inputs)
                user_emb = torch.cat([user_emb, per_user_emb], dim=0)
        
        user_emb = user_emb.detach().cpu().numpy()
        pickle.dump(user_emb, open("./data/{}/handled/usr_emb_{}.pkl".format(self.args.dataset, self.args.model_name), "wb"))

    

    def save_item_emb(self):

        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        try:
            self.model.load_state_dict(model_state_dict['state_dict'])
        except:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)

        all_index = torch.arange(start=1, end=self.item_num+1).to(self.device)
        item_emb = self.model._get_embedding(all_index)
        item_emb = item_emb.detach().cpu().numpy()
        pickle.dump(item_emb, open("./data/{}/handled/itm_emb_{}.pkl".format(self.args.dataset, self.args.model_name), "wb"))


    
    def test_group(self):

        print('')
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running Group test **********")
        desc = 'Testing'
        model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
        self.model.load_state_dict(model_state_dict['state_dict'])
        self.model.to(self.device)
        test_loader = self.test_loader
        
        self.model.eval()
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        target_items = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            inputs = self._prepare_eval_inputs(batch)
            seq_len = torch.cat([seq_len, torch.sum(inputs["seq"]>0, dim=1)])
            target_items = torch.cat([target_items, inputs["pos"]])
            
            with torch.no_grad():

                inputs["item_indices"] = torch.cat([inputs["pos"].unsqueeze(1), inputs["neg"]], dim=1)
                pred_logits = -self.model.predict(**inputs)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        # res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len, args=self.args)
        # res_pop_dict = metric_pop_report(pred_rank.detach().cpu().numpy(), self.item_pop, target_items.detach().cpu().numpy(), args=self.args)
        hr_len, ndcg_len, count_len = metric_len_5group(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), [5, 10, 15, 20])
        hr_pop, ndcg_pop, count_pop = metric_pop_5group(pred_rank.detach().cpu().numpy(), self.item_pop,  target_items.detach().cpu().numpy(), [5, 10, 20, 40])

        self.logger.info("Overall Performance:")
        for k, v in res_dict.items():
            self.logger.info('\t %s: %.5f' % (k, v))

        self.logger.info("User Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_len, ndcg_len)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        self.logger.info("Item Group Performance:")
        for i, (hr, ndcg) in enumerate(zip(hr_pop, ndcg_pop)):
            self.logger.info('The %d Group: HR %.4f, NDCG %.4f' % (i, hr, ndcg))
        
        
        return res_dict
    


