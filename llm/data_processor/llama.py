# here put the import lib
import numpy as np
from tqdm import tqdm
import copy
import random


class llama_train_mask(object):
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.tokenizer = tokenizer


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "chosen_ids": [],
            "rejected_ids": [],
            "chosen_labels": [],
            "rejected_labels": [],
            "chosen_mask": [],
            "rejected_mask": [],
        }


        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]
                chosen, rejected = dropout_feature(query, self.data_args.dropout_ratio)

                chosen_a_ids = self.tokenizer.encode(text=chosen, add_special_tokens=False)
                rejected_a_ids = self.tokenizer.encode(text=rejected, add_special_tokens=False)

                if len(chosen_a_ids) > self.data_args.max_source_length - 1:
                    chosen_a_ids = chosen_a_ids[: self.data_args.max_source_length - 1]
                if len(rejected_a_ids) > self.data_args.max_source_length - 1:
                    rejected_a_ids = rejected_a_ids[: self.data_args.max_source_length - 1]

                chosen_input_ids = self.tokenizer.build_inputs_with_special_tokens(chosen_a_ids)
                rejected_input_ids = self.tokenizer.build_inputs_with_special_tokens(rejected_a_ids)

                chosen_context_length = len(chosen_a_ids)
                chosen_input_ids = chosen_a_ids + [self.tokenizer.eos_token_id]
                rejected_context_length = len(rejected_a_ids)
                rejected_input_ids = rejected_a_ids + [self.tokenizer.eos_token_id]
                chosen_labels = [self.tokenizer.pad_token_id] * chosen_context_length + [self.tokenizer.eos_token_id]
                rejected_labels = [self.tokenizer.pad_token_id] * rejected_context_length + [self.tokenizer.eos_token_id]

                if self.data_args.ignore_pad_token_for_loss:
                    chosen_labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in chosen_labels]
                    rejected_labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in rejected_labels]
                
                chosen_mask = [True] * (chosen_context_length + 1)
                rejected_mask = [True] * (rejected_context_length + 1)

                model_inputs["chosen_ids"].append(chosen_input_ids)
                model_inputs["rejected_ids"].append(rejected_input_ids)
                model_inputs["chosen_labels"].append(chosen_labels)
                model_inputs["rejected_labels"].append(rejected_labels)
                model_inputs["chosen_mask"].append(chosen_mask)
                model_inputs["rejected_mask"].append(rejected_mask)

        return model_inputs
    


def dropout_feature(item_str, ratio):

    instruction = item_str.split("\n")[0]
    feat_list = item_str.split("\n")[1].split(";")

    # get two copies, shuffle the order, remove the last ratio_N
    feat_list_1 = copy.deepcopy(feat_list)
    feat_list_2 = copy.deepcopy(feat_list)
    random.shuffle(feat_list_1)
    random.shuffle(feat_list_2)

    dropout_N = int(len(feat_list) * ratio)
    for _ in range(dropout_N):  # dropout N times
        feat_list_1.pop()
        feat_list_2.pop()

    # assemble the feat list to item string
    item_str_1 = instruction + "\n"
    item_str_2 = instruction + "\n"
    if len(feat_list) > 1:
        for i in range(len(feat_list_1)):
            item_str_1 += (feat_list_1[i] + ";")
            item_str_2 += (feat_list_2[i] + ";")

    return item_str_1[:-1], item_str_2[:-1] # remove the last ";"



def get_mask(input_ids, start_id, end_id):

    mask = []
    attri_flag = False   # mark the words as attribute
    for i, token_id in enumerate(input_ids):
        if token_id == end_id or i == (len(input_ids)-1):  # judge whether ends at first
            attri_flag = False
        if attri_flag:
            mask.append(True)
        else:
            mask.append(False)
        if token_id == start_id:    # judge whether ends at last
            attri_flag = True

    return mask


    
class llama_eval_mask(object):
    
    def __init__(self, data_args, model_args, tokenizer) -> None:
    
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer


    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }


        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]
                a_ids = self.tokenizer.encode(text=query, add_special_tokens=False)

                if len(a_ids) > self.data_args.max_source_length - 1:
                    a_ids = a_ids[: self.data_args.max_source_length - 1]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids)

                context_length = len(a_ids)
                input_ids = a_ids + [self.tokenizer.eos_token_id]
                labels = [self.tokenizer.pad_token_id] * context_length + [self.tokenizer.eos_token_id]
                
                pad_len = max_seq_length - len(input_ids)

                if self.data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]
                
                output_mask = [True] * (context_length + 1)

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(input_ids)
                model_inputs["attention_mask"].append(output_mask)

        return model_inputs
    


