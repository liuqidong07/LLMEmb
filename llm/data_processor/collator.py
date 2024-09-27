# here put the import lib
from dataclasses import dataclass
from typing import Any, List, Dict, Sequence, Tuple
import torch
import transformers
from transformers import DataCollatorForSeq2Seq

IGNORE_INDEX = -100


@dataclass
class LongestSequenceMaskCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=False
            )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


@dataclass
class PairwiseDataCollatorWithPadding(LongestSequenceMaskCollator):

    tokenizer: transformers.PreTrainedTokenizer

    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_ids".format(key)],
                    "attention_mask": feature["{}_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }

                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)
    

