# here put the import lib
from typing import Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


class LlamaRSEmb(LlamaPreTrainedModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        
        super().__init__(config, *inputs, **kwargs)
        self.model = LlamaModel(config)
        self.pool_type = kwargs.pop("pool_type")
        self.tau = kwargs.pop("tau")
        self.tau = nn.Parameter(torch.FloatTensor([3]))

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.model.embed_tokens


    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]  # get the output from LLM (bs, seq_len, hidden_size)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(input_ids.device)
            else:
                sequence_lengths = -1

        # transformer_outputs.hidden_states = hidden_states[torch.arange(batch_size, device=logits.device), sequence_lengths]   # output the hidden_states for feature-based KD
        transformer_outputs.hidden_states = self._get_pool_emb(hidden_states, sequence_lengths, attention_mask)

        # get pooled logits
        pooled_logits = transformer_outputs.hidden_states

        # split the pair of outputs
        batch_size = input_ids.size(0) // 2
        chosen_logits, rejected_logits = pooled_logits.split(batch_size, dim=0)

        loss = None
        if labels is not None:

            loss_fct = Contrastive_Loss(self.tau)
            loss = loss_fct(chosen_logits, rejected_logits)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
    

    def _get_pool_emb(self, hidden_states, sequence_lengths, pooled_mask):
        """get the logits according to pool type"""
        if self.pool_type == "last":    # take out the last token as LLM embedding
            pooled_emb = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), 
                                       sequence_lengths]
        # average pooling all tokens as LLM embedding or average pooling attribute tokens as LLM embedding
        elif self.pool_type == "avg":   
            pooled_emb = torch.sum(hidden_states * pooled_mask.unsqueeze(-1), dim=1) / torch.sum(pooled_mask, dim=1).unsqueeze(-1) #sequence_lengths.unsqueeze(-1)

        return pooled_emb



class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau


    def forward(self, X, Y):
        
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (Y_loss + X_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    

    def cross_entropy(self, preds, targets, reduction='none'):

        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        






