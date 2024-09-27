# here put the import lib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import pickle
import torch
from datasets import load_dataset

from llm.peft import (
    LoraConfig,
    PeftModel,
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llm.llama import LlamaRSEmb
from llm.trainer_seq2seq import MedRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.llama import llama_train_mask, llama_eval_mask
from llm.data_processor.collator import LongestSequenceMaskCollator, PairwiseDataCollatorWithPadding


# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    def on_save(    
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
        

def train():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device_map = "auto"

    ## Load Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"  # define the padding direction

    ## Load Model ##
    if model_args.model_choice == "rsemb":

        model = LlamaRSEmb.from_pretrained(
            model_args.model_name_or_path,
            pool_type=model_args.pool_type,
            tau=model_args.tau,
        ).half().cuda()

        if model_args.peft_path is not None:    # for test model
        # Resume_training
            if training_args.resume_from_checkpoint is not None:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
            else:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
        else:   # for train model
            # Load Lora Config
            peft_config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.trainable.split(","),
                lora_dropout=model_args.lora_dropout,
                task_type="SEQ_CLS",
            )
            model = PeftModelForCLS(model, peft_config)  # LoRA wrapped llama

    else:

        raise ValueError("No such LLM model")

    

    if training_args.do_train:
        for name, param in model.named_parameters():    # activate the head attention parameters
            if "head_attn" in name:
                param.requires_grad = True
            if "tau" in name:
                try:
                    param.requires_grad = True
                except:
                    pass
            if "item_wte" in name:
                param.requires_grad = True
            if "projector" in name:
                param.requires_grad = True
            if "cls_head" in name:
                param.requires_grad = True

    # model.print_trainable_parameters()

    ## Load Dataset ##
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print("raw_datasets: ", raw_datasets)

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
    
    if training_args.do_train:
        preprocess_func = llama_train_mask(data_args, model_args, tokenizer)
        data_collator = PairwiseDataCollatorWithPadding(tokenizer)

    else:
        preprocess_func = llama_eval_mask(data_args, model_args, tokenizer)
        data_collator = LongestSequenceMaskCollator(tokenizer)

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on prediction dataset",
        )
    target_dataset.set_format("torch")

    training_args.remove_unused_columns = False  # important for pairwise dataset

    ## Set Trainer ##
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None), # substitute the original model saver
    )

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()

    if model_args.model_choice == "said":

        item_emb = model.item_wte.weight    # get the embedding
        item_emb = item_emb.detach().cpu().numpy().astype(float)  # convert to numpy
        
        item_emb = item_emb[1:, :]  # remove the padding item
        pickle.dump(item_emb, open("data/yelp/handled/{}.pkl".format(model_args.output_file), "wb"))

    ## Evaluation ##
    results = {}

    if training_args.do_predict:

        if model_args.model_choice == "said":

            item_emb = model.item_wte.weight    # get the embedding
            item_emb = item_emb.detach().cpu().numpy().astype(float)  # convert to numpy
            
            item_emb = item_emb[1:, :]  # remove the padding item
            pickle.dump(item_emb, open("data/yelp/handled/{}.pkl".format(model_args.output_file), "wb"))

            results = None

        else:

            list_test_samples = []
            with open(data_args.test_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    list_test_samples.append(line)

            # start_time = time.time()
            with torch.no_grad():
                predict_results = trainer.predict(
                    target_dataset,
                    metric_key_prefix="predict",
                )
            # end_time = time.time()

            if trainer.is_world_process_zero():
                predictions = predict_results.predictions
                assert len(predictions) == len(list_test_samples)
                hidden_states = predict_results.label_ids

                output_prediction_file = os.path.join(training_args.output_dir, model_args.output_file)

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, p in enumerate(predictions):
                        samp = list_test_samples[idx]
                        #samp["target"] = ehr_tokenizer.med_voc.idx2word[p]
                        samp["hidden_states"] = hidden_states[idx].astype(float).tolist()
                        samp["target"] = p.astype(float).tolist()
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

                results = None

    return results


if __name__ == "__main__":

    train()







