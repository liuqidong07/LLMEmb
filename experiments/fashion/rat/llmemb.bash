## Our Method LLMEmb
gpu_id=1
dataset="fashion"
seed_list=(42 43 44)
llm_emb_file="0722_avg_pca"
tau=2
alpha=0.01
ts_user=3
ts_item=4


# model_name="llmemb_sasrec"
# for seed in ${seed_list[@]}
# do
#     python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 128 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "llmemb" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze_emb \
#                 --llm_emb_file ${llm_emb_file} \
#                 --alpha ${alpha} \
#                 --tau ${tau} \
#                 --log
# done

model_name="llmemb_bert4rec"
for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 128 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "llmemb" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --freeze_emb \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --log
done

# model_name="llmemb_gru4rec"
# for seed in ${seed_list[@]}
# do
#     python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 128 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "llmemb" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze_emb \
#                 --llm_emb_file ${llm_emb_file} \
#                 --alpha ${alpha} \
#                 --tau ${tau} \
#                 --log
# done
