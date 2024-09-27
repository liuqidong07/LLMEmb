## Ablation Study
gpu_id=6
dataset="beauty"
seed_list=(42 43 44)
llm_emb_file="0722_avg_pca"
tau=2
alpha=0.01
ts_user=9
ts_item=4

## w/o adapter
model_name="said_sasrec"
llm_emb_file="0722_avg_128"
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

## w/o frozen
model_name="llmemb_sasrec"
llm_emb_file="0722_avg_pca"
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
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --log
done


## w/o align
model_name="llmemb_sasrec"
llm_emb_file="0722_avg_pca"
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
                --alpha 0 \
                --tau ${tau} \
                --log
done


## w/o sft
model_name="llmemb_sasrec"
llm_emb_file="notrain_avg_beauty_pca"
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



