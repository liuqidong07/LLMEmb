## LLM2X
gpu_id=1
dataset="beauty"
seed_list=(42 43 44)
llm_emb_file="0813_ada_pca"
ts_user=9
ts_item=4

model_name="said_sasrec"
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
                --check_path "llm2x" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --llm_emb_file ${llm_emb_file} \
                --log
done

model_name="said_bert4rec"
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
                --check_path "llm2x" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --llm_emb_file ${llm_emb_file} \
                --log
done

model_name="said_gru4rec"
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
                --check_path "llm2x" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --llm_emb_file ${llm_emb_file} \
                --log
done