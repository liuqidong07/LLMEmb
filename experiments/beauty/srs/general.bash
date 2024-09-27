## General Model without any enhancement
gpu_id=1
dataset="beauty"
seed_list=(42)
ts_user=9
ts_item=4

model_name="sasrec_seq"
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
                --check_path "" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --log
done

# get embedding
model_name="sasrec_seq"
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
                --check_path "" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --do_emb
done