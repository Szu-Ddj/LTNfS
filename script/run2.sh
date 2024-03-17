export CUDA_VISIBLE_DEVICES=7
lrs=("1e-3")
bert_lrs=("5e-6")

# model_names=("roberta_fol_text")
model_names=("bert_fol_rgraph")
# model_names=("lstm_fol")
input_types=("tt")
# datasets=("_dt" "_hc" "_la" "_fm")
# datasets=("la-" "dt-")
datasets=("vast_10")


# seeds=("3" "4" "5")
seeds=("0" "1" "2")

dropouts=("0.2")


label_ratios=(1)
FAD_ratios=(1)
RAD_ratios=(0)
for lr in ${lrs[*]}
do
    for bert_lr in ${bert_lrs[*]}
    do
        for model_name in ${model_names[*]}
        do
            for dataset in ${datasets[*]}
            do
                for seed in ${seeds[*]}
                do
                    for dropout in ${dropouts[*]}
                    do

                                            python3 ../codes/train.py \
                                            --lr $lr \
                                            --model_name $model_name \
                                            --dataset $dataset \
                                            --seed $seed \
                                            --dropout $dropout \
                                            --valset_ratio 0.0 \
                                            --bert_lr $bert_lr \
                                            --num_epoch 20 \
                                            --max_seq_len 60 \
                                            --nodes_num 25 \
                                            --edge_num 200 \
                                            # --with_text 1 \
                                            # --batch_size 64 \
                                            #  > "logs/"$learning_rate"_"$max_epoch"_"$batch_size".log" 
                    done
                done
            done
        done
    done
done

# nohup bash script/PStance_bernie_run.sh > logs/_PStance_bernie.out &
