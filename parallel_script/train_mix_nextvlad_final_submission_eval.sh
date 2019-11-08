#!/usr/bin/env bash

WORKDIR=/pool0/ml/elv-youliangyu/project/elv-ml/yt8m
datapath=$WORKDIR/data/v3/frame
eval_path=$WORKDIR/data/v2/frame/validate_validate_tiny
test_path=$WORKDIR/data/v2/frame/test

model_name=MixNeXtVladModel
parameters="--groups=8 --nextvlad_cluster_size=128 --nextvlad_hidden_size=2048 \
            --expansion=2 --gating_reduction=16  --drop_rate=0.5 --final_drop=0. \
            --mix_number=3 --cl_temperature=1 --cl_lambda=1 --num_gpu=1"

train_dir=parallel1_mix3_nextvlad_x2_1T1_8g_5l2_5drop_128k_2048_80_logistic_fix_final_Nov6
result_folder=results

echo "model name: " $model_name
echo "model parameters: " $parameters

echo "training directory: " $train_dir
echo "data path: " $datapath
echo "evaluation path: " $eval_path
echo "results folder: " $result_folder

python parallel_eval.py ${parameters} --batch_size=160 --video_level_classifier_model=LogisticModel --l2_penalty=1e-5\
               --label_loss=CrossEntropyLoss --eval_data_pattern=${eval_path}/validate*.tfrecord --train_dir ${train_dir} \
               --run_once=True --final_lambda=1 --final_temperature=1 #--segment_labels=True

#mkdir -p $result_folder
#python inference.py --output_file ${result_folder}/${train_dir}_v2_test_top20.csv \
#                    --input_data_pattern=${test_path}/*.tfrecord --train_dir ${train_dir} \
#                    --batch_size=80 --num_readers=8 --top_k=20

#python inference.py --output_file ${result_folder}/${train_dir}_val_single.csv \
#                    --input_data_pattern=${eval_path}/*.tfrecord --train_dir ${train_dir} \
#                    --batch_size=1024 --num_readers=8 --top_k=60 --segment_labels=True

#python segment_eval_inference.py   --submission_file=${result_folder}/${train_dir}_val_single.csv --eval_data_pattern=${eval_path}/*.tfrecord --top_n=100000

