for dataset in laptop14 rest14 rest15 rest16
do
  for top_k in $(seq 5 5)
  do
    for ratio in 0.01
#    for shot in 1 5 10
    do
      #for seed in 5 10 15 20 25 30 35 40 45 3407
      for seed in 3407
      do
      #export train_batch_size=8
      #export eval_batch_size=32
      #export save_steps=40
      #export eval_steps=40
      #export epochs=20

      export per_device_train_batch_size=8
      export per_device_eval_batch_size=32
      export epochs=200
      export save_steps=40
      export eval_steps=40


      export learning_rate=3e-4
      export alpha=0.5
      export data_format=AO
      export use_marker=False
      export constraint_decoding=True
      export model_name_or_path=
      export save_strategy=epoch
      export evaluation_strategy=epoch
      export marker_type=AO
#      export shot_ratio_index="${shot}[+]-1[+]0"
      export shot_ratio_index="-1[+]${ratio}[+]0"
      export warmup_ratio=0
      export load_best_model_at_end=True

      export CUDA_VISIBLE_DEVICES=1
      export task_name=aste
      export dataset=${dataset}
      export output_dir=./outputs/Mvp/${task_name}/${dataset}_seed${seed}/ratio/${ratio}
#      export output_dir=./outputs/${task_name}/${dataset}_seed${seed}/ratio/${ratio}
      export HF_DATASETS_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1

      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python ./run.py \
        --do_train True \
        --do_predict True\
        --predict_with_generate \
        --overwrite_output_dir \
        --model_name_or_path=${model_name_or_path} \
        --task_name=${task_name} \
        --dataset=${dataset} \
        --seed=${seed} \
        --output_dir=${output_dir} \
        --data_format=${data_format} \
        --per_device_train_batch_size=${per_device_train_batch_size} \
        --per_device_eval_batch_size=${per_device_eval_batch_size} \
        --learning_rate=${learning_rate} \
        --num_train_epochs=${epochs} \
        --save_strategy=${save_strategy} \
        --save_steps=${save_steps} \
        --lr_scheduler_type=linear \
        --use_marker=${use_marker} \
        --constraint_decoding ${constraint_decoding} \
        --alpha ${alpha} \
        --use_fast_tokenizer \
        --evaluation_strategy=${evaluation_strategy} \
        --eval_steps=${eval_steps} \
        --load_best_model_at_end=${load_best_model_at_end} \
        --metric_for_best_model eval_f1_score \
        --save_total_limit 1 \
        --shot_ratio_index=${shot_ratio_index} \
        --marker_type=${marker_type} \
        --warmup_ratio=${warmup_ratio} \
        --lowercase True \
        --multi_path True \
        --single_view_type rank \
        --sort_label True \
        --ctrl_token post \
        --multi_task True \
        --max_length 300 \
        --top_k=${top_k} \
        --progressive_feature False
      done
    done
  done
done
