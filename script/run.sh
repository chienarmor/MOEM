#export train_batch_size=8
#export eval_batch_size=32
#export save_steps=40
#export eval_steps=40
#export epochs=20

export train_batch_size=2
export eval_batch_size=2
export epochs=2
export save_steps=2
export eval_steps=2


export learning_rate=3e-4
export alpha=0.5
export data_format=AO
export use_marker=True
export constraint_decoding=True
export model_name_or_path=
export save_strategy=epoch
export evaluation_strategy=steps
export marker_type=AO
export shot_ratio_index="-1[+]-1[+]0"
export warmup_ratio=0
export load_best_model_at_end=False

export CUDA_VISIBLE_DEVICES=0
export dataset=unified
export seed=5
export output_dir=./outputs/${dataset}_${seed}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python ./run.py \
  --do_train True \
  --do_predict True\
  --predict_with_generate \
  --overwrite_output_dir \
  --model_name_or_path=${model_name_or_path} \
  --dataset=${dataset} \
  --seed=${seed} \
  --output_dir=${output_dir} \
  --data_format=${data_format} \
  --per_device_train_batch_size=${train_batch_size} \
  --per_device_eval_batch_size=${eval_batch_size} \
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
  --save_total_limit 10 \
  --shot_ratio_index=${shot_ratio_index} \
  --marker_type=${marker_type} \
  --warmup_ratio=${warmup_ratio} \
  --lowercase True \
  --multi_path True \
  --single_view_type rank \
  --sort_label True \
  --ctrl_token post \
  --multi_task True
