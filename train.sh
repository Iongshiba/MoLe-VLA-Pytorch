skip_layer_number=14
mse_weight=0.5
balance_weight=0.1
KD_weight=0.5
train_rout=false
route_in_for=false
use_index=false
add_lstm=false 
ema_decay=0.999
seed=32

TASK=close_box
FUTURE_ACTION_STEPS=15

export HF_HOME="/home/longshiba/.cache/huggingface"
export EMA_DECAY=${ema_decay}
export MSE=${mse_weight}
export BALANCE=${balance_weight}
export KD=${KD_weight}
export TRAIN_ROUTE=${train_rout}
export FOR_ROUTE=${route_in_for} 
export SKIP_LAYER_NUMBER=$skip_layer_number
export USE_INDEX=${use_index} 
export ADD_LSTM=${add_lstm} 

# export PATH=$PATH:/path/to/MoLE_VLA
# export PYTHONPATH=$PYTHONPATH:/path/to/MoLE_VLA
# export PATH=$PATH:/path/to/MoLE_VLA/vla
# export PYTHONPATH=$PYTHONPATH:/path/to/MoLE_VLA/vla 

# SETTING=freeze_vit_window${FUTURE_ACTION_STEPS}
SETTING=route_in_for_${route_in_for}_balance_${balance_weight}_KD_${KD_weight}_ema_${ema_decay}_seed_${seed}_skip_layer_${skip_layer_number}_mse_${mse_weight}_freeze_none_window${FUTURE_ACTION_STEPS}
# SETTING=index_strategy_freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_skip_llama_layer_${skip_layer_number}_lstm_architecture_${add_lstm}_${TASK}_route_in_for_${route_in_for}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_MoE_bz_4_new_DiT_baseline_test
echo ${SETTING}
# SETTING=freeze_none_window${FUTURE_ACTION_STEPS}_Llama2_bz_4_vla_llm_align_baseline_task3
FREEZE_VISON=true
FREEZE_LLM=true
LOAD_DIT=true
# LLM_LAYER=mix2_AdaptiveLayerDefaultDit
# LLM_LAYER=mix_avgNoneDpPooling_m2Pooling
LLM_LAYER=mix_freezeLLM_AdaptiveLayerDefaultDitsWithLearnable_test


device='0'  



SETTING=${SETTING} RANDOM_SEED=${seed} USE_INDEX=${use_index} FOR_ROUTE=${route_in_for} TRAIN_ROUTE=True ADD_LSTM=${add_lstm} SKIP_LAYER_NUMBER=$skip_layer_number CUDA_VISIBLE_DEVICES=$device torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/train.py \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix rlbench \
  --vla.expected_world_size 1 \
  --vla.global_batch_size 1 \
  --vla.per_device_batch_size 1 \
  --vla.learning_rate 2e-5 \
  --vla.epochs 1 \
  --vla.freeze_vision_backbone ${FREEZE_VISON} \
  --vla.freeze_llm_backbone ${FREEZE_LLM} \
  --data_root_dir /home/longshiba/tensorflow_datasets \
  --run_root_dir results/train/${TASK} \
  --run_id exp_cx_LLMLAYER_${LLM_LAYER}_${TASK}_${SETTING} \
  --image_aug false \
  --save_interval 600 \
  --action_dim 7 \
  --repeated_diffusion_steps 8 \
  --future_action_window_size ${FUTURE_ACTION_STEPS} \
  --load_dit ${LOAD_DIT} \
  --action_model_type DiT-B \
  --is_resume False \
  --pretrained_checkpoint "/home/longshiba/.cache/huggingface/hub/models--CogACT--CogACT-Base/snapshots/6550bf0992f162fc5d74f14ffee30771a9433363/checkpoints/CogACT-Base.pt" \
  






#### close_box  close_laptop_lid   put_rubbish_in_bin  unplug_charger  water_plants  toilet_seat_down
