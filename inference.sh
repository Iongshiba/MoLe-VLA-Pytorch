# ====== Arguments ======
skip_layer_number=14
mse_weight=0.5
balance_weight=0.1
KD_weight=0.5
train_rout=false
route_in_for=false
use_index=false

export EMA_DECAY=${ema_decay}
export MSE=${mse_weight}
export BALANCE=${balance_weight}
export KD=${KD_weight}
export TRAIN_ROUTE=${train_rout}
export FOR_ROUTE=${route_in_for} 
export SKIP_LAYER_NUMBER=$skip_layer_number
export USE_INDEX=${use_index} 
export ADD_LSTM=${add_lstm} 

python scripts/inference_episode.py
