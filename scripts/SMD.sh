python -u run.py \
    --seed 1727934215 \
    --mode test \
    --data_path ./data/SMD \
    --data_name SMD \
    --n_vars 38 \
    --model_name Autoencoder \
    --data_embed local \
    --e_layer 5 \
    --n_heads 8 \
    --d_model 128 \
    --d_ff 256 \
    --win_size 300 \
    --step_size 300 \
    --test_step_size 300 \
    --percentile 99.5 \
    --ckpt ckpt