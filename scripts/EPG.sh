python -u run.py \
    --seed 1729220393 \
    --mode test \
    --data_path ./data/UCR \
    --data_name EPG \
    --n_vars 1 \
    --model_name Autoencoder \
    --data_embed local \
    --e_layer 2 \
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512 \
    --win_size 30 \
    --step_size 30 \
    --test_step_size 30 \
    --percentile 99.8 \
    --ckpt ckpt