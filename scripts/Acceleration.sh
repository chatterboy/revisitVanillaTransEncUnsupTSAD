python -u run.py \
    --seed 1728959277 \
    --mode test \
    --data_path ./data/UCR \
    --data_name Acceleration \
    --n_vars 1 \
    --model_name Autoencoder \
    --data_embed local \
    --e_layer 2 \
    --n_heads 8 \
    --d_model 256 \
    --d_ff 512 \
    --win_size 200 \
    --step_size 400 \
    --test_step_size 200 \
    --norm standard \
    --percentile 99.9 \
    --ckpt ckpt