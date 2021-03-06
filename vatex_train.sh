python -m train \
    Data/bpe_data \
    --dataset-impl raw \
    --save-dir checkpoints \
    --ddp-backend=no_c10d \
    --task vatex_translation_lev \
    --criterion vatex_nat_loss \
    --arch vatex_cmlm_transformer \
    --noise random_mask \
    --share-all-embeddings \
    --max-source-positions 100 \
    --max-target-positions 250 \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 14000 \
    --save-interval-updates 10000 \
    --max-update 300000 \
    --restore-file checkpoints/checkpoint54.pt
    
    
    
# Tip: # max-source-positions=400, 1 samples have invalid sizes and will be skipped