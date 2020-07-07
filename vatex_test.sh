python -m generate \
    Data/bpe_data \
    --dataset-impl raw \
    --gen-subset valid \
    --task vatex_translation_lev \
    --path checkpoints/checkpoint_best.pt \
    --iter-decode-max-iter 9 \
    --iter-decode-eos-penalty 0 \
    --beam 1 --remove-bpe \
    --print-step \
    --batch-size 40
