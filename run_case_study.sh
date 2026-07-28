#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH -J "case_study_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100|A100-80G|A30|V100|P100|L40S"
#SBATCH -x gpu-6-[01-20]

#SBATCH -o ./logs/robust_opt_%j.out
#SBATCH -e ./logs/robust_opt_%j.err

# interactive command: srun -N 1 -n 2 --mem=32768 --time=120 --partition=short --gres=gpu:1 -x "gpu-6-[01-20]" --pty /usr/bin/bash

module load python/3.7.13/jz4yxoc
source ../../ditto/myenv/bin/activate

MATCHER_INPUT="input/wa_price_test.jsonl"
LABELED_TEST="data/wa_price/wa_test.txt"

# run the domain-optimization script
CUDA_VISIBLE_DEVICES=0 python train_ditto_domain_opt.py \
    --task "WA/price" \
    --batch_size 64 \
    --max_len 128 \
    --lr 5e-5 \
    --n_epochs 10 \
    --finetuning \
    --lm distilbert \
    --summarize \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile "./case_study/wa_price_results_DRO.txt" \
    --opt robust-dro-worst \
    --robust_weight 4.0 \
    --balanced_batches \
    --ema_alpha 0.8 \
    --weight_decay 0.01 \
    --eval_plots_path "./case_study/"


# generate predictions for the base model 
CUDA_VISIBLE_DEVICES=0 python matcher.py \
  --task WA/price \
  --input_path "$MATCHER_INPUT" \
  --output_path output/wa_price_predictions_base.jsonl \
  --lm distilbert \
  --max_len 128 \
  --summarize \
  --use_gpu \
  --fp16 \
  --checkpoint checkpoints/WA_price/robust-dro-worst/run_0/base_model.pt

# generate predictions for the robust model 
CUDA_VISIBLE_DEVICES=0 python matcher.py \
  --task WA/price \
  --input_path "$MATCHER_INPUT" \
  --output_path output/wa_price_predictions_robust.jsonl \
  --lm distilbert \
  --max_len 128 \
  --summarize \
  --use_gpu \
  --fp16 \
  --checkpoint checkpoints/WA_price/robust-dro-worst/run_0/robust_model.pt

# calculate the dollar amount preserved due to matching for the base moodel and robust model
CUDA_VISIBLE_DEVICES=0 python case_study_price_calculator.py \
  --task WA/price \
  --input_path "$LABELED_TEST" \
  --predictions output/wa_price_predictions_base.jsonl \
  --result_path case_study/wa_price_results_base.txt

CUDA_VISIBLE_DEVICES=0 python case_study_price_calculator.py \
  --task WA/price \
  --input_path "$LABELED_TEST" \
  --predictions output/wa_price_predictions_robust.jsonl \
  --result_path case_study/wa_price_results_robust.txt
