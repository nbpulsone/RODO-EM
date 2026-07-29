#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16g
#SBATCH -J "DITTO_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100|A100-80G|A30|V100|P100|L40S"
#SBATCH -x gpu-6-[01-20]

#SBATCH -o ./logs/robust_opt_%j.out
#SBATCH -e ./logs/robust_opt_%j.err

# interactive command: srun -N 1 -n 2 --mem=32768 --time=120 --partition=short --gres=gpu:1 -x "gpu-6-[01-20]" --pty /usr/bin/bash
# module load python/3.7.13/jz4yxoc
# source ../../ditto/myenv/bin/activate

# seed for current run
SEED=0

# parse cmd line arguments
OPTIMIZER="$1"
METRIC="$2"
TASK="$3"
OUTDIR="$4"

# optimization settings
OPT="robust-${OPTIMIZER}-${METRIC}"
LM="distilbert"
BS=64
LR=5e-5
N_EPOCHS=10
MAX_LEN=128
RW=4.0
EMA=0.0
WD=0.01
SM="resample_loss" # options: "resample_loss", "resample_f1", "balanced", "none"
echo "Optimizer is ${OPT}, running on dataset ${TASK}, with out directory ${OUTDIR}"

# out result file path
OUTFILE="${OUTDIR}/rodoem_results.txt"

EXTRA_ARGS=()
if [ "$SM" = "balanced" ]; then
    EXTRA_ARGS+=(--balanced_batches)
elif [ "$SM" = "resample_loss" ]; then
    EXTRA_ARGS+=(--domain_resample --resample_by loss)
elif [ "$SM" = "resample_f1" ]; then
    EXTRA_ARGS+=(--domain_resample --resample_by f1)
elif [ "$SM" = "resample_size" ]; then
    EXTRA_ARGS+=(--domain_resample --resample_by size)
fi

# Run the domain-optimization script
CUDA_VISIBLE_DEVICES=0 python train_ditto_domain_opt.py \
    --task "$TASK" \
    --run_id "$SEED" \
    --batch_size "$BS" \
    --max_len "$MAX_LEN" \
    --lr "$LR" \
    --n_epochs "$N_EPOCHS" \
    --finetuning \
    --lm "$LM" \
    --summarize \
    --fp16 \
    --save_model \
    --use_gpu \
    --outfile "$OUTFILE" \
    --opt "$OPT" \
    --robust_weight "$RW" \
    --ema_alpha "$EMA" \
    --weight_decay "$WD" \
    --eval_plots_path "$OUTDIR" \
    "${EXTRA_ARGS[@]}"
