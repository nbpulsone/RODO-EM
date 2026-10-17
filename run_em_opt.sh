#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J "RODOEM_test"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100-80G"
#SBATCH -x gpu-6-[01-20]

#SBATCH -o ./logs/robust_opt_%j.out
#SBATCH -e ./logs/robust_opt_%j.err

# interactive command: srun -N 1 -n 2 --mem=32768 --time=120 --partition=short --gres=gpu:1 -x "gpu-6-[01-20]" --pty /usr/bin/bash
# module load python/3.11.12/7vjddo3
# source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

# example run: ./run_em_opt.sh 1 worst WDC/category_50un_50cc verification_logs

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
EMA=0.0
WD=0.01
SM="resample_loss" # options: "resample_loss", "resample_f1", "balanced", "none"
USES_LORA=true # set true for llama
echo "Optimizer is ${OPT}, running on dataset ${TASK}, with out directory ${OUTDIR}"

# out result file path
OUTFILE="${OUTDIR}/rodoem_runtime_results_{$SLURM_JOB_ID}.txt"

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

MODEL_ARGS=()
if [ "$USES_LORA" = true ]; then
    MODEL_ARGS+=(
        --lora
        --lora_r 8
        --lora_alpha 16
        --lora_dropout 0.05
        --bf16
    )
else
    MODEL_ARGS+=(
        --fp16
    )
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
    --save_model \
    --use_gpu \
    --outfile "$OUTFILE" \
    --opt "$OPT" \
    --ema_alpha "$EMA" \
    --weight_decay "$WD" \
    --eval_plots_path "$OUTDIR" \
    --measure_runtimes \
    "${MODEL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
