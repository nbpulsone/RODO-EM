#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64g
#SBATCH -J "WDCMD_grid_adaptive"
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH -C "A100-80G"
#SBATCH -x gpu-6-[01-20]
#SBATCH --array=0-17%10
#SBATCH -o ./logs/WDCMD_grid_%A_%a.out
#SBATCH -e ./logs/WDCMD_grid_%A_%a.err

set -euo pipefail
mkdir -p ./logs

# module load python/3.11.12/7vjddo3
# source /home/nbpulsone/sandbox_research/rodoem_llm_ft/modenv/bin/activate

# use one shared local Hugging Face cache for every array task.
export HF_HOME="$HOME/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

if [ ! -d "$HF_HUB_CACHE" ]; then
    echo "ERROR: Hugging Face cache not found at $HF_HUB_CACHE"
    exit 1
fi

# -------------------------
# experiment settings
# -------------------------
TASK="WDC/category_50un_50cc"
N_EPOCHS=10
MAX_LEN=128
SEED=0
LM="distilbert"
OPT="robust-1-worst"
OUTDIR="./WDCMD_worst1_grid_results"
USES_LORA=false # set to true for llama

# -------------------------
# Light hyperparameter grid
# Total jobs = 3 * 3 * 2 = 18
# This grid was used for to construct the main results for each optimizer (Section 6.1)
# -------------------------
BATCH_SIZES=(64)
LRS=(5e-5)
EMA_ALPHAS=(0.0 0.4 0.8)
SAMPLERS=("none" "balanced" "resample_loss")
WEIGHT_DECAYS=(0.00 0.01)

N_BS=${#BATCH_SIZES[@]}
N_LR=${#LRS[@]}
N_EMA=${#EMA_ALPHAS[@]}
N_SM=${#SAMPLERS[@]}
N_WD=${#WEIGHT_DECAYS[@]}

TOTAL=$(( N_BS * N_LR * N_EMA * N_SM * N_WD ))
IDX=${SLURM_ARRAY_TASK_ID}

if [ "$IDX" -ge "$TOTAL" ]; then
    echo "Array index $IDX is outside grid size $TOTAL; exiting."
    exit 0
fi

WD_IDX=$(( IDX % N_WD ))
IDX=$(( IDX / N_WD ))

SM_IDX=$(( IDX % N_SM ))
IDX=$(( IDX / N_SM ))

EMA_IDX=$(( IDX % N_EMA ))
IDX=$(( IDX / N_EMA ))

LR_IDX=$(( IDX % N_LR ))
IDX=$(( IDX / N_LR ))

BS_IDX=$(( IDX % N_BS ))

BS=${BATCH_SIZES[$BS_IDX]}
LR=${LRS[$LR_IDX]}
EMA=${EMA_ALPHAS[$EMA_IDX]}
SM=${SAMPLERS[$SM_IDX]}
WD=${WEIGHT_DECAYS[$WD_IDX]}

mkdir -p "$OUTDIR"

TASK_SAFE=${TASK//\//_}
RUN_NAME="task=${TASK_SAFE}_opt=${OPT}_bs=${BS}_lr=${LR}_ema=${EMA}_sampler=${SM}_wd=${WD}_seed=${SEED}"
OUTFILE="${OUTDIR}/${RUN_NAME}.txt"
PLOTDIR="${OUTDIR}/plots_${RUN_NAME}"
MANIFEST="${OUTDIR}/grid_manifest.csv"

# write a lightweight manifest so the completed grid can be audited later.
if [ ! -f "$MANIFEST" ]; then
    echo "array_task_id,task,opt,batch_size,lr,ema_alpha,sampler,weight_decay,seed,outfile" > "$MANIFEST"
fi
printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${SLURM_ARRAY_TASK_ID}" "$TASK" "$OPT" "$BS" "$LR" "$EMA" "$SM" "$WD" "$SEED" "$OUTFILE" >> "$MANIFEST"

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

echo "Running grid job ${SLURM_ARRAY_TASK_ID}/${TOTAL}"
echo "TASK=$TASK OPT=$OPT BS=$BS LR=$LR EMA=$EMA SM=$SM WD=$WD SEED=$SEED"
echo "Allocated node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi || true

CUDA_VISIBLE_DEVICES=0 python train_ditto_domain_opt.py \
    --task "$TASK" \
    --run_id "$SEED" \
    --batch_size "$BS" \
    --max_len "$MAX_LEN" \
    --lr "$LR" \
    --n_epochs "$N_EPOCHS" \
    --finetuning \
    --summarize \
    --lm "$LM" \
    --save_model \
    --use_gpu \
    --outfile "$OUTFILE" \
    --opt "$OPT" \
    --ema_alpha "$EMA" \
    --weight_decay "$WD" \
    --eval_plots_path "$PLOTDIR" \
    "${MODEL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
