# source ../../ditto/myenv/bin/activate

export HF_HOME="$HOME/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

python - <<'PY'
import os
from transformers import AutoModel, AutoTokenizer

model_name = "roberta-base"
cache_dir = os.environ["TRANSFORMERS_CACHE"]

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print("Downloading model...")
AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

print("BERT cached successfully at:", cache_dir)
PY