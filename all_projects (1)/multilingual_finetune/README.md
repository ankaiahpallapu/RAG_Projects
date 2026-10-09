# Multilingual LLM Fine-tuning Pipeline

End-to-end: **data prep -> LoRA/QLoRA training -> per-language evaluation -> adapter merge -> latency benchmark**, configured from one YAML file. Ships configured for 6 languages (English, Hindi, Spanish, French, German, Telugu); add more in `languages:`.

```
 HF dataset / your JSONL ─► clean + dedupe ─► language-balanced sampling ─► stratified train/val
        └─► LoRA or QLoRA training (response-only loss) ─► adapter
              ├─► per-language perplexity: base vs tuned
              └─► merge into dense checkpoint ─► latency benchmark vs unmerged baseline
```

## Run it

```bash
pip install -r requirements.txt                 # needs a CUDA GPU for real training

python tests/test_data.py && python tests/test_formatting.py     # offline sanity tests

# dry-run the data stage on synthetic samples
python -m mlft.sample_data --out data/sample.jsonl
python -m mlft.data_prep --set data.source=jsonl --set data.jsonl_path=data/sample.jsonl \
    --set 'data.columns={instruction: instruction, response: response, lang: lang}'

# real run (downloads Aya, trains, evaluates, merges, benchmarks)
python -m mlft.data_prep --config config.yaml
python -m mlft.train     --config config.yaml
python -m mlft.evaluate  --config config.yaml
python -m mlft.merge     --config config.yaml
python -m mlft.benchmark --config config.yaml --baseline-adapter outputs/run1/adapter --baseline-dtype float32
```

Any setting can be overridden: `--set train.epochs=1 --set base_model=Qwen/Qwen2.5-7B-Instruct --set train.qlora=true`.

## Design notes

- **Language balancing** (`sampling_alpha`): examples per language ∝ `count^alpha`. `1.0` keeps natural proportions, `0` is equal shares; `0.5` stops English from drowning Telugu. Small languages are never over-sampled past what exists.
- **Response-only loss**: prompt tokens are masked (`-100`), so the model learns to answer, not to echo prompts.
- **Same chat template for training and inference** (`formatting.render_prompt`), which avoids a common silent quality loss.
- **QLoRA** (`train.qlora=true`) fits 7B-8B models on a 24 GB GPU; **merging happens on a 16-bit base**, never into 4-bit weights.
- **Checkpoints** are written every `eval_steps` and training auto-resumes from the latest.

## About the "70% latency reduction"

Latency gains come from the serving path, not from fine-tuning itself:

| Lever | Why it helps |
|---|---|
| Merge LoRA into the weights | removes the extra adapter matmuls/kernel launches on every token |
| fp32 -> bf16/fp16 | halves memory traffic; decoding is memory-bound |
| Batched generation (`--batch-size`) | amortises each weight read across requests |
| `--compile`, or serving the merged model with vLLM / int4 quantisation | fewer kernel launches, higher throughput |

`benchmark.py` runs the same prompts, greedy decoding and forced token count through a baseline (base + *unmerged* adapter) and the optimised model, and prints the measured per-request reduction per language. **The percentage depends on the baseline you pick and your hardware, so report the number the script gives you, with the baseline described.**

## Layout

```
config.yaml         all settings              mlft/data_utils.py  cleaning, balancing, splitting (pure Python)
mlft/data_prep.py   build train/val sets      mlft/formatting.py  chat template, label masking, padding
mlft/train.py       LoRA / QLoRA trainer      mlft/evaluate.py    per-language perplexity
mlft/merge.py       adapter -> dense model    mlft/benchmark.py   latency comparison
```
