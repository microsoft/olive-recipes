# tencent-HY-MT1.5-1.8B - Baseline PyTorch Evaluation

This folder contains an Olive recipe for evaluating the Hugging Face PyTorch base model `tencent/HY-MT1.5-1.8B` on WMT18 Chinese-to-English translation with LM-eval.

## Setup

```bash
pip install -r requirements.txt
```

## Run evaluation

```bash
olive run --config tencent-HY-MT1.5-1.8B_pytorch_with_eval.json
```

## Evaluation results

PyTorch baseline WMT18 Chinese-to-English translation metrics, run on GPU:

- BLEU: `12.634090637487896`
- chrF: `35.290981008260474`
- TER: `85.7224234039956`
- Samples: `3981`
