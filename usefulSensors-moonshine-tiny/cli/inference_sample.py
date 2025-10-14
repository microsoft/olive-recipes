import onnxruntime as ort
import numpy as np
from transformers import AutoProcessor, AutoConfig, AutoTokenizer
from datasets import load_dataset, Audio

model_name = 'UsefulSensors/moonshine-tiny'

model_path = "optimized-model/model.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

inputs = processor(
    sample["array"],
    return_tensors="np",
    sampling_rate=processor.feature_extractor.sampling_rate
)

seconds = len(sample["array"]) / processor.feature_extractor.sampling_rate
upper = int(seconds * 6 + 8)
max_length = min(upper, getattr(config, "max_position_embeddings", upper))

start_id = int(config.decoder_start_token_id)
eos_id = int(config.eos_token_id)
generated_ids = [start_id]

def feed(name, val, feeds):
    if any(i.name == name for i in session.get_inputs()):
        feeds[name] = val

for step in range(max_length):
    ort_inputs = {}
    feed('input_values', inputs.input_values.astype(np.float32), ort_inputs)
    feed('decoder_input_ids', np.array([generated_ids], dtype=np.int64), ort_inputs)

    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    next_id = int(np.argmax(logits[0, -1, :]))
    generated_ids.append(next_id)

    if next_id == eos_id:
        break
    if (step + 1) % 10 == 0:
        print(f"Generated {len(generated_ids)} tokens...")

text = tokenizer.decode(np.array(generated_ids), skip_special_tokens=True)
print("\nTranscription:", text)
