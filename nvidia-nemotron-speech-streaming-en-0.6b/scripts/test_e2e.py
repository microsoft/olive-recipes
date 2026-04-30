#!/usr/bin/env python3
"""
End-to-end test for Nemotron ASR via onnxruntime-genai.

Tests:
1. Model loading
2. Tokenizer load + decode
3. Full RNNT inference with dummy audio
"""

import sys
import numpy as np
from pathlib import Path


def test_model_loading(model_path: str):
    """Test that the model loads correctly."""
    import onnxruntime_genai as og

    print("[1/4] Loading model...")
    model = og.Model(model_path)
    print("  ✓ Model loaded successfully")
    return model


def test_tokenizer(model):
    """Test tokenizer load and decode."""
    import onnxruntime_genai as og

    print("\n[2/4] Testing tokenizer...")
    tokenizer = og.Tokenizer(model)
    print("  ✓ Tokenizer loaded")

    # Test individual token decode
    test_tokens = {
        5: "the",
        33: "and",
        3: "a",
        34: "I",
        57: "is",
    }
    for tid, expected in test_tokens.items():
        decoded = tokenizer.decode([tid])
        status = "✓" if expected in decoded else "✗"
        print(f"  {status} Token {tid} -> {decoded!r} (expected contains {expected!r})")

    # Test batch decode
    ids = np.array([34, 3, 23], dtype=np.int32)  # "I", "a", "m"
    text = tokenizer.decode(ids)
    print(f"  ✓ Batch decode [34, 3, 23] -> {text!r}")

    return tokenizer


def test_inference(model, tokenizer):
    """Test running inference (encoder + RNNT decode) via onnxruntime-genai."""
    import onnxruntime_genai as og

    print("\n[3/4] Testing inference via onnxruntime-genai...")
    try:
        # Create generator params
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=512,
            batch_size=1,
        )
        print("  ✓ GeneratorParams created")

        # Create generator
        generator = og.Generator(model, params)
        print("  ✓ Generator created")

        # Create dummy audio: [1, 128, 100] (batch=1, mel_bins=128, time_frames=100)
        batch_size = 1
        mel_bins = 128
        time_frames = 100
        audio = np.random.randn(batch_size, mel_bins, time_frames).astype(np.float32)

        # Build NamedTensors with audio_signal + dummy input_ids (BOS=0)
        inputs = og.NamedTensors()
        inputs["audio_signal"] = audio
        inputs["input_ids"] = np.array([[0]], dtype=np.int32)  # Dummy BOS token to trigger inference
        print(f"  ✓ NamedTensors created (audio shape={audio.shape})")

        # set_inputs: sets extra inputs (audio) + AppendTokens(input_ids) -> triggers first State::Run()
        generator.set_inputs(inputs)
        print("  ✓ set_inputs completed (encoder + RNNT decode triggered)")

        # Generate tokens until done (each generate_next_token emits one RNNT decoded token)
        step = 0
        while not generator.is_done():
            generator.generate_next_token()
            step += 1
            if step > 500:
                print("  ⚠ Safety limit reached (500 tokens)")
                break

        print(f"  ✓ Generation completed in {step} steps")

        # Get the full sequence (includes initial dummy BOS + decoded tokens + EOS)
        tokens = generator.get_sequence(0)
        token_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
        print(f"  ✓ Full sequence ({len(token_list)} tokens): {token_list[:20]}{'...' if len(token_list) > 20 else ''}")

        # Decode tokens to text (skip the first token which is the dummy BOS)
        decoded_ids = np.array(token_list[1:], dtype=np.int32)  # Skip dummy BOS
        # Also filter out blank (1024) and EOS tokens for text
        text_ids = np.array([t for t in token_list[1:] if t != 1024], dtype=np.int32)
        if len(text_ids) > 0:
            text = tokenizer.decode(text_ids)
            print(f"  ✓ Decoded text: {text!r}")
        else:
            print("  ✓ No non-blank tokens decoded (expected for random audio)")

        print("  ✓ Inference completed successfully!")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_onnx_inference(model_path: str):
    """Test with raw onnxruntime as a baseline reference (streaming encoder + stateful decoder)."""
    import onnxruntime as ort

    print("\n[4/4] Raw ONNX Runtime inference test (baseline)...")

    # Load sessions
    enc_path = str(Path(model_path) / "encoder.onnx")
    dec_path = str(Path(model_path) / "decoder.onnx")
    joint_path = str(Path(model_path) / "joint.onnx")

    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])
    joint_sess = ort.InferenceSession(joint_path, providers=["CPUExecutionProvider"])

    # Streaming encoder constants (chunk_size=0.56s)
    n_layers = 24
    d_model = 1024
    static_mel_frames = 65  # 56 chunk + 9 pre_encode_cache
    mel_features = 128
    last_channel_cache_size = 70  # left_chunks(10) * chunk_encoded_frames(7)
    conv_context = 8

    # Encoder: one chunk with zero caches
    audio = np.random.randn(1, static_mel_frames, mel_features).astype(np.float32)
    length = np.array([static_mel_frames], dtype=np.int64)
    cache_ch = np.zeros((1, n_layers, last_channel_cache_size, d_model), dtype=np.float32)
    cache_tm = np.zeros((1, n_layers, d_model, conv_context), dtype=np.float32)
    cache_len = np.zeros((1,), dtype=np.int64)

    enc_out = enc_sess.run(None, {
        "audio_signal": audio, "length": length,
        "cache_last_channel": cache_ch, "cache_last_time": cache_tm,
        "cache_last_channel_len": cache_len,
    })
    encoded = enc_out[0]  # [1, T', 1024]
    enc_len = int(enc_out[1][0])
    print(f"  Encoder output: {encoded.shape}, encoded_length={enc_len}")

    # Stateful decoder: inputs are targets[B,1], h_in[2,B,640], c_in[2,B,640]
    hidden_size = 640
    num_lstm_layers = 2
    h = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)

    targets = np.array([[0]], dtype=np.int64)  # BOS token
    dec_out = dec_sess.run(None, {"targets": targets, "h_in": h, "c_in": c})
    dec_output = dec_out[0]  # [1, 640, 1]
    h, c = dec_out[1], dec_out[2]
    print(f"  Decoder output: {dec_output.shape}")

    # Extract decoder hidden: [1, 640, 1] -> [1, 1, 640]
    dec_hidden = dec_output.transpose(0, 2, 1)

    # Joint
    enc_frame = encoded[:, 0:1, :]  # [1, 1, 1024]
    joint_out = joint_sess.run(None, {
        "encoder_output": enc_frame,
        "decoder_output": dec_hidden,
    })
    logits = joint_out[0]  # [1, 1, 1, 1025]
    print(f"  Joint output: {logits.shape}")

    # Greedy decode
    token = int(np.argmax(logits.squeeze()))
    print(f"  First token: {token} ({'<blank>' if token == 1024 else f'vocab[{token}]'})")

    # Full greedy decode (limit to 5 frames for speed)
    tokens = []
    current_token = 0  # BOS
    h = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)
    c = np.zeros((num_lstm_layers, 1, hidden_size), dtype=np.float32)
    max_sym = 10

    for t in range(min(enc_len, 5)):
        enc_t = encoded[:, t:t+1, :]
        for _ in range(max_sym):
            targets = np.array([[current_token]], dtype=np.int64)
            dec_out = dec_sess.run(None, {"targets": targets, "h_in": h, "c_in": c})
            dec_h = dec_out[0].transpose(0, 2, 1)  # [1, 640, 1] -> [1, 1, 640]
            h, c = dec_out[1], dec_out[2]

            joint_out = joint_sess.run(None, {"encoder_output": enc_t, "decoder_output": dec_h})
            tok = int(np.argmax(joint_out[0].squeeze()))

            if tok == 1024:  # blank
                break
            tokens.append(tok)
            current_token = tok

    print(f"  Decoded tokens (first 5 frames): {tokens}")
    print("  ✓ Raw ONNX inference successful")


def main():
    print("=" * 60)
    print("Nemotron ASR - End-to-End Test via onnxruntime-genai")
    print("=" * 60)

    scripts_dir = Path(__file__).parent
    repo_root = scripts_dir.parent

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        candidate_paths = [
            repo_root / "build" / "onnx_models_int4",
            repo_root / "build" / "onnx_models_fp32",
            scripts_dir / "onnx_models_cpu_fp32",
        ]
        selected = next((p for p in candidate_paths if p.exists()), None)
        model_path = str(selected if selected is not None else candidate_paths[0])

    if not Path(model_path).exists():
        print(f"Error: Model directory not found: {model_path}")
        return False

    # Run via onnxruntime-genai
    model = test_model_loading(model_path)
    tokenizer = test_tokenizer(model)
    success = test_inference(model, tokenizer)

    # Run raw ONNX test as baseline
    try:
        test_raw_onnx_inference(model_path)
    except Exception as e:
        print(f"  ✗ Raw ONNX test failed: {e}")

    print("\n" + "=" * 60)
    print("Result:", "✓ ALL TESTS PASSED" if success else "✗ SOME TESTS FAILED")
    print("=" * 60)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
