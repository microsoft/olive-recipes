import os
import argparse
import onnxruntime_genai as og
from transformers import AutoProcessor


class OnnxGenaiInference:
    """Text-only inference using ONNX GenAI (language model only)."""

    def __init__(
        self,
        model_path: str,
        hf_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        cache_dir: str = None
    ):
        """
        Initialize ONNX GenAI inference.

        Args:
            model_path: Path to the ONNX model directory (containing genai_config.json)
            hf_model_path: HF model for processor/tokenizer (for chat template)
            cache_dir: Cache directory for HF model
        """
        print(f"Loading ONNX GenAI model from: {model_path}")
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)

        # Load HF processor for chat template
        print(f"Loading processor from: {hf_model_path}")
        self.processor = AutoProcessor.from_pretrained(
            hf_model_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )

        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ) -> str:
        """Generate text using ONNX GenAI model."""

        # Build message (text only)
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        input_tokens = self.tokenizer.encode(text)

        # Set generation parameters
        params = og.GeneratorParams(self.model)
        params.set_search_options(
            max_length=len(input_tokens) + max_new_tokens,
            do_sample=do_sample,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 1,
            temperature=temperature if do_sample else 1.0
        )
        params.input_ids = input_tokens

        # Generate
        generator = og.Generator(self.model, params)

        output_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            output_tokens.append(new_token)

        # Decode output
        output_text = self.tokenizer.decode(output_tokens)

        return output_text


def interactive_mode(model, args):
    """Run in interactive mode."""
    print("\n" + "="*50)
    print("Interactive Mode - Enter 'quit' or 'exit' to stop")
    print("="*50 + "\n")

    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in ['quit', 'exit']:
            break
        if not prompt:
            print("Please enter a prompt.")
            continue

        print("\nGenerating response...")
        try:
            response = model.generate(
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.greedy
            )
            print(f"\nResponse:\n{response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
            import traceback
            traceback.print_exc()

        print("-"*50 + "\n")

    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="ONNX GenAI Inference script for Qwen2.5-VL language model (text only)"
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the ONNX model directory"
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model for processor/tokenizer"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for HF model"
    )

    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for inference"
    )

    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding"
    )

    # Mode arguments
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Load model
    model = OnnxGenaiInference(
        model_path=args.model_path,
        hf_model_path=args.hf_model,
        cache_dir=args.cache_dir
    )

    if args.interactive:
        interactive_mode(model, args)
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")

        print("\nGenerating response...")
        response = model.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.greedy
        )

        print(f"\nResponse:\n{response}")
    else:
        print("Please provide --prompt or use --interactive mode")
        parser.print_help()


if __name__ == "__main__":
    main()
