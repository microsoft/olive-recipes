from argparse import ArgumentParser
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.utils import pack_to_uint8


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to the Hugging Face model repository.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed Olive model.",
    )
    return parser.parse_args()


def unpack_packed_32bit(packed: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Unpack a packed int32 tensor into its per-value integers in [0, 2^bits - 1].

    Returns a tensor with one extra last dimension of size (32//bits).
    Example: packed [A,B] -> unpacked [A,B,32//bits]
    """
    assert 32 % bits == 0, "bits must divide 32"
    # vals_per_int = 32 // bits
    mask = (1 << bits) - 1

    # Avoid sign-extension on right shift:
    x = packed.to(torch.int64) & 0xFFFFFFFF  # treat as uint32 payload

    shifts = torch.arange(0, 32, bits, device=x.device, dtype=torch.int64)  # [vals_per_int]
    # [A,B,vals_per_int]
    return (x.unsqueeze(-1) >> shifts.view(1, 1, -1)) & mask  # int64


def process_qweight(tensor: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Process quantized weight tensor by unpacking and repacking."""
    n = tensor.shape[1]
    w_u = unpack_packed_32bit(tensor, bits=bits)  # [Kp, N, V]
    # Arrange so each packed row expands to V consecutive K rows:
    # [Kp, V, N] -> [Kp*V, N]
    qweight_int = w_u.permute(0, 2, 1).reshape(-1, n).t().to(torch.uint8)
    return pack_to_uint8(qweight_int, bits=bits)


def process_qzeros(tensor: torch.Tensor, bits: int = 2) -> torch.Tensor:
    """Process quantized zero point tensor by repacking."""
    g = tensor.shape[0]
    z_u = unpack_packed_32bit(tensor, bits=bits)  # [G, Np, V]
    # Expand packed out_features: [G, Np*V]
    qzeros_int = z_u.reshape(g, -1).t().to(torch.uint8)
    return pack_to_uint8(qzeros_int, bits=bits)


def main(model_path: str, save_path: str):
    # load and process weights into Olive quant format
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = load_file(hf_hub_download(repo_id=model_path, filename="model.safetensors"), device=device)
    for key in tqdm(list(weights), desc="Processing weights"):
        if key.endswith("g_idx"):
            del weights[key]
            continue

        if key.endswith("qweight"):
            weights[key] = process_qweight(weights[key])
        elif key.endswith("qzeros"):
            weights[key] = process_qzeros(weights[key])
        elif key.endswith("scales"):
            weights[key] = weights[key].t().contiguous()

        weights[key] = weights[key].cpu()

    # save processed weights, config, tokenizer
    config = AutoConfig.from_pretrained(model_path)
    config.quantization_config = OliveHfQuantizationConfig(bits=2, group_size=128, symmetric=False)
    config.save_pretrained(save_path)
    AutoTokenizer.from_pretrained(model_path).save_pretrained(save_path)
    save_file(weights, Path(save_path) / "model.safetensors")


if __name__ == "__main__":
    args = parse_args()
    main(args.model_path, args.output_path)
