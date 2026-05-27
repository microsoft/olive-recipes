"""User script for Qwen3-VL vision evaluation with Olive.

Provides post-processing function that decodes model output to answer text
for use with Olive's vision evaluation metrics (exact_match).
"""

import re


def post_process_vision_output(output):
    """Post-process Qwen3-VL model output to extract answer text.

    For multiple-choice VQA (AI2D), extracts the first digit (1-4) from output.
    For open-ended VQA, returns the decoded text directly.

    Args:
        output: Model output tensor or decoded text string.

    Returns:
        str or list[str]: Predicted answer(s).
    """
    if isinstance(output, str):
        text = output.strip()
    elif isinstance(output, list) and output and isinstance(output[0], str):
        return [_extract_answer(t) for t in output]
    else:
        # If tensor output, convert token ids to text
        # This would require a tokenizer - for genai models, output is already decoded
        text = str(output)

    return _extract_answer(text)


def _extract_answer(text: str) -> str:
    """Extract answer from model response text.

    For multiple-choice, look for a digit 1-4.
    Otherwise return the full stripped response.
    """
    text = text.strip()
    # Try to find a single digit answer (multiple choice)
    m = re.search(r"\b([1-4])\b", text)
    if m:
        return m.group(1)
    # For short answers, return as-is
    # Take only the first line/sentence for cleaner matching
    first_line = text.split("\n")[0].strip()
    if len(first_line) < 100:
        return first_line
    return text[:100]
