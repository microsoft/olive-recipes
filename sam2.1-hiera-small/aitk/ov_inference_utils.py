from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from PIL import Image


def add_ep_for_device(session_options, ep_name: str, device_type: str, ep_options: Optional = None):
    """
    Select an EP device by matching ep_name and ep_metadata['ov_device']

    'device_type' is a string (e.g. 'CPU', 'GPU', 'NPU', 'GPU.0')
    Matching is case-insensitive prefix on 'ep_metadata['ov_device']'
    So, 'GPU' matches 'GPU.0', 'GPU.1', etc. and the first such candidate is chosen
    """
    ep_devices = ort.get_ep_devices()
    candidates = [
        epd
        for epd in ep_devices
        if epd.ep_name == ep_name
        and getattr(epd, "ep_metadata", {}).get("ov_device", "").lower().startswith(device_type.lower())
    ]
    if not candidates:
        raise RuntimeError(f"No EP device matched ep_name={ep_name}, device_type={device_type}")
    ep_device = candidates[0]
    matched_ov_device = getattr(ep_device, "ep_metadata", {}).get("ov_device", "")
    print(f"Adding {ep_name} for {device_type} matched ov_device={matched_ov_device}")

    # Ambiguous prefix handling
    if len(candidates) > 1 and matched_ov_device.lower() != device_type.lower():
        print(
            f"Warning: device_type='{device_type}' matched {len(candidates)} "
            f"candidates; selected ov_device='{matched_ov_device}'. OVEP may "
            f"compile for a different device. Pass a fully-qualified device "
            f"name (e.g. '{matched_ov_device}') to disambiguate."
        )
    session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)


class IPromptedImageSegmentationModel(ABC):
    # There might be other prompt formats. For now, we only define the bbox one.
    @dataclass(frozen=True)
    class Prompt:
        foreground_points: List[Tuple[float, float]] = field(default_factory=list)
        background_points: List[Tuple[float, float]] = field(default_factory=list)
        # x, y, w, h
        bbox: Optional[Tuple[float, float, float, float]] = None

    @dataclass(frozen=True)
    class Prediction:
        mask: npt.NDArray

    @abstractmethod
    def segment_image(
        self, image: Image.Image, prompt: "IPromptedImageSegmentationModel.Prompt"
    ) -> "IPromptedImageSegmentationModel.Prediction":
        pass


def resize_image(
    image: Image.Image,
    target_size: int,
) -> Tuple[Image.Image, Tuple[float, float]]:
    """
    Resize image to (target_size, target_size) using bilinear interpolation.

    Returns:
        Tuple[Image.Image, Tuple[float, float]]: (resized_image, (scale_x, scale_y))
    """
    scale_x = target_size / image.width
    scale_y = target_size / image.height
    resized = image.resize((target_size, target_size), Image.Resampling.BILINEAR)
    return resized, (scale_x, scale_y)


def resize_normalize_adjust_dim(
    image: Image.Image,
    target_size: int,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize image to square, normalize with ImageNet mean/std, and convert to NCHW.

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: (pixels in NCHW, (scale_x, scale_y))
    """
    image = image.convert("RGB")
    resized, (scale_x, scale_y) = resize_image(image=image, target_size=target_size)
    pixels = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (pixels - mean) / std
    pixels = np.transpose(pixels, (2, 0, 1))
    pixels = np.expand_dims(pixels, axis=0)
    return pixels, (scale_x, scale_y)


class FacebookSam2_1HieraSmall(
    IPromptedImageSegmentationModel,
):
    _INPUT_IMAGE_SIZE = 1024
    _N_POINTS = 5
    _STABILITY_RANGE = 0.05
    _STABILITY_THRESHOLD = 0.98

    def __init__(
        self,
        model_path: dict[str, Path],
        device: Optional[str] = "CPU",
        execution_provider: Optional[str] = "CPUExecutionProvider",
    ) -> None:
        """
        FacebookSam2_1HieraSmall constructor

        Args:
            model_path (dict[str, Path]): A dictionary containing paths to the encoder and decoder ONNX models. Expected keys are 'vision_encoder' and 'mask_decoder'.
            device (Optional[str]): The target device for inference (e.g., 'CPU', 'GPU', 'NPU'). Default is 'CPU'.
            execution_provider (Optional[str]): The ONNX Runtime execution provider to use (e.g., 'OpenVINOExecutionProvider'). Default is 'CPUExecutionProvider'.

        Returns:
            None

        Raises:
            FileNotFoundError: If the encoder or decoder model files are not found at the specified paths.
        """
        self._encoder_path = model_path["vision_encoder"]
        self._decoder_path = model_path["mask_decoder"]
        if not self._encoder_path.exists():
            raise FileNotFoundError(f"Encoder model file not found: {self._encoder_path}")
        if not self._decoder_path.exists():
            raise FileNotFoundError(f"Decoder model file not found: {self._decoder_path}")

        # Initialize ONNX Runtime sessions for encoder and decoder
        if device is not None and execution_provider is not None:
            session_options = ort.SessionOptions()

            # device
            device_str = device.upper()

            # add EP for device
            add_ep_for_device(
                session_options,
                execution_provider,
                device_str,
            )
            print(f"Using specified EP '{execution_provider}' with device '{device_str}'")

            # create encoder and decoder session objects
            self._encoder = ort.InferenceSession(self._encoder_path, sess_options=session_options)
            self._decoder = ort.InferenceSession(self._decoder_path, sess_options=session_options)
            print(
                f"Successfully created ONNX Runtime sessions with specified EP '{execution_provider}' and device '{device_str}'."
            )
        else:
            print(f"Using default providers for ONNX Runtime sessions.")
            self._encoder = ort.InferenceSession(self._encoder_path)
            self._decoder = ort.InferenceSession(self._decoder_path)
            print(f"Successfully created ONNX Runtime sessions with default providers.")

        # Precompute encoder output index → decoder input name mapping
        self._encoder_decoder_map = self._build_encoder_decoder_map()

    def _build_encoder_decoder_map(self) -> dict:
        """
        Build a mapping from encoder output index to decoder input name.
        If encoder output names match decoder input names, uses them directly.
        Otherwise, falls back to matching by shape (C, H, W) for 4D tensors.
        """
        encoder_outputs_meta = self._encoder.get_outputs()
        decoder_inputs_meta = self._decoder.get_inputs()
        decoder_input_names = {x.name for x in decoder_inputs_meta}

        # Fast path: encoder output names already match decoder input names
        encoder_output_names = [x.name for x in encoder_outputs_meta]
        if all(name in decoder_input_names for name in encoder_output_names):
            return {i: name for i, name in enumerate(encoder_output_names)}

        # Fallback: match by shape (C, H, W) for 4D spatial tensors
        decoder_shape_to_name = {}
        for meta in decoder_inputs_meta:
            shape = meta.shape
            if isinstance(shape, list) and len(shape) == 4 and all(isinstance(d, int) for d in shape):
                decoder_shape_to_name[tuple(shape[1:])] = meta.name

        mapping = {}
        for i, meta in enumerate(encoder_outputs_meta):
            shape = meta.shape
            if isinstance(shape, list) and len(shape) == 4 and all(isinstance(d, int) for d in shape):
                key = tuple(shape[1:])
                if key in decoder_shape_to_name:
                    mapping[i] = decoder_shape_to_name[key]
                    continue
            raise ValueError(
                f"Encoder output {i} (name={meta.name}, shape={shape}) could not be matched "
                f"to any decoder input. Known decoder shapes: {decoder_shape_to_name}"
            )

        return mapping

    def _encode_prompt(
        self,
        prompt: IPromptedImageSegmentationModel.Prompt,
        scale_x: float,
        scale_y: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode the prompt into decoder model inputs

        Args:
            prompt (IPromptedImageSegmentationModel.Prompt): The input prompt containing
                the bounding box, foreground points, and background points.
            scale_x (float): The scaling factor for the X (width) axis, applied to
                prompt coordinates to map them into the model's input image space.
            scale_y (float): The scaling factor for the Y (height) axis, applied to
                prompt coordinates to map them into the model's input image space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements
                - point_coords (np.ndarray): The coordinates of the points in the prompt.
                - point_labels (np.ndarray): The labels of the points in the prompt.
        """
        point_coords = np.zeros((1, self._N_POINTS, 2), dtype=np.float32)
        # Default to -1: unused points
        point_labels = np.ones((1, self._N_POINTS), dtype=np.float32) * -1
        i_points = 0
        if prompt.bbox is not None:
            bbox_x, bbox_y, bbox_w, bbox_h = prompt.bbox
            point_coords[0, i_points, 0] = bbox_x * scale_x
            point_coords[0, i_points, 1] = bbox_y * scale_y
            point_labels[0, i_points] = 2
            i_points += 1
            point_coords[0, i_points, 0] = (bbox_x + bbox_w) * scale_x
            point_coords[0, i_points, 1] = (bbox_y + bbox_h) * scale_y
            point_labels[0, i_points] = 3
            i_points += 1
        for x, y in prompt.foreground_points:
            if i_points >= self._N_POINTS:
                break
            point_coords[0, i_points, 0] = x * scale_x
            point_coords[0, i_points, 1] = y * scale_y
            point_labels[0, i_points] = 1
            i_points += 1
        for x, y in prompt.background_points:
            if i_points >= self._N_POINTS:
                break
            point_coords[0, i_points, 0] = x * scale_x
            point_coords[0, i_points, 1] = y * scale_y
            point_labels[0, i_points] = 0
            i_points += 1
        return point_coords, point_labels

    def _calculate_stability_score(self, mask: np.ndarray) -> float:
        values = mask.flatten()
        certain_count = float(np.sum(values > self._STABILITY_RANGE))
        certain_and_uncertain_count = float(np.sum(values > -self._STABILITY_RANGE))
        if certain_and_uncertain_count == 0:
            return 1.0
        return certain_count / certain_and_uncertain_count

    def _select_mask(self, masks: np.ndarray, iou_predictions: np.ndarray) -> np.ndarray:
        """
        Reference:
        https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/mask_decoder.py
        """
        single_mask_stability_score = self._calculate_stability_score(masks[0])
        if single_mask_stability_score >= self._STABILITY_THRESHOLD:
            return masks[0]
        best_mask_index = np.argmax(iou_predictions[1:])
        return masks[best_mask_index + 1]

    def segment_image(
        self, image: Image.Image, prompt: IPromptedImageSegmentationModel.Prompt
    ) -> IPromptedImageSegmentationModel.Prediction:
        """
        Forward pass through the Encoder and Decoder models to get the predicted mask

        Args:
            image (PIL.Image.Image): The input image to be segmented.
            prompt (IPromptedImageSegmentationModel.Prompt): The input prompt containing
                the bounding box, foreground points, and background points.

        Returns:
            IPromptedImageSegmentationModel.Prediction: The predicted mask and associated information.
        """
        # Preprocess the image and get X and Y scaling factors for prompt coordinate adjustment
        pixels, (scale_x, scale_y) = resize_normalize_adjust_dim(image=image, target_size=self._INPUT_IMAGE_SIZE)

        # Forward pass through the Encoder model
        encoder_input_name = self._encoder.get_inputs()[0].name
        encoder_outputs = self._encoder.run(None, {encoder_input_name: pixels})

        # Build Decoder model inputs using precomputed encoder→decoder name mapping
        decoder_inputs = {}
        for i, output in enumerate(encoder_outputs):
            decoder_inputs[self._encoder_decoder_map[i]] = np.array(output)

        # Encode the prompt into decoder inputs
        point_coords, point_labels = self._encode_prompt(prompt=prompt, scale_x=scale_x, scale_y=scale_y)
        decoder_inputs.update(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
        )

        # Forward Pass through the Decoder model
        decoder_outputs = self._decoder.run(None, decoder_inputs)

        # capture output masks and IoU predictions from Decoder outputs
        output_masks = np.array(decoder_outputs[0])[0]
        output_iou_predictions = np.array(decoder_outputs[1])[0]

        # Post-process: resize masks to original image dimensions
        resized_masks = np.empty((output_masks.shape[0], image.height, image.width), dtype=output_masks.dtype)
        for i in range(output_masks.shape[0]):
            mask_img = Image.fromarray(output_masks[i], mode="F")
            mask_img = mask_img.resize((image.width, image.height), Image.Resampling.BILINEAR)
            resized_masks[i] = np.array(mask_img)
        output_masks = resized_masks

        # select best mask and build binary mask to return
        best_mask = self._select_mask(output_masks, output_iou_predictions)
        binary_mask = (best_mask > 0).astype(np.uint8)
        assert binary_mask.shape == (image.height, image.width), (
            f"Mask shape {binary_mask.shape} != image size ({image.height}, {image.width})"
        )
        return IPromptedImageSegmentationModel.Prediction(mask=binary_mask)
