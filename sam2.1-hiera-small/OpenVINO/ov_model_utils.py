import torch

from sam2.sam2_image_predictor import SAM2ImagePredictor


class SamImageEncoderModel(torch.nn.Module):
    _INPUT_IMAGE_SIZE = 1024


    def __init__(self, predictor) -> None:
        super().__init__()
        self.image_encoder = predictor.model.image_encoder
        self.base_model = predictor.model
        self._bb_feat_sizes = predictor._bb_feat_sizes

        # precompute positional encodings for the image encoder input size
        trunk = self.image_encoder.trunk
        patch_size = 4
        spatial_size = self._INPUT_IMAGE_SIZE // patch_size

        with torch.no_grad():
            pos_embed_interp = torch.nn.functional.interpolate(
                trunk.pos_embed.data,
                size=(spatial_size, spatial_size),
                mode="bicubic"
            )
            window_embed = trunk.pos_embed_window.data
            tiled_window = window_embed.tile(
                [x // y for x, y in zip(pos_embed_interp.shape, window_embed.shape)]
            )
            self._precomp_pos_embed = (pos_embed_interp + tiled_window).permute(0, 2, 3, 1)

        del trunk._parameters["pos_embed"]
        del trunk._parameters["pos_embed_window"]

        # override trunk._get_pos_embed with precomputed embed
        trunk._get_pos_embed = lambda x: self._precomp_pos_embed

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
    ):
        backbone_out = self.base_model.forward_image(image)

        _, vision_feats, _, _ = self.base_model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.base_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.base_model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        return feats[-1], feats[0], feats[1]


class SamImageMaskPredictionModel(torch.nn.Module):
    def __init__(
        self, model
    ) -> None:
        super().__init__()
        self.mask_decoder = model.sam_mask_decoder
        self.mask_decoder.use_high_res_features = True
        self.model = model
        self.model.use_high_res_features_in_sam = True
        self.img_size = model.image_size
        self.prompt_encoder = model.sam_prompt_encoder

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1).to(torch.float32)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (point_labels == -1).to(torch.float32)

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i).to(torch.float32)

        return point_embedding

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        high_res_feats_256: torch.Tensor,
        high_res_feats_128: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)

        low_res_masks, iou_pred, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=[high_res_feats_256, high_res_feats_128],
        )

        upscaled_masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        return upscaled_masks, iou_pred, low_res_masks


def load_sam21_image_encoder(model_path: str) -> SamImageEncoderModel:
    predictor = SAM2ImagePredictor.from_pretrained(model_path, device="cpu")
    m = SamImageEncoderModel(predictor)
    m.eval()
    return m


def load_sam21_mask_decoder(model_path: str) -> SamImageMaskPredictionModel:
    predictor = SAM2ImagePredictor.from_pretrained(model_path, device="cpu")
    m = SamImageMaskPredictionModel(predictor.model)
    m.eval()
    return m


def example_sam21_image_encoder_input() -> torch.Tensor:
    return torch.randn(1, 3, 1024, 1024)


def example_sam21_mask_decoder_input() -> tuple[torch.Tensor, ...]:
    image_embedding = torch.randn(1, 256, 64, 64)
    high_res_feats_256 = torch.randn(1, 32, 256, 256)
    high_res_feats_128 = torch.randn(1, 64, 128, 128)
    point_coords = torch.rand(1, 5, 2) * 1024
    point_labels = torch.ones(1, 5)

    return image_embedding, high_res_feats_256, high_res_feats_128, point_coords, point_labels
