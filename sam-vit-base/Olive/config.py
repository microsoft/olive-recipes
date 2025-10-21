class config:
    model_name = "facebook/sam-vit-base"
    data_dir = "quantization_dataset"
    ve_input_name = "pixel_values"
    ve_sample_size = 1024
    ve_channel_size = 3
    mask_point_input_names = ['input_points', 'image_embeddings']
    mask_point_input_shapes = [(1,1,2), (256, 64, 64)]
    
    mask_box_input_names = ['input_boxes', 'image_embeddings']
    mask_box_input_shapes = [(1,4), (256, 64, 64)]