import onnx
from onnx import helper, TensorProto

input_file = "C:\\Users\\hualxie\\repos\\olive-recipes\\openai-whisper-large-v3-turbo\\olive-ort-genai\\models\\whisper-large-v3-turbo_decoder_fp16_old.onnx"
output_file = "C:\\Users\\hualxie\\repos\\olive-recipes\\openai-whisper-large-v3-turbo\\olive-ort-genai\\models\\whisper-large-v3-turbo_decoder_fp16.onnx"

model = onnx.load(input_file, load_external_data=False)

def transpose_input(model, input_name, output_name, perm, input_shape):
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[output_name],
        perm=perm
    )
    model.graph.node.append(transpose_node)
    model.graph.input.append(helper.make_tensor_value_info(
        input_name, TensorProto.FLOAT, input_shape))
    model.graph.input.remove(next(inp for inp in model.graph.input if inp.name == output_name))

def transpose_output(model, input_name, output_name, perm, output_shape):
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[output_name],
        perm=perm
    )
    model.graph.node.append(transpose_node)
    model.graph.output.append(helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, output_shape))
    model.graph.output.remove(next(inp for inp in model.graph.output if inp.name == input_name))

for i in range(4):
    # inputs
    input_name = f'past_key_self_{i}'
    output_name = f'k_cache_self_{i}_in'
    transpose_input(model, input_name, output_name, [1, 0, 3, 2], [1, 20, 199, 64])

    input_name = f'past_value_self_{i}'
    output_name = f'v_cache_self_{i}_in'
    transpose_input(model, input_name, output_name, [1, 0, 2, 3], [1, 20, 199, 64])

    input_name = f'past_key_cross_{i}'
    output_name = f'k_cache_cross_{i}'
    transpose_input(model, input_name, output_name, [1, 0, 3, 2], [1, 20, 1500, 64])

    input_name = f'past_value_cross_{i}'
    output_name = f'v_cache_cross_{i}'
    transpose_input(model, input_name, output_name, [1, 0, 2, 3], [1, 20, 1500, 64])
    # outputs
    input_name = f'k_cache_self_{i}_out'
    output_name = f'present_key_self_{i}'
    transpose_output(model, input_name, output_name, [1, 0, 3, 2], [1, 20, 199, 64])

    input_name = f'v_cache_self_{i}_out'
    output_name = f'present_value_self_{i}'
    transpose_output(model, input_name, output_name, [1, 0, 2, 3], [1, 20, 199, 64])

# Create the target shape as a constant
target_shape = helper.make_tensor(
    name='logits_reshape_shape',
    data_type=TensorProto.INT64,
    dims=[3],
    vals=[1, 1, 51866]
)

# Create a Constant node for the shape
shape_node = helper.make_node(
    'Constant',
    inputs=[],
    outputs=['logits_reshape_shape'],
    value=target_shape
)

# Create the Reshape node
reshape_node = helper.make_node(
    'Reshape',
    inputs=['logits', 'logits_reshape_shape'],  # tensor to reshape + target shape
    outputs=['logits_reshaped']
)

# Add nodes to the model
model.graph.node.append(shape_node)
model.graph.node.append(reshape_node)

model.graph.output.append(helper.make_tensor_value_info(
    'logits_reshaped', 
    TensorProto.FLOAT,  # Data type
    [1, 1, 51866]   # Shape
))
model.graph.output.remove(next(inp for inp in model.graph.output if inp.name == 'logits'))


onnx.save(model, output_file)
