import onnx
from onnx import helper, TensorProto, shape_inference
import numpy as np
from onnx import numpy_helper
from pathlib import Path

this_folder = Path(__file__).parent

input_file =  str(this_folder / 'models' / 'encoder_old.onnx')
output_file = str(this_folder / 'models' / 'encoder.onnx')

model = onnx.load(input_file, load_external_data=False)

# # Run shape inference to populate value_info
# inferred_model = shape_inference.infer_shapes(model)

# # Now check any tensor (input, output, or intermediate)
# for value_info in inferred_model.graph.value_info:
#     if "/encoder/layer_norm/LayerNormalization_output_0" in value_info.name:  # Search by partial name
#         shape = [dim.dim_value or dim.dim_param or "?" 
#                  for dim in value_info.type.tensor_type.shape.dim]
#         print(f"{value_info.name}: {shape}")
# exit(0)

for i in range(4):
    input_name = f'k_cache_cross_{i}'
    output_name = f'present_key_cross_{i}'
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[output_name],
        perm=[1, 0, 3, 2]  # Reorder dimensions: [20,1,64,1500] -> [1,20,1500,64]
    )
    model.graph.node.append(transpose_node)
    model.graph.output.append(helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, 20, 1500, 64]))
    model.graph.output.remove(next(inp for inp in model.graph.output if inp.name == input_name))

    input_name = f'v_cache_cross_{i}'
    output_name = f'present_value_cross_{i}'
    transpose_node = helper.make_node(
        'Transpose',
        inputs=[input_name],
        outputs=[output_name],
        perm=[1, 0, 2, 3]  # Reorder dimensions: [20,1,1500,64] -> [1,20,1500,64]
    )
    model.graph.node.append(transpose_node)
    model.graph.output.append(helper.make_tensor_value_info(
        output_name, TensorProto.FLOAT, [1, 20, 1500, 64]))
    model.graph.output.remove(next(inp for inp in model.graph.output if inp.name == input_name))

# Create an Identity node to rename the input
identity_node = helper.make_node(
    'Identity',
    inputs=['audio_features'],  # The existing node output
    outputs=['input_features']        # The new name you want
)

# Add it to your model
model.graph.node.append(identity_node)

# Add the new output to the graph outputs
model.graph.input.append(helper.make_tensor_value_info(
    'audio_features', 
    TensorProto.FLOAT,  # Data type
    [1, 128, 3000]   # Shape
))
model.graph.input.remove(next(inp for inp in model.graph.input if inp.name == 'input_features'))


# Add encoder_hidden_states
# Add shape as initializer
shape_initializer = numpy_helper.from_array(
    np.array([1, 1500, 1280], dtype=np.int64),
    name='encoder_hidden_states_shape'
)
model.graph.initializer.append(shape_initializer)

# Create Reshape node
reshape_node = helper.make_node(
    'Reshape',
    inputs=['/encoder/layer_norm/LayerNormalization_output_0', 'encoder_hidden_states_shape'],
    outputs=['encoder_hidden_states']
)

model.graph.node.append(reshape_node)

model.graph.output.append(helper.make_tensor_value_info(
    'encoder_hidden_states', 
    TensorProto.FLOAT,
    [1, 1500, 1280]
))
onnx.save(model, output_file)
