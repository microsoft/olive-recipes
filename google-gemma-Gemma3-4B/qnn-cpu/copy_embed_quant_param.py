import onnx
from onnx import OperatorSetIdProto

QUANT_MODEL = r"models/gemma3_qnn/model/embeddings.onnx"
IMAGE_MODEL = r"models/gemma-3-4b-it-embed/model/model.onnx"
# TARGET_NODE = r"/embedding_layer/Gather"
TARGET_NODE = r"node_embedding"
SOURCE_NODE = r"/model/embed_tokens/Gather_Q4"

OUTPUT_MODEL = r"models/gemma3_qnn/model/embed_quant.onnx"

def find_node_by_name(m, node_name):
    for idx, node in enumerate(m.graph.node):
        if node.name == node_name:
            return (idx, node)
    assert(f"node {node_name} not found in graph" == None)

def find_initializer_by_name(m, init_name):
    for idx, init in enumerate(m.graph.initializer):
        if init.name == init_name:
            return (idx, init)
    assert(f"initializer {init_name} not found in graph" == None)

im = onnx.load(IMAGE_MODEL)
qm = onnx.load(QUANT_MODEL)

s_idx, s_node = find_node_by_name(qm, SOURCE_NODE)
i_idx, i_node = find_node_by_name(im, TARGET_NODE)

assert(s_node.op_type == 'GatherBlockQuantized')
assert(i_node.op_type == 'Gather')

w_idx, w_init = find_initializer_by_name(im, i_node.input[0]) # Gather weight
wq_idx, wq_init = find_initializer_by_name(qm, s_node.input[0]) # Gather Q4 weight
ws_idx, ws_init = find_initializer_by_name(qm, s_node.input[2]) # Gather Q4 weight

# Modify graph
new_import = OperatorSetIdProto()
new_import.domain = s_node.domain
new_import.version = 1
im.opset_import.extend([new_import])

gather_out = i_node.output[0]
del im.graph.node[i_idx]
im.graph.node.insert(i_idx, qm.graph.node[s_idx])
im.graph.node[i_idx].output[0] = gather_out

del im.graph.initializer[w_idx]
im.graph.initializer.append(wq_init)
im.graph.initializer.append(ws_init)

onnx.save(im, OUTPUT_MODEL, save_as_external_data = True, all_tensors_to_one_file = True, location = OUTPUT_MODEL.split('/')[-1] + '.data')
