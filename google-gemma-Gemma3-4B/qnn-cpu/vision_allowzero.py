import os
import onnx

mp = r'models/gemma-3-4b-it-vis-onnx/model.onnx'
op = r'models/gemma-3-4b-it-vis-onnx/model_updated.onnx'

m = onnx.load(mp)
for node in m.graph.node:
    del node.metadata_props[:]
    if node.op_type == 'Reshape':
        for attr in node.attribute:
            if attr.name == 'allowzero':
                attr.i = 0
onnx.save(m, op)

