import os
import onnx

ctx_path = r'models/gemma3_qnn/model/context.onnx'
itr_path = r'models/gemma3_qnn/model/iterator.onnx'

cm = onnx.load(ctx_path, load_external_data=False)
im = onnx.load(itr_path, load_external_data=False)

for ci, ii in zip(cm.graph.initializer, im.graph.initializer):
    if ci.name != ii.name:
        print(f'initializer are not same {ci.name} <=> {ii.name}')
        break
    c_loc_idx = None
    i_loc_idx = None
    if ci.external_data[0].key != 'location' or ii.external_data[0].key != 'location':
        print(f'unexpeted mismatch in external data')
        continue
    if (ci.external_data[1] == ii.external_data[1] and ci.external_data[2] == ci.external_data[2]):
        ii.external_data[0].value = ci.external_data[0].value

onnx.save(im, itr_path)
                

