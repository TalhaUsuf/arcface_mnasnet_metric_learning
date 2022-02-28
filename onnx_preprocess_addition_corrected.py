import onnx
import sclblonnx as so
from rich.console import Console
from onnx import helper
import numpy as np



# ==========================================================================
#                             onnx related                                  
# ==========================================================================

model = so.graph_from_file("SAVED_MODEL_TF/mnasnet_embedder.onnx")

# g = so.clean(model)
# so.check(g)





# ==========================================================================
#                        mean=tensor([0.4850, 0.4560, 0.4060]), 
#                        std=tensor([0.2290, 0.2240, 0.2250])                                  
# ==========================================================================



# # --------------------------------------------------------------------------
# #                              making tensors                        
# # --------------------------------------------------------------------------


m = np.array([[[[0.4850]],[[0.4560]],[[0.4060]]]])
s = np.array([[[[4.36681223]],[[4.46428571]],[[4.44444444]]]])

Console().log(f"mean ---> {m.shape}")
Console().log(f"std ---> {s.shape}")

mean = helper.make_tensor('mean', data_type=onnx.TensorProto.FLOAT,dims=m.shape,vals=m.flatten())
std = helper.make_tensor('std', data_type=onnx.TensorProto.FLOAT,dims=s.shape,vals=s.flatten())

Console().log(f"mean ---> {mean}")
Console().log(f"std ---> {std}")


# Console().log(model.initializer)

model.initializer.append(mean)
model.initializer.append(std)


# --------------------------------------------------------------------------
#                          making the empty nodes
#           open model in netron app and see the names of the nodes                        
# --------------------------------------------------------------------------

sub_node = helper.make_node('Sub', inputs=['input', 'mean'], outputs=['sub_out'])
# reci_node = helper.make_node('Mul', inputs=['std'], outputs=['denominator'])
norm_node = helper.make_node('Mul', inputs=['sub_out', 'std'], outputs=['normalized_out'])


# print the very ist node (without the input , output)
Console().log(model.node[0])
# get the inputs going into conv-0 node
Console().log(model.node[0].input)




model.node[0].input.insert(0, 'normalized_out')
model.node[0].input.remove('input')


model.node.insert(0, sub_node)
model.node.insert(1, norm_node)


so.graph_to_file(model, 'SAVED_MODEL_TF/mnasnet_embedder_pre_processed.onnx')

onnx_model = onnx.load_model("SAVED_MODEL_TF/mnasnet_embedder_pre_processed.onnx")
onnx.checker.check_model(onnx_model)

import onnxoptimizer

passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)

onnx.save(optimized_model, "SAVED_MODEL_TF/mnasnet_embedder_pre_processed_OPTIMIZED.onnx")

# # c1 = helper.make_node('Constant', inputs=[], outputs=['c1'], name="c1-node",
# #                       value=helper.make_tensor(name="c1v", data_type=onnx.TensorProto.FLOAT, dims=m.shape, vals=m.flatten()))
# #
# # c2 = helper.make_node('Constant', inputs=[], outputs=['c2'], name="c2-node",
# #                        value=helper.make_tensor(name="c2v", data_type=onnx.TensorProto.FLOAT,dims=s.shape,vals=s.flatten()))
# #
# #
# # n1 = helper.make_node('Sub', inputs=['x', 'c1'], outputs=['xmin'], name='n1')
# # n2 = helper.make_node('Div', inputs=['xmin', 'c2'], outputs=['zx'], name="n2")
# #
# #
# # g1 = helper.make_graph([c1, n1, c2, n2], 'preprocessing',
# #  [helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [3])],
# #  [helper.make_tensor_value_info('zx', onnx.TensorProto.FLOAT, [3])])
# #
# #
# #
# #
# #
# # # Create the model and check
# # m1 = helper.make_model(g1, producer_name='scailable-demo')
# # onnx.checker.check_model(m1)
# # # Save the model
# # onnx.save(m1, '/home/stagingserver/workspace/Data_Sets/scripts/pre-processing.onnx')