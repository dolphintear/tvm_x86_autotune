# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm, relay
import tvm
from tvm import relay

# os and numpy
import numpy as np
import os.path
import time
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow imports
import tensorflow as tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

# Test image
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)

######################################################################
# Tutorials
# ---------
# Please refer docs/frontend/tensorflow.md for more details for various models
# from tensorflow.

model_name = '/home/dolphin/Downloads/frozen_inference_graph_fuse_bn_640x360.pb'
model_url = os.path.join(repo_base, model_name)


# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
layout = "NCHW"
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
from tvm.contrib.download import download_testdata

model_path = '/home/dolphin/Downloads/frozen_inference_graph_fuse_bn_640x360.pb'
img_path = download_testdata(image_url, img_name, module='data')

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.

with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    #with tf.Session() as sess:
        #graph_def = tf_testing.AddShapesToGraphDef(sess, 'softmax')

######################################################################
# Decode image
# ------------
# .. note::
#
#   tensorflow frontend import doesn't support preprocessing ops like JpegDecode.
#   JpegDecode is bypassed (just return source node).
#   Hence we supply decoded frame to TVM instead.
#

from PIL import Image
#image = Image.open(img_path).resize((299, 299))
image = Image.open(img_path).resize((256, 256))

x = np.array(image)

######################################################################
# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
input_shape = (1, 256, 256,3)
output_shape = (1, 256,256)
dtype = 'float32'
#sym, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

print("Tensorflow protobuf imported to relay frontend.")
"""
Auto-tuning a convolutional network for x86 CPU

This is a tutorial about how to tune convolution neural network
for x86 CPU.
"""
import numpy as np

from tvm import autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".

model_name = 'mmnet_1'
log_file = "%s.log" % model_name

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)


#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.

tuning_option = {
    'log_filename': log_file,
    'tuner': 'random',
    'early_stopping': 2,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=1, repeat=1,
                                   min_repeat_ms=10),
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        print('op_name:%', op_name)
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

#tune_and_evaluate(tuning_opt):
# extract workloads from relay program
print("Extract tasks...")
net, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape={'preprocess/truediv': input_shape})
tasks = autotvm.task.extract_from_program(net, target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

# run tuning tasks
print("Tuning...")
tune_kernels(tasks, **tuning_option)

# compile kernels with history best records
with autotvm.apply_history_best(log_file):
    print("Compile...")
    with relay.build_config(opt_level=3):
        #graph, lib, params = relay.build(net, target=target, target_host=target_host, params=params)
        #there is difference between the two build  ???
        graph, lib, params = relay.build_module.build(
            net, target=target, params=params)

    # upload parameters to device
    ctx = tvm.cpu()
    #data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    dtype = 'uint8'
    module = runtime.create(graph, lib, ctx)
    for i in range(10):
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('preprocess/truediv', data_tvm)
        module.set_input(**params)
        t0 = time.time()
        module.run()
        print('%: %',(i, time.time()-t0))
    tvm_output = module.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))
    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))




