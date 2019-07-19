---
layout: post
title: TensorFlow Notes
categories: [Deep Learning]
tags: [TensorFlow]
---
## Docs
* [VarHandleOp](#varhandleop)
* [Get tensor shape as list](#get-tensor-shape-as-list)
* [Add Regularization Loss](#add-regularization-loss)
* [Turn Off Verbose](#turn-off-verbose)
* [Gradients Clipping](#gradients-clipping)
* [Using TensorRT in TF](#using-tensorrt-in-tf)
* [Freeze a TF Graph](freeze-a-tf-graph)

## Bugs
* [Freeze Graph](#freeze-graph)
<!--excerpt-->

## Docs

### VarHandleOp
As seen in [freeze_saved_model.cc](https://github.com/tensorflow/tensorflow/blob/bd13eb08e410787e28e7c5cd0153fad28e3cf9f1/tensorflow/cc/tools/freeze_saved_model.cc),
we need to get its tensor by the similar behavior as Identity Op.
```
  if (node_def->op() == "VarHandleOp") {
    // If this is a resource variable, we have to run the corresponding
    // ReadVariableOp.
    tensor_names.push_back(node_name + "/Read/ReadVariableOp:0");
  } else {
    tensor_names.push_back(node_name + ":0");
  }
```

### Get tensor shape as list
```
  tensor.get_shape().as_list()
```

### Add Regularization Loss
Sometimes, using var_scope can not add regularizers to weights. To check it, use
```
  tf.losses.get_regularization_loss()
```
To manually add regularizers, define
```
  l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in in tf.trainable_variables(): ]) * regularization_weights
``` 

### Turn Off Verbose
```
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
  import logging
  logging.getLogger("tensorflow").setLevel(logging.ERROR)
```

### Gradients Clipping
```
  tvars = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.config.max_grad_norm)
  optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
  train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
  return train_op
```
According to TensorFlow's doc, the idea is to collect graidents of all trainable tensors. If its l2 norm is bigger than clip norm, clip it by a ratio of clip_norm/global.
```
  To perform the clipping, the values `t_list[i]` are set to:

      t_list[i] * clip_norm / max(global_norm, clip_norm)

  where:

      global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

  If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
  otherwise they're all shrunk by the global ratio.
```

### Using TensorRT in TF
```
  with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph("/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600.meta")

            # Then restore your training data from checkpoint files:
            saver.restore(sess, "/workspace/exps/ptb/ptb_rnn_128/ptb_rnn_128-72600")

            # Finally, freeze the graph:
            your_outputs = ['model/dense/BiasAdd']
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                sess,
                tf.get_default_graph().as_graph_def(),
                output_node_names=your_outputs)
            trt_graph = trt.create_inference_graph(
                input_graph_def=frozen_graph,
                outputs=your_outputs,
                max_batch_size=10,
                max_workspace_size_bytes=2 << 20,
                precision_mode='fp16')
            tf.train.write_graph(trt_graph, "/workspace/exps/ptb/ptb_rnn_128", "trt_model.pb", as_text=False)
    print('done')
```
* [Nvidia Docs](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#benefits)
* [TF Example](https://github.com/tensorflow/tensorrt/blob/master/tftrt/examples/image-classification/image_classification.py)

### Freeze a TF Graph
* [Python API](https://zhuanlan.zhihu.com/p/64099452)
* Also, it can be done manually
```
frozen_graph = tf.graph_util.convert_variables_to_constants(
                tf_sess,
                tf_sess.graph_def,
                output_node_names=['logits', 'classes']
            )
```

## Bugs

### Freeze Graph
This is due to the **tf.nn.rnn_cell.DropoutWrapper** in model definition. I suppose this method is not supported by freeze graph. Yet, freeze graph does not provide relevant traceback for it.
```
Traceback (most recent call last):
  File "/usr/lib/python3.5/runpy.py", line 184, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/lib/python3.5/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/workspace/code/rnnquant/mains/freeze_graph.py", line 35, in <module>
    initializer_nodes=''
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py", line 363, in freeze_graph
    checkpoint_version=checkpoint_version)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py", line 190, in freeze_graph_with_def_protos
    var_list=var_list, write_version=checkpoint_version)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saver.py", line 832, in __init__
    self.build()
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saver.py", line 844, in build
    self._build(self._filename, build_save=True, build_restore=True)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saver.py", line 881, in _build
    build_save=build_save, build_restore=build_restore)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saver.py", line 487, in _build_internal
    names_to_saveables)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saving/saveable_object_util.py", line 338, in validate_and_slice_inputs
    for converted_saveable_object in saveable_objects_for_op(op, name):
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saving/saveable_object_util.py", line 207, in saveable_objects_for_op
    variable, "", name)
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/training/saving/saveable_object_util.py", line 83, in __init__
    self.handle_op = var.op.inputs[0]
  File "/usr/local/lib/python3.5/dist-packages/tensorflow/python/framework/ops.py", line 2195, in __getitem__
    return self._inputs[i]
IndexError: list index out of range

```
