---
layout: post
title: TensorFlow Notes
categories: [Deep Learning]
tags: [TensorFlow]
---

* [VarHandleOp](#varhandleop)
* [Get tensor shape as list](#get-tensor-shape-as-list)
* [Add Regularization Loss](#add-regularization-loss)
* [Turn Off Verbose](#turn-off-verbose)
* [Gradients Clipping](#gradients-clipping)
<!--excerpt-->

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
