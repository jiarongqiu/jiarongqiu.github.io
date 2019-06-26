---
layout: post
title: TensorFlow Notes
categories: [Deep Learning]
tags: [TensorFlow]
---

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
```

### Avoid TF Verbose Info
```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
```
