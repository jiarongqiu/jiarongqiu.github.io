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
