---
layout: post
title: Python Notes
categories: [Computer Science]
tags: [Python]
---

* [Time Decorator](#time-decorator)

<!--excerpt-->

### Time Decorator
```
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print 'func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts)
        return result
    return wrap
 ```
 Usage
 ```
 @timing
def f(a):
    for _ in range(a):
        i = 0
    return -1
```
