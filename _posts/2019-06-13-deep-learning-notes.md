## Training Tricks
* [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) 
* [Early Stopping Mechanism](https://www.datalearner.com/blog/1051537860479157)
* [How to Tame the Valley](https://medium.com/autonomous-agents/how-to-tame-the-valley-hessian-free-hacks-for-optimizing-large-neuralnetworks-5044c50f4b55)
* [Plateau at the Beginning of Training](#plateau-at-the-beginning-of-training)
<!--excerpt-->

### Plateau at the Beginning of Training
  This is often due to bad configuration of hyper parameters or bad initialization of weights:
  * Case I: tuning the learning rate. Large learning rate often cause optimization unstable. Small learning rate may result in learning procedure get stuck.
  * Case II: if use default random uniform initializier for word embedding, it need at least 20 epochs for traning loss to decrease. Yet, using a truncated uniformed initializer with a std of 1e-2, the loss can decrease right at the beginning.
