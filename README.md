# Deep_Dream


## Theory
<p align="center">
<img src="https://github.com/JitheshPavan/deep_dream/blob/main/data/modified%20images/lion_output.png" alt="lion_output" width="400" height="300">
</p>
An AI Model is ultimately characterized by its structure: the layers, the operations, and the number of parameters. Model is what is unchanging throughout training. Even most of the hyperparameters are not really a characteristic of the model. The parameter does not have an inherent relation to the model, for parameters can be any number. In this perspective, the input and the parameters are not different. They are both numbers. They both can change. The model cannot differentiate between input and parameters. We impose categorization among them by only changing the parameters. For example, take an intermediate layer; during backpropagation, the layer receives a gradient, which is then used to calculate the gradient w.r.t to the input and the parameters of that layer. The gradient calculated w.r.t to the input is passed to the layer before it. This fact is true for the first layer as well. The input backpropagated in the first layer is the gradient w.r.t to the input. Input may well be the output from an unknown layer. But these gradients are ignored. We change the parameters so that the loss function may decrease w.r.t to the input. But these input gradients are real, and we can change the input to fit the parameters. The deep dream algorithm is built on this idea. What if we alter the image w.r.t to the parameters? That is, we alter the image so that the output of a layer is maximized. We aim to maximize the input to that layer. What does this mean? The image is altered to fit what the layer searches in it. 

## Mantis
<p align="center">
<img src="https://github.com/JitheshPavan/deep_dream/blob/main/data/mantis.jpeg">
<img src="https://github.com/JitheshPavan/deep_dream/blob/main/data/modified%20images/mantis_lr_0.001_iter_5_pyramratio_1.5_nop_4.png">
</p>
                
What I noticed during deep dreams, especially during w.r.t the initial layers, is a differentiation of colors. Red, Blue, and Green. At every group of pixels, a specific primary color is maximized. This is why the pattern is psychedelic. The fundamental colors (3 arrays) are being separated. At a specific pixel, a color is maximized (It is gradient ascent, after all). The more activated a pixel is, the more it is affected.

## How gradients work in PyTorch
There are certain types of tensors known as leaf tensors. There are user-created tensors. These tensors have a special method known as requires_grad. If set to true, as the name suggests, it will get the ability to create computational graphs. Computational graphs keep track of operations. With the help of these graphs, PyTorch can calculate gradients during backpropagation. Any subsequent tensors created from these tensors will also have required_grad set as true. This is necessary to make a computational graph and keep track of the gradients. But these will not be leaf tensors. So, any tensors created from leaf tensors are not leaf tensors unless required_grad is set to false.


The computational graph is used to flow gradients when we call .backward(). With the help of a computational graph, backpropagation will be performed. Gradients will flow backward through the operations as specified by the computational graph. These grads will flow till we reach our initial tensor (leaf tensor). Gradients are not retained if the tensor is not leaf. Thus, only initial tensors will have gradients. That is why they are called leaves. They are the beginning of the graph. So leaf tensors can be thought of as user-created tensors, are the starting points, and capable of storing gradients. 
   
However, there are cases when we need to keep the gradients of intermediary tensors. This is possible with .retain_grad() (A method unlike requires_grad, which is a boolean). To keep the gradients, a tensor has to be leaf ( requires_grad is presupposed since the computational graph is not formed without it), or the retain_grad() method has to be activated beforehand. It is useless to call retain_grad on leaf tensors since they are already required to preserve grad. 

 **Example**
 
 import torch
 
 l= torch.rand(2,2,3)
 
 k= l +2
 
 k.requires_grad=True
 
 y=k **2 
 
 loss= y +1
 
 print(l.is_leaf,k.is_leaf,y.is_leaf, y.requires_grad) # True, True, False , True
 
 loss.sum().backward()
 
 print(y.grad, k.grad, l.grad) # None, Values, None with user warning. The computational graph began at k. 

 ## TODO
Currently rewriting the script in C++ for improved computational speed
