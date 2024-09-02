# Problem

You have just joined a robotics startup focused on developing autonomous delivery robots designed to operate in complex urban environments. One of your first tasks is to build a machine learning model that can classify and predict the types of obstacles the robot might encounter on the sidewalk. The robot is equipped with multiple sensors that collect various data inputs such as distance measurements, speed, surface texture, and environmental noise levels.

Given $d$ inputs (e.g., sensor readings for distance, speed, texture, etc.), your model must identify, at regular intervals, which of the $c$ possible obstacles (e.g., pedestrian, bicycle, trash can, etc.) are present in the robot's path. The classification model will be a multi-label binary classifier since multiple obstacles could appear simultaneously.

However, the startup's custom-built robots use a proprietary software stack that isn't compatible with standard machine learning libraries like PyTorch or TensorFlow. As a result, you must implement both the forward and backward pass in numpy to ensure the model can be deployed directly onto the robot's onboard system.

You decide to implement a simple 3-layer feedforward neural network with the following architecture:

```latex
$input\ (\mathbb{R}^{d}) 
\to \mathrm{\mathbf{linear}} 
\to \mathrm{\mathbf{ScaledReLU}} 
\to tmp\ (\mathbb{R}^{h}) 
\to \mathrm{\mathbf{linear}} 
\to \mathrm{\mathbf{ScaledReLU}} 
\to tmp\ (\mathbb{R}^{h}) 
\to \mathrm{\mathbf{linear}}  
\to logits\ (\mathbb{R}^{c})
\to \mathrm{\mathbf{Sigmoid}}
\to prediction \ (\mathbb{R}^{c})
$
```

The loss function to be used for training this network is the mean negative log-likelihood. Below is the final structure of the MLP class. Your job is to fill in the definitions of the functions called by the MLP class, ensuring they are compatible with the robot's custom software environment.

```python
class MLP:
    def __init__(self, d, h, c):
        self.layers = [
            Linear(d, h), ScaledReLU(h),  # input_layer
            Linear(h, h), ScaledReLU(h),  # hidden_layer
            Linear(h, c), Sigmoid(c)  # output_layer
        ]
        
    def forward(self, X):
        """Predict classes for a batch of n examples"""
        out = X  # $\in \mathbb{R}^{n \times d}$
        for layer in self.layers:
            out = layer.forward(out) 
        return out
    
    def sgd(self, X_train, Y_train):
        # Train each layer of the MLP with SGD
        n_train = len(X_train)
        epochs = 10
        bs = 100  # batch_size
        for _ in range(epochs):
            for b_idx in range(0, n_train, bs):
                X_batch = X_train[b_idx:b_idx + bs]
                Y_batch = Y_train[b_idx:b_idx + bs]
                # update all parameters
                self.single_step_sgd(X_batch, Y_batch)

    def single_step_sgd(self, X, Y):  # $X\in \mathbb{R}^{n\times d}$, $Y\in \mathbb{R}^{n\times c}$
        """Train the MLP with a single step of batched SGD."""
        Y_hat = self.forward(X)  # $\in \mathbb{R}^{n \times c}$
        loss_nabla_O = get_grad_log_loss(Y, Y_hat)
        for layer in self.layers[::-1]:  # backward pass reverses layers
            loss_nabla_O, *param_grads = layer.backward(loss_nabla_O)
            layer.update_params_(*param_grads, lr=0.1)
```

In the following sections, you should write the `forward`, `backward`, `update_params_`, and `__init__` methods for the `Linear`, `ScaledReLU`, and `Sigmoid` classes, as well as the `get_grad_log_loss` function. Note that `ScaledReLU` is similar to ReLU, but it scales the output by 2 if the input is greater than 1 and 0 otherwise. For the linear layer, use Kaiming initialization for the weights and zero initialization for the bias. Use activation checkpointing to save memory wherever possible. Implement everything using NumPy. Your code should be fully functional.

Considering only matrix multiplications in your code, approximately how much slower (in terms of FLOPs) would it be to train our MLP on $n = 10,000$ examples versus performing inference on $n = 10,000$ examples? Your answer should be an integer (round it if you get a float). Provide your derivation.

# Potential Solution

```python
import numpy as np
class Sigmoid:
    def __init__(self, c):
        self.c = c
        
    def forward(self, X):
        O = 1 / (1 + np.exp(-X))
        self.O = O
        return O

    def backward(self, loss_nabla_O):      
        loss_nabla_X = loss_nabla_O * self.O * (1-self.O)
        return loss_nabla_X
    
    def update_params_(self, lr):
        pass # Sigmoid has no parameters

class ScaledReLU:
    def __init__(self, h):
        self.h = h
        
    def forward(self, X):
        self.mask = (X > 1).astype(X.dtype) * 2
        return self.mask * X 

    def backward(self, loss_nabla_O):
        loss_nabla_X = loss_nabla_O * self.mask
        return loss_nabla_X

    def update_params_(self, lr):
        pass # ReLU has no parameters

class Linear:
    def __init__(self, d, h):
        self.W = np.random.randn(d, h) * np.sqrt(2/d)
        self.b = np.zeros(h)
        
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, loss_nabla_O):
        loss_nabla_W = self.X.T @ loss_nabla_O
        loss_nabla_b = loss_nabla_O.sum(axis=0)
        loss_nabla_X = loss_nabla_O @ self.W.T
        return loss_nabla_X, loss_nabla_W, loss_nabla_b

    def update_params_(self, loss_nabla_W, loss_nabla_b, lr):
        self.W -= lr * loss_nabla_W
        self.b -= lr * loss_nabla_b

def get_grad_log_loss(Y, Y_hat, eps=1e-8):
    return 1/len(Y) * (Y_hat-Y)/((Y_hat * (1-Y_hat))+ eps)
```

Complexity: only linear layer has matrix multiplications so we should only consider that. As a reminder, the FLOPs for multiplication of two matrices of shape $n \times o$ and $i \times o$ is approximately $2nio$ because there are $no$ dot products, each for vectors of shape $i$ and dot product two vectors of dimensionality $d$ is approx $2d$ (one $d$ for mult, one $d-$ for add). The forward for the linear thus requires $2nio$ FLOPs. The backward requires $4nio$ FLOPs: $2nio$ for the gradients w.r.t. inputs $2nio$ for gradients w.r.t. weights. So $4nio$. Training requires both the backward and the forward pass so $6nio$ in total. Inference only requires the forward pass so $2nio$. The ratio is $\frac{6nio}{2nio}=3$ so training requires 3x more compute than inference. This is independent of $n$ (or rather large $n$ is equivalent to only considering matrix multiplications).
