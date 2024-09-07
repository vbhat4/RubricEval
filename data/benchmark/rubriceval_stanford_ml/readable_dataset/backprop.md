# <category>:
Stats & ML
# <instruction>:
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
# <expert_solution>:
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
# <expert_checklist>:
[
  "Does the implementation of the Linear class include Kaiming initialization for weights and zero initialization for biases?",
  "Is the ScaledReLU activation function correctly implemented to scale the output by 2 if the input is greater than 1 and 0 otherwise?",
  "Are the forward and backward methods for each layer (Linear, ScaledReLU, Sigmoid) correctly implemented using NumPy?",
  "Does the get_grad_log_loss function correctly compute the gradient of the mean negative log-likelihood loss?",
  "Is activation checkpointing used effectively to save memory during the forward pass?",
  "Are the update_params_ methods for each layer correctly updating the parameters using the gradients and learning rate?",
  "Is the forward pass of the MLP class correctly chaining the forward methods of each layer?",
  "Does the backward pass of the MLP class correctly reverse the layers and compute gradients for each layer?",
  "Is the SGD method correctly implementing mini-batch stochastic gradient descent over multiple epochs?",
  "Is the calculation of FLOPs for training versus inference correctly derived and explained?"
]
# <expert_checklist_time_sec>:

# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect implementation of Linear layer",
    "description": "The Linear layer should use Kaiming initialization for weights and zero initialization for biases. Check if the weights are initialized with `np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)` and biases with `np.zeros(output_dim)`. Incorrect initialization can lead to poor model performance.",
    "delta_score": -1.5
  },
  {
    "error_name": "Incorrect ScaledReLU function",
    "description": "The ScaledReLU should output `X * 2` if `X > 1` and `0` otherwise. Verify that the implementation uses `np.where(X > 1, X * 2, 0)`. Incorrect activation function can lead to incorrect model behavior.",
    "delta_score": -1
  },
  {
    "error_name": "Sigmoid function not implemented correctly",
    "description": "The Sigmoid function should compute `1 / (1 + np.exp(-X))`. Ensure the implementation correctly calculates this. Incorrect sigmoid implementation affects the final prediction probabilities.",
    "delta_score": -1
  },
  {
    "error_name": "Backward pass not correctly implemented",
    "description": "Each layer's backward function should correctly compute gradients for inputs, weights, and biases. Check if the gradients are calculated as `grad_input = grad_output @ self.weights.T`, `grad_weights = self.input.T @ grad_output`, and `grad_bias = np.sum(grad_output, axis=0)`. Incorrect gradients will lead to improper model training.",
    "delta_score": -2
  },
  {
    "error_name": "Incorrect gradient calculation in get_grad_log_loss",
    "description": "The gradient of the log loss should be calculated as `-(Y - Y_hat) / (Y_hat * (1 - Y_hat))`. Verify this formula is used. Incorrect gradient calculation will lead to incorrect weight updates.",
    "delta_score": -1.5
  },
  {
    "error_name": "Missing activation checkpointing",
    "description": "Activation checkpointing should be used to save memory during training. Check if intermediate activations are stored and reused efficiently. Missing this can lead to excessive memory usage.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect FLOPs calculation",
    "description": "The FLOPs calculation should conclude that training is approximately 2 times slower than inference due to the backward pass. Ensure the explanation and calculation are clear and correct. Incorrect FLOPs estimation can mislead performance expectations.",
    "delta_score": -0.5
  }
]
# <expert_list_error_rubric_time_sec>:

# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Implementation Accuracy",
    "weight": 30.0,
    "checklist": [
      "Are the forward and backward methods correctly implemented for each layer?",
      "Is the ScaledReLU activation function correctly scaling outputs based on the input condition?",
      "Does the Linear layer use Kaiming initialization for weights and zero initialization for biases?",
      "Is the Sigmoid function correctly implemented without parameters?",
      "Is the get_grad_log_loss function correctly computing the gradient of the log loss?"
    ]
  },
  {
    "criterion": "Code Efficiency",
    "weight": 20.0,
    "checklist": [
      "Is the code implemented using numpy without unnecessary dependencies?",
      "Are matrix operations optimized to minimize computational overhead?",
      "Is activation checkpointing used to save memory?",
      "Does the code avoid redundant calculations?",
      "Is the code structured to facilitate easy debugging and testing?"
    ]
  },
  {
    "criterion": "Compatibility and Deployment",
    "weight": 20.0,
    "checklist": [
      "Is the code compatible with the proprietary software stack?",
      "Are there any dependencies that might conflict with the robot's onboard system?",
      "Is the code modular and adaptable for future updates or changes?",
      "Does the implementation support deployment directly onto the robot's system?",
      "Are there clear instructions for deploying the model on the robot?"
    ]
  },
  {
    "criterion": "Computational Complexity Analysis",
    "weight": 15.0,
    "checklist": [
      "Is the FLOP calculation for training versus inference correctly derived?",
      "Does the analysis correctly identify the FLOP requirements for matrix multiplications?",
      "Is the ratio of training to inference FLOPs accurately calculated?",
      "Are the assumptions and simplifications in the analysis clearly stated?",
      "Is the analysis presented in a clear and logical manner?"
    ]
  },
  {
    "criterion": "Documentation and Clarity",
    "weight": 15.0,
    "checklist": [
      "Is the code well-documented with comments explaining each function and class?",
      "Are variable names descriptive and indicative of their purpose?",
      "Is the overall structure of the code logical and easy to follow?",
      "Are there examples or test cases provided to demonstrate the code's functionality?",
      "Is the documentation sufficient for a non-expert to understand the implementation?"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:

