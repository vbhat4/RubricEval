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

You should write the `forward`, `backward`, `update_params_`, and `__init__` methods for the `Linear`, `ScaledReLU`, and `Sigmoid` classes, as well as the `get_grad_log_loss` function. Note that `ScaledReLU` is similar to ReLU, but it scales the output by 2 if the input is greater than 1 and 0 otherwise. For the linear layer, use Kaiming initialization for the weights and zero initialization for the bias. Use activation checkpointing to save memory wherever possible. Implement everything using NumPy. Your code should be fully functional.

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
        self.scaled_mask = (X > 1).astype(X.dtype) * 2
        return self.scaled_mask * X 

    def backward(self, loss_nabla_O):
        loss_nabla_X = loss_nabla_O * self.scaled_mask
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

Complexity: only linear layer has matrix multiplications so we should only consider that. As a reminder, the FLOPs for multiplication of two matrices of shape $n \times o$ and $i \times o$ is approximately $2nio$ because there are $no$ dot products, each for vectors of shape $i$ and dot product two vectors of dimensionality $d$ is approx $2d$ (one $d$ for mult, and $d-1$ for add). The forward for the linear thus requires $2nio$ FLOPs. The backward requires $4nio$ FLOPs: $2nio$ for the gradients w.r.t. inputs $2nio$ for gradients w.r.t. weights. So $4nio$. Training requires both the backward and the forward pass so $6nio$ in total. Inference only requires the forward pass so $2nio$. The ratio is $\frac{6nio}{2nio}=3$ so training requires 3x more compute than inference. This is independent of $n$ (or rather large $n$ is equivalent to only considering matrix multiplications).
# <expert_checklist>:
[
  "Is Kaiming initialization correctly used for weights in the Linear layer (np.random.randn(d, h) * np.sqrt(2/d))? (High importance)",
  "Are biases in the Linear layer initialized to zeros (np.zeros(h))? (Moderate importance)",
  "Is the forward pass of the Linear layer correctly implemented as X @ self.W + self.b? (High importance)",
  "Is the backward pass of the Linear layer w.r.t. activations correct (loss_nabla_O @ self.W.T)? (High importance)",
  "Is the backward pass of the Linear layer w.r.t. weights correct (self.X.T @ loss_nabla_O)? (High importance)",
  "Is the backward pass of the Linear layer w.r.t. bias correct (loss_nabla_O.sum(axis=0))? (Moderate importance)",
  "Is activation checkpointing used to save X in the forward pass of the Linear layer? (Low importance)",
  "Is the forward pass of ScaledReLU correctly implemented to scale output by 2 if input > 1, and 0 otherwise? For example, using np.where(X > 1, X * 2, 0) or X * self.scaled_mask where scaled_mask = (X > 1).astype(X.dtype) * 2. (High importance)",
  "Is the backward pass of ScaledReLU correctly implemented (loss_nabla_O * self.scaled_mask)? (High importance)",
  "Is activation checkpointing used to save scaled_mask in the forward pass of ScaledReLU? (Low importance)",
  "Is the forward pass of Sigmoid correctly implemented as 1 / (1 + np.exp(-X))? (High importance)",
  "Is the backward pass of Sigmoid correctly implemented (loss_nabla_O * self.O * (1-self.O))? (High importance)",
  "Is activation checkpointing used to save O in the forward pass of Sigmoid? (Low importance)",
  "Is get_grad_log_loss correctly implemented as something along the lines of 1/len(Y) * (Y_hat-Y)/((Y_hat * (1-Y_hat))+ eps)? (High importance)",
  "Is normalization by batch size done only in get_grad_log_loss rather than in layer backward passes? (Moderate importance)",
  "Is a small epsilon added in get_grad_log_loss to avoid division by zero? (Moderate importance)",
  "Is update_params_ correctly implemented for the Linear layer (self.W -= lr * loss_nabla_W, self.b -= lr * loss_nabla_b)? (High importance)",
  "Is update_params_ correctly skipped for ScaledReLU and Sigmoid as they have no parameters? (Low importance)",
  "Is the number of FLOPs for forward pass of linear layer correctly calculated as 2nio? (Moderate importance)",
  "Is the number of FLOPs for backward pass of linear layer correctly calculated as 4nio (2nio for gradient w.r.t. activations and 2nio for gradient w.r.t. weights)? (Moderate importance)",
  "Is the total number of FLOPs for training correctly calculated as inference + backward pass FLOPs? Note that you shouldn't penalize if they correctly added their forward and backward FLOPs, but one of their two FLOPs is wrong (because they will already be penalized for that above). If everything else is correct it should be 4nio+2nio=6nio. (Moderate importance)",
  "Is the number of FLOPs for inference correctly calculated as just the inference FLOPs? i.e. 2nio. (Moderate importance)",
  "Is the ratio of training to inference FLOPs correctly calculated? Note that you shouldn't penalize if the ratio they computed is correct for their FLOPs but their training or inference FLOPs are wrong (because they will already be penalized for that above). If everything else is correct it should be training/inference=6nio/2nio=3. (Moderate importance)",
  "Is it noted that the ratio is independent of n (or that large n is equivalent to only considering matrix multiplications)? (Low importance)",
  "Is the code fully functional and attempting to implement all required components? (High importance)",
  "Does the code follow good coding practices, including clear variable names? (Moderate importance)",
  "Is the code simple, concise, and easy to understand? (Moderate importance)",
  "Are unnecessary or irrelevant details avoided in the code and explanation? (Low importance)",
  "Does the code handle potential numerical instabilities (e.g., using epsilon in get_grad_log_loss)? (High importance)",
  "Is the code efficient, e.g., using vectorized operations instead of loops? (High importance)",
  "Are all required functions/classes implemented: Linear, ScaledReLU, Sigmoid, get_grad_log_loss? (High importance)"
]
# <expert_checklist_time_sec>:
830.0
# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect initialization of Linear layer weights",
    "description": "The weights of the Linear layer should be initialized using Kaiming initialization, i.e., `np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)`. Check if the weights are initialized correctly by inspecting the code for this specific initialization method.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect initialization of Linear layer bias",
    "description": "The bias of the Linear layer should be initialized to zeros, i.e., `np.zeros(output_dim)`. Verify if the bias is initialized correctly by checking the code for zero initialization.",
    "delta_score": -0.25
  },
  {
    "error_name": "Incorrect forward pass in Linear layer",
    "description": "The forward pass of the Linear layer should compute `X @ self.W + self.b`. Check if the implementation correctly performs this matrix multiplication and addition.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect forward pass in ScaledReLU",
    "description": "The ScaledReLU should output `X * 2` if `X > 1` and `0` otherwise. This can be implemented using `np.where(X > 1, X * 2, 0)`. Verify if the forward pass correctly applies this logic.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect forward pass in Sigmoid",
    "description": "The Sigmoid function should be computed as `1 / (1 + np.exp(-X))`. Check if the forward pass correctly implements this function.",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing activation checkpointing",
    "description": "Activation checkpointing should be used to save memory. For the Linear layer, save the input `X`; for the Sigmoid, save the output `O`; and for the ScaledReLU, save the scaled mask. Ensure these are saved and used in the backward pass.",
    "delta_score": -1.5
  },
  {
    "error_name": "Incorrect backward pass in Sigmoid",
    "description": "The backward pass of the Sigmoid should compute `loss_nabla_X = loss_nabla_O * self.O * (1-self.O)`. Ensure the chain rule is applied correctly by checking if `loss_nabla_O` is multiplied in the computation.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect backward pass in ScaledReLU",
    "description": "The backward pass of the ScaledReLU should compute `loss_nabla_X = loss_nabla_O * self.scaled_mask`. Verify if the gradients are scaled correctly as per the forward pass logic.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect backward pass in Linear layer w.r.t activations",
    "description": "The backward pass of the Linear layer w.r.t. activations should compute `loss_nabla_X = loss_nabla_O @ self.W.T`. Check if the matrix multiplication is performed correctly and dimensions match.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect backward pass in Linear layer w.r.t weights",
    "description": "The backward pass of the Linear layer w.r.t. weights should compute `loss_nabla_W = self.X.T @ loss_nabla_O`. Ensure the matrix multiplication is correct and dimensions are appropriate.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect backward pass in Linear layer w.r.t bias",
    "description": "The backward pass of the Linear layer w.r.t. bias should compute `loss_nabla_b = loss_nabla_O.sum(axis=0)`. Verify if the sum is performed over the correct axis.",
    "delta_score": -0.5
  },
  {
    "error_name": "Unnecessary update_params_ implementation for ScaledReLU or Sigmoid",
    "description": "The ScaledReLU and Sigmoid layers should not have an `update_params_` method as they have no parameters to update. Check if the code incorrectly implements this method for these layers.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect update_params_ in Linear layer",
    "description": "The `update_params_` method in the Linear layer should update weights and biases using SGD, i.e., `self.W -= lr * loss_nabla_W` and `self.b -= lr * loss_nabla_b`. Verify if the updates are implemented correctly.",
    "delta_score": -0.5
  },
  {
    "error_name": "Minor mistake across backward pass",
    "description": "A minor mistake across multiple backward passes, such as incorrect scaling of gradients or transposed gradients. Penalize once for consistent minor errors across backward passes.",
    "delta_score": -0.75
  },
  {
    "error_name": "Incorrect gradient calculation in get_grad_log_loss",
    "description": "The gradient of the log loss should be calculated as `1/len(Y) * (Y_hat-Y)/((Y_hat * (1-Y_hat))+ eps)`. Check if the gradient is normalized by the number of examples and computed correctly.",
    "delta_score": -0.5
  },
  {
    "error_name": "Other coding mistakes",
    "description": "Other coding mistakes not listed above, such as syntax errors or missing imports, that prevent the code from running. Penalize if the code doesn't execute due to such errors.",
    "delta_score": -0.5
  },
  {
    "error_name": "Poor coding practices",
    "description": "Poor coding practices, such as unclear variable names or lack of comments, that affect readability. Penalize partially for significant readability issues.",
    "delta_score": -0.5
  },
  {
    "error_name": "Inefficient code",
    "description": "Inefficient code, such as using loops instead of vectorized operations, that could be optimized. Penalize for major inefficiencies that impact performance.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect FLOPs calculation for forward pass",
    "description": "The FLOPs for the forward pass of the Linear layer should be `2nio`. Verify if the FLOPs calculation is performed correctly for the forward pass.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect FLOPs calculation for backward pass",
    "description": "The FLOPs for the backward pass of the Linear layer should be `4nio`. Check if the FLOPs calculation is correct for the backward pass.",
    "delta_score": -1
  },
  {
    "error_name": "Failure to recognize training requires both forward and backward pass",
    "description": "Training requires both forward and backward passes, totaling `6nio` FLOPs. Ensure the response recognizes this requirement and calculates the total FLOPs correctly.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect computation of training to inference FLOPs ratio",
    "description": "The ratio of training to inference FLOPs should be `3`, calculated as `6nio/2nio`. Verify if the ratio is computed correctly based on the FLOPs calculations.",
    "delta_score": -0.5
  },
  {
    "error_name": "Unclear answer or formatting",
    "description": "The answer is unclear or lacks necessary details, making it difficult to understand. Penalize for lack of clarity or poor formatting.",
    "delta_score": -0.5
  },
  {
    "error_name": "Irrelevant or excessive details",
    "description": "The answer includes irrelevant or excessive details that are not necessary for the solution. Penalize for verbosity that detracts from the main points.",
    "delta_score": -0.5
  }
]
# <expert_list_error_rubric_time_sec>:
2350.0
# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Following assignment instructions",
    "weight": 10.0,
    "checklist": [
      "Does the code attempt to use activation checkpointing? (Moderate importance)",
      "Does the code use Kaiming initialization for weights and zero initialization for bias? (Moderate importance)",
      "Are all the required functions/classes implemented: Linear, ScaledReLU, Sigmoid, get_grad_log_loss? (High importance)",
      "Does the FLOPs calculation only consider matrix multiplications? (Moderate importance)",
      "Is the code fully functional or at least attempting to be? (Moderate importance)",
      "Does the code follow all the instructions given in the prompt? (Moderate importance)",
      "Is the final answer for the FLOPs an integer? (Low importance)"
    ]
  },
  {
    "criterion": "Implementation of the forward pass",
    "weight": 25.0,
    "checklist": [
      "Is the forward pass for linear layer correctly implemented as X @ W + b? (Moderate importance)",
      "Is the forward pass for ScaledReLU correctly implemented as X * 2 if X > 1 else 0? For example it can be computed as `X * scaled_mask` where `scaled_mask=(X > 1) * 2` or it can be computed as `np.where(X > 1, X * 2, 0)` (Moderate importance)",
      "Is the forward pass for sigmoid correctly implemented as 1 / (1 + np.exp(-X))? (Moderate importance)",
      "Does the activation checkpointing correctly save O for the sigmoid, scaled_mask for the ScaledReLU, and X for the linear layer? (Moderate importance)",
      "Are the weights of the linear layer correctly initialized using Kaiming initialization (np.random.randn(d, h) * np.sqrt(2/d))? (Moderate importance)",
      "Is the bias of the linear layer correctly initialized to zero (np.zeros(h))? (Low importance)"
    ]
  },
  {
    "criterion": "Implementation of the backward pass and gradient calculation",
    "weight": 25.0,
    "checklist": [
      "Is the backward pass for linear layer correctly implemented (loss_nabla_X = loss_nabla_O @ self.W.T, loss_nabla_W = self.X.T @ loss_nabla_O, loss_nabla_b = loss_nabla_O.sum(axis=0))? (Moderate importance)",
      "Is the backward pass for ScaledReLU correctly implemented (loss_nabla_X = loss_nabla_O * self.scaled_mask)? (Moderate importance)",
      "Is the backward pass for sigmoid correctly implemented (loss_nabla_X = loss_nabla_O * self.O * (1-self.O))? (Moderate importance)",
      "Is the gradient calculation in get_grad_log_loss correct (1/len(Y) * (Y_hat-Y)/((Y_hat * (1-Y_hat))+ eps))? (Moderate importance)",
      "Is normalization by the number of examples in the batch only done in get_grad_log_loss? (Low importance)",
      "Is SGD correctly applied in the update_params_ method of the linear layer (self.W -= lr * loss_nabla_W and self.b -= lr * loss_nabla_b)? (Moderate importance)",
      "Is the update_params_ method correctly skipped for ScaledReLU and Sigmoid as they have no parameters? (Moderate importance)",
      "Are the correct gradients returned for each layer and the loss? (Moderate importance)",
      "Is activation checkpointing correctly used in the backward pass instead of recomputing the activations? (Moderate importance)"
    ]
  },
  {
    "criterion": "Overall code quality",
    "weight": 15.0,
    "checklist": [
      "Is the code fully functional? (High importance)",
      "Does the code follow good coding practices? (Low importance)",
      "Is the code simple and easy to understand? (Low importance)",
      "Is the code concise and free of unnecessary or irrelevant details? (Low importance)",
      "Are clear and descriptive names used for variables and functions? (Low importance)",
      "Is the code efficient, avoiding unnecessary operations and loops (i.e., uses vectorized operations)? (Moderate importance)",
      "Does the code handle edge cases and potential errors (e.g., division by zero, invalid inputs)? (Low importance)",
      "Are necessary comments and docstrings included to explain the code if not self-explanatory? (Low importance)",
      "Does the code follow correct Python style and formatting? (Low importance)",
      "Is the code numerically stable? (Moderate importance)"
    ]
  },
  {
    "criterion": "Computational Complexity Analysis",
    "weight": 20.0,
    "checklist": [
      "Does the analysis correctly consider only the linear layer for matrix multiplications? (Low importance)",
      "Is the number of FLOPs for a matrix multiplication correctly calculated as approximately 2nio? or no(2i-1) if you consider the fact that there's one less add than multiplications. (Moderate importance)",
      "Is the number of FLOPs for the forward pass of the linear layer correctly calculated as 2nio? (Moderate importance)",
      "Is the number of FLOPs for the backward pass of the linear layer correctly calculated as 4nio? 2nio for the gradient w.r.t. activations and 2nio for the gradient w.r.t. weights. (Moderate importance)",
      "Is the number of FLOPs for inference correctly calculated as just the inference FLOPs? i.e. 2nio. (Moderate importance)",
      "Is the total number of FLOPs for training correctly calculated as inference + backward pass FLOPs? Note that you shouldn't penalize if they correctly added their forward and backward FLOPs, but one of their two FLOPs is wrong (because they will already be penalized for that above). If everything else is correct it should be `4nio+2nio=6nio`. (Moderate importance)",
      "Is the ratio of training to inference FLOPs correctly calculated? Note that you shouldn't penalize if the ratio they computed is correct for their FLOPs but their training or inference FLOPs are wrong (because they will already be penalized for that above). If everything else is correct it should be training/inference=6nio/2nio=3. (Moderate importance)",
      "Does the response correctly notice that the ratio is independent of $n$ (or rather large $n$ is equivalent to only considering matrix multiplications). (Low importance)"
    ]
  },
  {
    "criterion": "Clarity and conciseness of explanation",
    "weight": 5.0,
    "checklist": [
      "Is the explanation of the FLOPs calculation clear and easy to follow? (Moderate importance)",
      "Does the response avoid irrelevant or unnecessary details not asked in the question? (Low importance)",
      "Is the response concise and to the point? (Low importance)",
      "Are the key concepts and calculations clearly explained? (Low importance)"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:
1120.0
# <expert_rubric>:
[
  {
    "criterion": "Following Assignment Instructions",
    "weight": 10.0,
    "performance_to_description": {
      "excellent": "The code implements all required functions and classes: Linear, ScaledReLU, Sigmoid, and get_grad_log_loss. It uses activation checkpointing, Kaiming initialization for weights, and zero initialization for biases. The FLOPs calculation considers only matrix multiplications, and the final answer is an integer. The code is fully functional and follows all instructions in the prompt.",
      "good": "The code implements most of the required functions and classes, with minor omissions or errors. It attempts to use activation checkpointing and Kaiming initialization, but may have minor issues. The FLOPs calculation is mostly correct, but may include minor errors. The code is mostly functional and follows most instructions.",
      "fair": "The code implements some of the required functions and classes, but with significant omissions or errors. It may attempt activation checkpointing or Kaiming initialization, but with major issues. The FLOPs calculation is attempted but contains significant errors. The code is partially functional and follows some instructions.",
      "poor": "The code fails to implement most of the required functions and classes. It does not attempt activation checkpointing or Kaiming initialization. The FLOPs calculation is incorrect or missing. The code is non-functional and does not follow instructions."
    }
  },
  {
    "criterion": "Implementation of the Forward Pass",
    "weight": 25.0,
    "performance_to_description": {
      "excellent": "The forward pass for each layer is correctly implemented: Linear as X @ W + b, ScaledReLU as X * 2 if X > 1 else 0, and Sigmoid as 1 / (1 + np.exp(-X)). Activation checkpointing is correctly used, and weights are initialized using Kaiming initialization. Biases are initialized to zero.",
      "good": "The forward pass is mostly correct, with minor errors in one or two layers. Activation checkpointing is attempted but may have minor issues. Weights are mostly correctly initialized, but there may be minor errors in bias initialization.",
      "fair": "The forward pass is partially correct, with significant errors in multiple layers. Activation checkpointing is attempted but with major issues. Weights and biases are not correctly initialized.",
      "poor": "The forward pass is incorrect for most layers. Activation checkpointing is not attempted. Weights and biases are not initialized correctly."
    }
  },
  {
    "criterion": "Implementation of the Backward Pass and Gradient Calculation",
    "weight": 25.0,
    "performance_to_description": {
      "excellent": "The backward pass for each layer is correctly implemented: Linear layer gradients are calculated correctly, ScaledReLU and Sigmoid backward passes are correct, and get_grad_log_loss is implemented accurately. SGD is correctly applied, and activation checkpointing is used in the backward pass.",
      "good": "The backward pass is mostly correct, with minor errors in one or two layers. Gradient calculations are mostly accurate, and SGD is applied with minor issues. Activation checkpointing is attempted but may have minor errors.",
      "fair": "The backward pass is partially correct, with significant errors in multiple layers. Gradient calculations are attempted but contain major errors. SGD is applied incorrectly, and activation checkpointing is not used correctly.",
      "poor": "The backward pass is incorrect for most layers. Gradient calculations are missing or incorrect. SGD is not applied, and activation checkpointing is not used."
    }
  },
  {
    "criterion": "Overall Code Quality",
    "weight": 15.0,
    "performance_to_description": {
      "excellent": "The code is fully functional, follows good coding practices, is simple and easy to understand, and uses efficient, vectorized operations. It handles edge cases, includes necessary comments, and follows Python style and formatting. The code is numerically stable.",
      "good": "The code is mostly functional, follows most coding practices, is relatively easy to understand, and uses some vectorized operations. It handles some edge cases, includes some comments, and mostly follows Python style. The code is mostly stable.",
      "fair": "The code is partially functional, follows some coding practices, is somewhat difficult to understand, and uses few vectorized operations. It handles few edge cases, includes few comments, and partially follows Python style. The code has stability issues.",
      "poor": "The code is non-functional, does not follow coding practices, is difficult to understand, and does not use vectorized operations. It does not handle edge cases, lacks comments, and does not follow Python style. The code is unstable."
    }
  },
  {
    "criterion": "Computational Complexity Analysis",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The analysis correctly considers only the linear layer for matrix multiplications, calculates FLOPs accurately for both forward and backward passes, and correctly determines the training to inference FLOPs ratio. The explanation is clear and concise.",
      "good": "The analysis is mostly correct, with minor errors in FLOPs calculation or ratio determination. The explanation is mostly clear, with minor issues.",
      "fair": "The analysis is partially correct, with significant errors in FLOPs calculation or ratio determination. The explanation is somewhat unclear.",
      "poor": "The analysis is incorrect, with major errors in FLOPs calculation or ratio determination. The explanation is unclear or missing."
    }
  },
  {
    "criterion": "Clarity and Conciseness of Explanation",
    "weight": 5.0,
    "performance_to_description": {
      "excellent": "The explanation of the FLOPs calculation is clear, concise, and easy to follow. It avoids irrelevant details and clearly explains key concepts and calculations.",
      "good": "The explanation is mostly clear and concise, with minor issues. It mostly avoids irrelevant details and explains key concepts.",
      "fair": "The explanation is somewhat unclear or verbose, with significant issues. It includes some irrelevant details and does not clearly explain key concepts.",
      "poor": "The explanation is unclear, verbose, or missing. It includes irrelevant details and does not explain key concepts."
    }
  }
]
# <expert_rubric_time_sec>:

