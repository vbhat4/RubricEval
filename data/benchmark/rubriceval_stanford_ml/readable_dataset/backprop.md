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
    "weight": 20.0,
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
    "weight": 35.0,
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
    "weight": 10.0,
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
      "good": "The code implements most of the required functions and classes, with 1 minor omission. For example, it might have forgotten to zero initialize the bias or it returned a decimal number ratio of training to inference instead of an integer. Errors for this criterion are not about implementation but about missing details in the instructions. Minor omissions are those that would lead to very similar solutions and would have trivial fixes.",
      "fair": "The code implements some of the required functions and classes, but has one moderate omission or two minor ones. For example, it didn't attempt Kaiming initialization (moderate omission) or implemented ReLU instead of ScaledReLU (moderate omission). Multiple small omissions may include not initializing the bias (minor omission) and considering non-matrix multiplications in FLOP calculation (minor omission). Moderate omissions are those that would lead to different solutions and would require some thinking to fix.",
      "poor": "The code has a major omission or multiple moderate ones. For example, it didn't implement activation checkpointing at all (major omission), the code uses pseudocode (major omission), or one of the required functions or classes is missing (major omission). Major omissions are those that would lead to significantly different solutions and that would require significant thinking to fix."
    }
  },
  {
    "criterion": "Implementation of the Forward Pass",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The forward pass for each layer is correctly implemented: Linear as `X @ W + b`, ScaledReLU as `X * 2 if X > 1 else 0`, and Sigmoid as `1 / (1 + np.exp(-X))`. Activation checkpointing is correctly used in the forward pass: Linear saves self.X, ScaledReLU saves the mask, and Sigmoid saves the output self.O. Biases are initialized to zero using `np.zeros(h)` and weights are initialized using Kaiming initialization `np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)`.",
      "good": "The forward pass is mostly correct, with 1 minor error. For example, the mask is not saved in ScaledReLU (activation checkpointing). Minor errors are those that would nearly not affect the final code and solution (e.g., activation checkpointing for ScaledReLU) or use `np.sqrt(1. / input_dim)` instead of `np.sqrt(2. / input_dim)` in Kaiming initialization.",
      "fair": "The forward pass is partially correct, with one moderate error or two minor ones. For example, scaled ReLU doesn't do what it's supposed to (moderate error), or Kaiming initialization is completely wrong (moderate error), or the sigmoid forgot to take the negative sign in the exponent (moderate error), or using shape mistakes e.g. `W@X + b` instead of `X @ W + b` in the linear layer. Moderate errors are either those that could give different final answers (e.g., forgetting to initialize the weight, or forgetting the sign in the exponent) but can be easily fixed, or those that would fail loudly and are close to being correct e.g., a shape mistake.",
      "poor": "The forward pass contains multiple moderate errors or a major one. Major errors are those that would require significant thinking to fix, e.g., `O = 1 / (1 + np.exp(-X))` is not implemented or is completely wrong."
    }
  },
  {
    "criterion": "Implementation of the Backward Pass and Gradient Calculation",
    "weight": 35.0,
    "performance_to_description": {
      "excellent": "The backward pass for each layer is correctly implemented: Linear layer gradients are calculated correctly using the chain rule (e.g., of correct implementation `dL_dW = self.X.T @ dL_dY`, `dL_db = dL_dY.sum(axis=0)`, `dL_dX = dL_dY @ W.T` where `self.X` is checkpointed/stored in the forward pass), ScaledReLU backward pass correctly applies the double gradient only where the input was greater than 1 (e.g., `dL_dX = dL_dY * self.scaled_mask` where `self.scaled_mask = (X > 1) * 2` is checkpointed/stored in forward pass), and Sigmoid backward pass correctly applies the derivative of the sigmoid function (e.g., `dL_dX = dL_dY * self.O * (1 - self.O)` where `self.O=1 / (1 + np.exp(-self.X))` is checkpointed/stored in forward pass). The get_grad_log_loss function accurately computes the gradient of the log loss with respect to the weights (e.g., `1/len(Y) * (Y_hat - Y) / ((Y_hat * (1 - Y_hat)) + eps)` including the normalization by batch size). Stochastic Gradient Descent (SGD) is correctly applied by updating the weights using the computed gradients and a learning rate (e.g., `self.W -= lr * dL_dW` and `self.b -= lr * dL_db`). Activation checkpointing is used in the backward pass to save intermediate activations and those are reused in the backward pass.",
      "good": "The backward pass is mostly correct, with minor errors in one backward pass. For example, one of the intermediate activations is not reused in the backward pass, or `get_grad_log_loss` forgot to normalize by the batch size. Minor errors are those that would nearly not affect the final code / solution and are easy to fix, or those that would fail loudly and thus be easy to spot (e.g., matrix multiplication wouldn't run due to shape mismatch).",
      "fair": "The backward pass has one moderate error or two minor ones. Moderate errors are those that could give very different final answers (e.g., the sign of the update in SGD is wrong, or the chain rule is not applied correctly `dL_dX = self.O * (1 - self.O)`) but are easy to fix (the solution is close to correct), or errors that would fail loudly and thus be easy to spot (e.g., matrix multiplication wouldn't run due to shape mismatch).",
      "poor": "The backward pass contains multiple moderate errors or a major one. Major errors are those that would require significant thinking to fix, e.g., `loss_nabla_X = loss_nabla_O @ self.W.T` is not implemented or is completely wrong."
    }
  },
  {
    "criterion": "Overall Code Quality",
    "weight": 10.0,
    "performance_to_description": {
      "excellent": "The code is fully functional, efficient, and follows best practices in Python programming. It uses vectorized operations instead of loops, handles potential numerical instabilities (e.g., using epsilon in get_grad_log_loss), and uses clear, descriptive variable names. The code is simple, concise, and easy to understand, with no unnecessary or irrelevant details. For example, it adds useful/concise documentation when needed and arguments to functions when needed.",
      "good": "The code is functional and mostly follows best practices, with one minor issue. For example, it may forget the epsilon in get_grad_log_loss, use unnecessarily many lines of code, or doesn't type the arguments to functions. The code is generally easy to understand but may have a few unnecessary details. Minor issues are those that would most often not affect the final outcome of the code and are easy to fix.",
      "fair": "The code is mostly functional but has two minor issues or one moderate one. For instance, it might use loops instead of vectorization somewhere (moderate issue), or have unclear variable names throughout (minor issue) as well as lacking docstrings (minor issue). The code may be somewhat difficult to understand or contain several unnecessary details. Moderate issues are those that would significantly impact the final outcome of the code but are not wrong (e.g., slow due to a for loop), or those that significantly impact the code quality but are easy to fix.",
      "poor": "The code contains multiple moderate errors or a major one. Major errors are those that would require significant refactoring to fix and significantly impact the code quality."
    }
  },
  {
    "criterion": "Computational Complexity Analysis",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The analysis correctly considers only the linear layer for matrix multiplications. It accurately calculates the FLOPs for forward pass as 2nio, backward pass as 4nio (2nio for gradients w.r.t. activations and 2nio for gradients w.r.t. weights), and total training as 6nio (forward + backward). It correctly determines the ratio of training to inference FLOPs as 3 (6nio / 2nio), and notes that this ratio is independent of n for large n (or equivalent to only considering matrix multiplications). The explanation is clear, concise, and shows a deep understanding of the computational complexity involved, including the reasoning behind the 2nio approximation for matrix multiplication (no dot products, each for vectors of shape i, and dot product of two vectors of dimensionality d is approx 2d).",
      "good": "The analysis is mostly correct, with one minor issue. For example, it may have the correct equations but have some computation error (minor issue), or may not recognize that the ratio is independent of n for large n (or rather large n is equivalent to only considering matrix multiplications) (minor issue). Minor issues are such that the overall reasoning is sound and a simple proofreading would help find the error.",
      "fair": "The analysis contains a moderate error. For instance, a common moderate error is to consider training to only perform the backward pass (rather than forward + backward) which would lead to conclude that the ratio is $4nio/2nio=2$ rather than $3$. Another common moderate error would be to consider the forward pass as $nio$ rather than $2nio$ (common and easy to fix error). Another common error may be to only consider the gradients w.r.t. activations in the backward pass (rather than both activations and weights) which would be a 2nio rather than 4nio. Moderate errors uncover a lack of understanding of at least one thing but are easy to fix.",
      "poor": "The analysis contains multiple moderate errors or a major one. Major errors are those that require significant thinking to fix. For example, the answer gets the number of FLOPS in the backward pass completely wrong."
    }
  },
  {
    "criterion": "Clarity and Conciseness of Explanation",
    "weight": 5.0,
    "performance_to_description": {
      "excellent": "The explanation is clear, concise, and easy to follow. It avoids irrelevant or unnecessary details, focuses on the key concepts and calculations, and presents information in a logical order. The language used is precise and appropriate for the technical nature of the topic.",
      "good": "The explanation is mostly clear and concise, with a minor issue. A minor issue may be that it includes a few unnecessary details or has slight inconsistencies in the logical flow, but overall it's easy to follow and focuses on the key points.",
      "fair": "The explanation is somewhat clear but has one moderate issue or two minor ones. For example, it might include several irrelevant details, skip over some important points, or present information in a confusing order. The language might be imprecise or inconsistent in places. Moderate issues are those that would still make the answer understandable but significantly impact the clarity and quality of the explanation. These should be relatively easy to fix.",
      "poor": "The explanation contains major issues that make it unclear, verbose, or difficult to follow. It might include many irrelevant details, miss crucial points, or present information in a disorganized manner. The language used might be inappropriate or consistently imprecise, making it hard to understand the key concepts and calculations. Major issues are those that would require significant thinking to fix and significantly impact the clarity and quality of the explanation."
    }
  }
]
# <expert_rubric_time_sec>:
3600.0
