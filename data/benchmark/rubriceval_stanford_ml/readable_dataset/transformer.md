# <category>:
Stats & ML
# <instruction>:
Think of a transformer called MyT that has the following architecture:
\begin{itemize}
    \item $57$ transformer blocks, i.e. $N$ in the figure.
    \item model dimensionality of $d=1756$, i.e. all the inputs, outputs and activations of transformer blocks are vectors in $\mathbb{R}^{d}$. In particular, the Key, Value, Query, and Output matrices are in $\mathbb{R}^{d\times d}$.
    \item a max context length of $C=2800$, i.e. there are $C$ positional embeddings in $\mathbb{R}^{d}$.
    \item 439 attention heads.
    \item normalization layers are RMSNorm
    \item a vocabulary size of $V=63257$, i.e., there are $V$ input embeddings in $\mathbb{R}^{d}$, and the last linear layer is a softmax over $V$ classes.
    \item the feedforward model is two layer MLP with temporary hidden activations of size $\mathbb{R}^{4*d}$, i.e. the computational graph is $input\ (\mathbb{R}^{d}) \to \mathrm{\mathbf{linear}} \to tmp\ (\mathbb{R}^{4*d}) \to \mathrm{\mathbf{ReLU}} \to tmp\ (\mathbb{R}^{4*d}) \to \mathrm{\mathbf{linear}}  \to output\ (\mathbb{R}^{d})$ where layers are in bold and there's no bias in Linear layers.
\end{itemize}

Where do most of the parameters come from in MyT?
Your answer should be one of the following:
The input embeddings, the positional embeddings, all the multi-head attention blocks, all the normalization layers, all the feedforward blocks, the final linear layer. How many parameters does that part have?
# <expert_solution>:
Feedforward blocks require $2*4*LD^2$ parameters which is the bottleneck (2 layers per block, each with $4D^2$ parameters). In contrast, all the self-attention layers require $4LD^2$ parameters (one for K,V,Q,O). All other layers have a negligible amount of parameters as they either don't have matrix parameters (e.g. activation / normalization layers) or no dependencies on the number of layers (e.g. embeddings and last layer). Putting all the numbers in, we have $2*4*1756^2*57 = 1406092416$ parameters for the feedforward blocks.
# <expert_checklist>:
[
  "Does the output correctly focuses the feedforward and self-attention layers as they have matrix parameters so the most parameters are there?",
  "Does the output correctly states that the feedforward blocks have the most parameters?",
  "Does the output recognizes that feedforward blocks require $2*4*LD^2$ parameters? 2 because there are two layers in the MLP, 4*LD^2 is the number of parameters per matrix where the 4 comes from the upscaling from d to 4d?",
  "Correctly recognizes that the number of heads doesn't affect the number of parameters?",
  "Correct formula for number of parameters in feedforward blocks: $2*4*LD^2=2*4*1756^2*57$",
  "Correct multiplication for total number of parameters: $2*4*1756^2*57=1406092416$",
  "Correctly recognizes that the vocabulary size and the context length don't affect the number of parameters in a feedforward block",
  "Correctly doesn't consider the bias in the linear layers as it's mentioned that there's no bias parameters in the assignment",
  "Provides explanation for the reasoning as to why feedforward blocks have the most parameters",
  "Is well written and easy to understand",
  "Is the answer one of the provided options: 'input embeddings, the positional embeddings, all the self-attention blocks, all the normalization layers, all the feedforward blocks, the final linear layer'",
  "Doesn't add many unecessary details",
  "Does the answer show clear understanding of the transformer architecture"
]
# <expert_checklist_time_sec>:
360
# <expert_list_error_rubric>:
[
  {
    "error_name": "Not one of the provided options",
    "description": "The answer is not one of the provided options: 'input embeddings, the positional embeddings, all the multi-head attention blocks, all the normalization layers, all the feedforward blocks, the final linear layer'. It's ok to paraphrase those options (doesn't need to be strict match).",
    "delta_score": -0.5
  },
  {
    "error_name": "Doesn't focus on feedforward and self-attention blocks",
    "description": "The answer doesn't focus on feedforward and self-attention blocks, i.e. it focuses (it's fine to talk about other layers but not to majorily focus on them) on another layer that doesn't have much parameters. Any answer that states that the layer with the most parameters is not feedforward or self-attention should be fully penalized here. Answers that spends as much time talking about other layers as both feedforward and self-attention are also penalized here should be half penalized.",
    "delta_score": -0.5
  },
  {
    "error_name": "Doesn't state that feedforward blocks have the most parameters",
    "description": "The answer doesn't state that feedforward blocks have the most parameters. Any other answer should be fully penalized here.",
    "delta_score": -3
  },
  {
    "error_name": "Incorrect formula for number of parameters in the layer with most parameters",
    "description": "The formula of number of parameters for the layer that the answer identifies as the one with the most parameters, is incorrect. For feedforward the correct formula is $2*4*L*D^2$, for self-attention it's $4*L*D^2$, for input embeddings and final linear layer it's $V*D$, for positional embeddings it's $C*D$, for normalization layers it's $L*D$",
    "delta_score": -2
  },
  {
    "error_name": "Incorrect application of the formula",
    "description": "The formula is correct but the math (i.e actual final answer after multiplication) is wrong. Give full credit for that even if the formula is wrong or the feedforward blocks are not correctly identified. I.e. this is jsut to know if the output is multiplying correctly.",
    "delta_score": -1
  },
  {
    "error_name": "Unecessarily long and detailed",
    "description": "The answer is unnecessarily long and detailed, it talks about things that are not asked/relevant to the question.",
    "delta_score": -0.5
  },
  {
    "error_name": "Lack of clarity in explanation",
    "description": "The answer is hard to understand",
    "delta_score": -0.75
  },
  {
    "error_name": "Doesn't provide any explanation",
    "description": "Doesn't explain the reasoning of the answer.",
    "delta_score": -0.75
  }
]
# <expert_list_error_rubric_time_sec>:
1054
# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Following assignment instructions",
    "weight": 10.0,
    "checklist": [
      "Is the answer one of the provided options: 'input embeddings, the positional embeddings, all the multi-head attention blocks, all the normalization layers, all the feedforward blocks, the final linear layer'?",
      "Does the output answer 'where do most of the parameters come from in MyT?'",
      "Does the output answer 'How many parameters does that part have?'"
      "Does the output consider all the provided information, e.g. there's no bias in the linear layers, normalization layers are RMSNorm, the dimensionality of the model is d=1756, etc?"
    ]
  },
  {
    "criterion": "Understanding of Transformer Architecture",
    "weight": 65.0,
    "checklist": [
      "Does the output correctly identifies the feedforward blocks as the component with the most parameters?",
      "Does the output correctly states that feedforward blocks require $2*4*LD^2$ parameters which is the bottleneck?",
      "Does the output correctly states focuses on self-attention and feedforward blocks? All other layers have a negligible amount of parameters as they either don't have matrix parameters (e.g. activation / normalization layers) or no dependencies on the number of layers (e.g. embeddings and last layer)?",
      "If the output mentions the number of parameter for each layers, is it correct? For feedforward blocks it's $2*4*L*D^2$, for self-attention it's $4*L*D^2$, for input embeddings and final linear layer it's $V*D$, for positional embeddings it's $C*D$, for normalization layers it's $L*D$"
    ]
  },
   {
    "criterion": "Mathematical Capabilities",
    "weight": 10.0,
    "checklist": [
      "Are all the calculations done correctly? E.g. for the number of parameters of the feedforward blocks we have $2*4*1756^2*57=1406092416$? This shouldn't test whether the equation is correct, but rather if the answer is correctly calculated."
    ]
  },
  {
    "criterion": "Clarity and Formatting",
    "weight": 15.0,
    "checklist": [
      "Is the explanation of the parameter distribution clear and easy to follow?",
      "Does the response use clear and consistent notation?",
      "Does the response explain the reasoning behind the calculations?",
      "Does the response not add irrelevant/unnecessary details which are not asked in the question?",
      "Does the reponse explain why the feedforward blocks have the most parameters?",
      "Is the response easy to understand and follow?"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:
813