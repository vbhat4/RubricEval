# <category>:
Stats & ML
# <instruction>:

The Least Squares objective is denoted by:

\begin{align*}
    \argmin_{w} \quad ( Y - Xw)^T(Y - Xw)
\end{align*}

where $Y \in \mathbb{R}^n$, $w \in \mathbb{R}^d$ are the parameters, and $X \in \mathbb{R}^{n \times d}$ is the feature matrix.\\

The Ridge-regression regularized objective is:
\begin{align*}
\argmin_{w} \quad (Y - Xw)^T(Y - Xw) + \lambda w^Tw
\end{align*}

Show that $||w^*_r ||_2^2 \leq ||w^*_l ||_2^2$. Where $w^*_l$ and $w^*_r$ respectively denote the optimal solution for the least squares objective and the ridge regression objective.\\
# <expert_solution>:
Let $x$ be a fixed design matrix and $y$ a fixed response vector. Let $V \Sigma U^{T}$ be the SVD of $x$ and $U \Lambda U^{T}$ be the eigendecomposition of $x^{T} x$, where it is important to recall that $\Lambda$ has no negative entries, i.e. $x^{T} x$ is positive semidefinite, and also $\Sigma^2 = \Lambda$. Recall that $U$ and $V$ are orthogonal matrices, and that $\Lambda$ and $\Sigma$ are diagonal.

The ridge estimator with penalty parameter $\lambda$ has coefficients
$$\hat{\beta}_{\text{ridge}}^{\lambda} = (x^{T} x + \lambda I)^{-1} x^{T} y.$$
We can compute its squared norm by plugging in the factorization above. 
$$\|\hat{\beta}_{\text{ridge}}^{\lambda}\|^{2} = (\hat{\beta}_{\text{ridge}}^{\lambda})^{T}\hat{\beta}_{\text{ridge}}^{\lambda} = y^{T} x (x^{T} x + \lambda I)^{-2} x^{T} y$$
$$ = y^{T} V \Sigma U^{T} (U \Lambda U^{T} + \lambda I)^{-2} U \Sigma V^{T} y$$
$$ = y^{T} V \Sigma U^{T} (U (\Lambda + \lambda I) U^{T} )^{-2} U \Sigma V^{T} y$$
$$ = y^{T} V \Sigma U^{T} (U (\Lambda + \lambda I)^{-2} U^{T} ) U \Sigma V^{T} y$$
$$ = y^{T} V \frac{\Sigma^{2}}{(\Lambda + \lambda I)^{2}} V^{T} y.$$
I'm abusing notation a bit in this last inequality by writing these matrices as a fraction, but I really just mean pointwise division of their elements since all of the matrices involved are diagonal. Next, substituting in that the eigenvalues of $x^{T} x$ are the squared singular values of $x$, we have
$$= y^{T} V \frac{ \Lambda}{(\Lambda + \lambda I)^{2}} V^{T} y.$$
Denote $V^{T} y$ by $w$. We have
$$y^{T} V \frac{ \Lambda}{(\Lambda + \lambda I)^{2}} V^{T} y = w^{T} \frac{ \Lambda}{(\Lambda + \lambda I)^{2}} w$$
$$ = \sum_{i=1}^{p} \frac{\Lambda_{i,i}}{(\Lambda_{i,i} + \lambda)^{2}} w_{i}^{2}.$$

Therefore, for two penalty parameters, $\lambda_{1} \leq \lambda_{2}$, we have
$$\|\hat{\beta}_{\text{ridge}}^{\lambda_{2}}\|^{2} \leq \|\hat{\beta}_{\text{ridge}}^{\lambda_{1}}\|^{2}  \iff \sum_{i=1}^{p} \frac{\Lambda_{i,i}}{(\Lambda_{i,i} + \lambda_{2})^{2}} w_{i}^{2} \leq \sum_{i=1}^{p} \frac{\Lambda_{i,i}}{(\Lambda_{i,i} + \lambda_{1})^{2}} w_{i}^{2}.$$
This last inequality is true because $\frac{\Lambda_{i,i}}{(\Lambda_{i,i} + \lambda_{2})^{2}} \leq \frac{\Lambda_{i,i}}{(\Lambda_{i,i} + \lambda_{1})^{2}}$ for each $i \in \{1,...,p\}$, which follows because $\Lambda_{i,i} \geq 0$.
# <expert_checklist>:
[
  "Does the solution correctly identify and define the least squares and ridge regression objectives?",
  "Is the mathematical derivation for showing that ||w^*_r ||_2^2 \\leq ||w^*_l ||_2^2 correct and logically sound?",
  "Are all mathematical steps clearly explained and justified in the derivation?",
  "Does the solution correctly interpret the role of the regularization term \\lambda w^Tw in the ridge regression objective?",
  "Is the comparison between the optimal solutions of least squares and ridge regression clearly articulated?",
  "Are any assumptions made during the derivation explicitly stated and reasonable?",
  "Does the solution include any relevant examples or illustrations to enhance understanding?",
  "Is the final conclusion about the relationship between ||w^*_r ||_2^2 and ||w^*_l ||_2^2 clearly stated and supported by the derivation?",
  "Are there any errors or omissions in the mathematical notation or logic?",
  "Is the overall explanation coherent and easy to follow for someone with a background in linear algebra and optimization?"
]
# <expert_checklist_time_sec>:

# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect least squares solution",
    "description": "The least squares solution should be \\(w^*_l = (X^TX)^{-1}X^TY\\). Check if the LLM's output correctly states this formula. Incorrect responses might omit the inverse or transpose operations.",
    "delta_score": -1.5
  },
  {
    "error_name": "Incorrect ridge regression solution",
    "description": "The ridge regression solution should be \\(w^*_r = (X^TX + \\lambda I)^{-1}X^TY\\). Verify that the LLM's output includes the regularization term \\(\\lambda I\\). Missing or incorrect terms should be noted.",
    "delta_score": -1.5
  },
  {
    "error_name": "Misinterpretation of regularization effect",
    "description": "The response should explain that the regularization term \\(\\lambda w^Tw\\) in ridge regression shrinks the coefficients, leading to a smaller \\(L_2\\) norm. Check for a clear explanation of how regularization affects the solution.",
    "delta_score": -1
  },
  {
    "error_name": "Failure to compare eigenvalues",
    "description": "The explanation should include a comparison of eigenvalues between \\((X^TX)^{-1}\\) and \\((X^TX + \\lambda I)^{-1}\\), highlighting why the ridge regression solution is smaller. Look for a discussion on eigenvalue shrinkage.",
    "delta_score": -1
  },
  {
    "error_name": "Lack of conclusion on norm comparison",
    "description": "The response should conclude that \\(||w^*_r ||_2^2 \\leq ||w^*_l ||_2^2\\) due to the regularization effect. Ensure the conclusion is explicitly stated and logically follows from the explanation.",
    "delta_score": -1
  }
]
# <expert_list_error_rubric_time_sec>:

# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Mathematical Rigor",
    "weight": 30.0,
    "checklist": [
      "Does the response correctly apply the singular value decomposition (SVD) of the design matrix X?",
      "Is the eigendecomposition of X^TX accurately described and used?",
      "Are the properties of orthogonal matrices U and V correctly utilized?",
      "Is the positive semidefiniteness of X^TX mentioned and explained?",
      "Does the response correctly derive the ridge regression solution using the penalty parameter \u03bb?",
      "Is the comparison between the norms of the least squares and ridge regression solutions mathematically sound?",
      "Are the steps in the derivation logically sequenced and clearly explained?"
    ]
  },
  {
    "criterion": "Clarity and Explanation",
    "weight": 25.0,
    "checklist": [
      "Is the explanation of the mathematical steps clear and easy to follow?",
      "Are technical terms and symbols adequately defined for a non-expert audience?",
      "Does the response avoid unnecessary jargon and complexity?",
      "Are the key concepts of least squares and ridge regression clearly distinguished?",
      "Is the reasoning behind each mathematical step explained in simple terms?"
    ]
  },
  {
    "criterion": "Correctness of Solution",
    "weight": 30.0,
    "checklist": [
      "Does the response correctly state the least squares and ridge regression objectives?",
      "Is the derivation of the ridge regression solution accurate?",
      "Are the conditions under which ||w*_r||_2^2 <= ||w*_l||_2^2 correctly identified?",
      "Does the response correctly conclude that the ridge regression norm is less than or equal to the least squares norm?",
      "Are any assumptions made in the derivation clearly stated and justified?"
    ]
  },
  {
    "criterion": "Use of Examples",
    "weight": 15.0,
    "checklist": [
      "Are examples or analogies used to illustrate complex concepts?",
      "Do the examples help clarify the mathematical derivation?",
      "Are the examples relevant and accurately reflect the mathematical principles discussed?",
      "Is there a balance between examples and theoretical explanation?",
      "Do the examples enhance the understanding of the ridge regression and least squares comparison?"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:

