# <category>:
Stats & ML
# <instruction>:

In online convex optimization the setting is the following:
- the learner makes a decision $x_t \in \mathcal{D} \subseteq \mathbb{R}^d$
- nature gives a convex function $f_t : \mathcal{D} \to \mathbb{R}$
- the learner incurs a loss $f_t(x_t)$ based on the selected decision
- the learner uses this feedback to update its decision-making strategy and we start a new round

The goal of the learner is to minimize the cumulative loss over the entire sequence of rounds, compared to the best \textit{fixed} decision in hindsight. 
The regret of a learning algorithm mapping previous $\mathcal{A}: \{f_1, \dots, f_{t_1}\} \to x_t^{\mathcal{A}}$ losses to a current prediction is the worst loss over possible function from nature.

```latex
\begin{equation}
\mathrm{Reg}_T(\mathcal{A}) = \sup_{\{f_1, \dots, f_T\}} [\sum_t^T f_t(x_t^{\mathcal{A}}) - \min_{x \in \mathcal{X}} \sum_t^T f_t(x)]
\end{equation}
```

Note that the regret is always positive. 
Ideally, the average per-round regret should go down to zero as the number of rounds increases, meaning that you tend to the best fixed strategy.
Slighlty more formally, we would like the regret to be sublinear as a function of T, i.e, $\mathrm{Reg}_T(\mathcal{A}) = o(T)$.

Prove that under reasonable assumptions the online projected subgradient descent algorithm achieves sublinear regret. State your assumptions and prove the regret bound.
# <expert_solution>:
A very natural and simple algorithm for online convex optimization is to run online gradient descent.
The two challenges main challenges are that (1) we have to ensure that the decisions are in the feasible set $\mathcal{D}$; and (2) the gradients of the function $f_t$ may not be defined. 
To alleviate these issues we thus use online projected subgradient descent. 
As a reminder, a subgradient $\nabla f_t(x_t)$ of a convex function $f_t$ at $x_t$ satisfies, for all $x' \in \mathcal{D}$

```latex
\begin{equation}
f_t(x') - f_t(x_t) \geq \nabla f_t(x_t) \cdot (c' - c_t).
\end{equation}
```

A subgradient always exists for a convex function but it may not be unique. 
Given a subgradient, we can then perform projected subgradient descent to ensure that $x_{t+1} \in \mathcal{D}$.

```latex
\begin{equation}
x_{t+1} = \Pi_{\mathcal{D}} [x_t - \eta \nabla f_t(x_t)]
\end{equation}
```

where $\eta$ is the learning rate and $\Pi_{\mathcal{D}}$ projects the result back to the closest point (under L2 norm) in $\mathcal{D}$, i.e.,

```latex
\begin{equation}
\Pi_{\mathcal{D}} [x] \in \argmin_{x' \in \mathcal{D}} \| x - x' \|
\end{equation}
```

which ensures that $x_{t+1} \in \mathcal{D}$ as desired.

Now that we defined the online projected subgradient descent, let's bound the regret of such algorithm.


Assume that $\mathcal{D}$ is convex, closed, non-empty and bounded. In particular, there exists a constant $D$  s.t. for all $x,x' \in \mathcal{D}$ we have

```latex
\begin{equation}\label{eq:bound_D}
    \| x - x' \| \leq D
\end{equation}
```

also assume that $f_t$ is $M$-lipschitz. In particular, as it is convex we have that for all $t$

```latex
\begin{equation}\label{eq:bound_M}
    \| \nabla f_t(x_t) \| \leq M
\end{equation}
```

And finally set the learning rate

```
\begin{equation}\label{eq:eta}
\eta \defeq \frac{D}{M}\sqrt{\frac{1}{T}}
\end{equation}
```

Then we can show that the regret for the above online projected subgradient descent is 

```
\begin{equation}
\mathrm{Reg}_T(\mathcal{A}) \leq D M \sqrt{T} = o(T)
\end{equation}
```

i.e. online projected sugradient descent achieves the desired sublinear regret.



To show that this is the case let's denote $\nabla_t \defeq\nabla f_t(x_t) $, $x_t \defeq x_t^{\mathcal{A}}$ for notational convenience, and $x^* \defeq \argmin_{x \in \mathcal{X}} \sum_t^T f_t(x)$ (which exists since $\mathcal{D}$ is closed and convex). 
By convexity we have 

```latex
\begin{align}
\mathrm{Reg}_T(\mathcal{A}) &\defeq \sup_{\{f_1, \dots, f_T\}} [\sum_t^T f_t(x_t) - f_t(x^*) ] \\
&\leq \sup_{\{f_1, \dots, f_T\}} [\sum_t^T \nabla_t (x - x^*) ]  \label{eq:oco_upper}
\end{align}
```

now note that 

```latex
\begin{align}
\| x_t -x^* \|^2 - \| x_{t+1} - x^* \|^2 
&= \| x_t -x^* \|^2 - \| \Pi_{\mathcal{D}} [x_t - \eta \nabla_t] - x^* \|^2  \\
&\geq \| x_t -x^* \|^2 - \| x_t - \eta \nabla_t - x^* \|^2  \\
&= 2 \eta \nabla_t \cdot (x_t - x^*) - \eta^2 \| \nabla_t \|^2  \\
\nabla_t \cdot (x_t - x^*) &\leq \frac{1}{2\eta} (\| x_t -x^* \|^2 - \| x_{t+1} - x^* \|^2 ) + \frac{\eta}{2} \| \nabla_t \|^2     \label{eq:oco_ineq}
\end{align}
```

where the first inequality uses the following property of projections into convex bodies: $\| \Pi_D[x'] - x \|^2 \leq \| x' - x \|^2$ for any $x' \in \mathbb{R}^d$ and $x \in \mathcal{D}$.

Putting back \cref{eq:oco_ineq} into \cref{eq:oco_upper} we get

```latex
\begin{align}
\mathrm{Reg}_T &= \sup_{\{f_1, \dots, f_T\}} [\sum_t^T f_t(x_t) - f_t(x^*) ] \\
&\leq \sup_{\{f_1, \dots, f_T\}} [\sum_t^T (\frac{1}{2\eta} (\| x_t -x^* \|^2 - \| x_{t+1} - x^* \|^2 ) + \frac{\eta}{2} \| \nabla_t \|^2) ] \\
&\leq \sup_{\{f_1, \dots, f_T\}} [ \frac{1}{2\eta} (\| x_1 -x^* \|^2 - \| x_{T+1} - x^* \|^2) + \frac{\eta}{2} \| \nabla_t \|^2T ] \label{eq:telescope}\\
&\leq \sup_{\{f_1, \dots, f_T\}} [ \frac{1}{2\eta} D + \frac{\eta}{2} \| \nabla_t \|^2T ] \label{eq:use_bound_D} & \text{\cref{eq:bound_D}}\\
&\leq \sup_{\{f_1, \dots, f_T\}} [ \frac{1}{2\eta} D^2 + \frac{\eta}{2}  M^2T ] \label{eq:use_bound_M} & \text{\cref{eq:bound_M}}\\
%
&=  \frac{M\sqrt{T}}{2D} D^2 + \frac{1}{2} \frac{D}{M}\sqrt{\frac{1}{T}}  M^2T  & \text{\cref{eq:eta}}\\
%
&= MD\sqrt{T}   
\end{align}
```

as desired where \cref{eq:telescope} uses a telescoping sum.
# <expert_checklist>:
[
  "Does the proof clearly state all necessary assumptions for the online projected subgradient descent algorithm to achieve sublinear regret?",
  "Is the regret bound derived correctly and does it follow logically from the stated assumptions?",
  "Are the mathematical notations and equations used in the proof consistent and correctly formatted?",
  "Does the proof include a step-by-step explanation of how the online projected subgradient descent algorithm updates its decision-making strategy?",
  "Is there a clear explanation of how the cumulative loss is minimized compared to the best fixed decision in hindsight?",
  "Does the proof demonstrate that the average per-round regret tends to zero as the number of rounds increases?",
  "Are there any gaps or missing steps in the logical flow of the proof that could affect its validity?",
  "Is the concept of sublinear regret explained and applied correctly within the context of the proof?",
  "Does the proof consider the worst-case scenario for the regret bound, as required by the definition of regret?",
  "Is the language and terminology used in the proof precise and appropriate for a mathematical audience?"
]
# <expert_checklist_time_sec>:

# <expert_list_error_rubric>:
[
  {
    "error_name": "Missing Assumptions",
    "description": "The response should clearly state the assumptions necessary for the proof, including the convexity and compactness of the decision set, Lipschitz continuity of the functions, and bounded subgradients. Check if these assumptions are explicitly mentioned.",
    "delta_score": -1.5
  },
  {
    "error_name": "Incorrect Update Rule",
    "description": "The update rule for the online projected subgradient descent should be correctly stated as \\( x_{t+1} = \\Pi_{\\mathcal{D}}(x_t - \\eta_t g_t) \\). Verify if the LLM's output correctly describes this update rule.",
    "delta_score": -2
  },
  {
    "error_name": "Incorrect Regret Definition",
    "description": "The regret should be defined as \\( \\mathrm{Reg}_T(\\mathcal{A}) = \\sum_{t=1}^T f_t(x_t) - \\min_{x \\in \\mathcal{D}} \\sum_{t=1}^T f_t(x) \\). Ensure the LLM's output uses this correct definition.",
    "delta_score": -1
  },
  {
    "error_name": "Missing or Incorrect Regret Bound Proof",
    "description": "The proof should demonstrate that the regret is sublinear, specifically \\( O(D^2 \\sqrt{T}) \\). Check if the proof is complete and correctly shows this bound.",
    "delta_score": -2.5
  },
  {
    "error_name": "Incorrect Step Size Selection",
    "description": "The step size \\( \\eta_t \\) should be chosen as \\( \\frac{D}{G\\sqrt{t}} \\). Verify if the LLM's output specifies this step size correctly.",
    "delta_score": -1.5
  },
  {
    "error_name": "Lack of Clarity in Explanation",
    "description": "The explanation should be clear and logically structured, making it easy for non-experts to follow. Check if the response is unnecessarily complex or lacks logical flow.",
    "delta_score": -1
  }
]
# <expert_list_error_rubric_time_sec>:

# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Understanding of Online Convex Optimization",
    "weight": 25.0,
    "checklist": [
      "Does the response clearly define online convex optimization?",
      "Is the role of the learner and nature in the optimization process explained?",
      "Are the concepts of decision-making and loss in this context accurately described?",
      "Is the goal of minimizing cumulative loss compared to the best fixed decision in hindsight mentioned?",
      "Is the concept of regret and its significance in online learning explained?"
    ]
  },
  {
    "criterion": "Explanation of Online Projected Subgradient Descent",
    "weight": 25.0,
    "checklist": [
      "Is the online projected subgradient descent algorithm correctly described?",
      "Are the challenges of ensuring decisions are in the feasible set and undefined gradients addressed?",
      "Is the concept of subgradients and their role in the algorithm explained?",
      "Is the update rule for the algorithm accurately stated?",
      "Is the projection operation and its purpose clearly explained?"
    ]
  },
  {
    "criterion": "Mathematical Rigor and Assumptions",
    "weight": 20.0,
    "checklist": [
      "Are the assumptions about the feasible set \\(\\mathcal{D}\\) clearly stated?",
      "Is the Lipschitz condition for the functions \\(f_t\\) mentioned and explained?",
      "Is the learning rate \\(\\eta\\) correctly defined and justified?",
      "Are the mathematical derivations leading to the regret bound clearly presented?",
      "Is the final regret bound \\(\\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T}\\) correctly derived?"
    ]
  },
  {
    "criterion": "Clarity and Structure",
    "weight": 15.0,
    "checklist": [
      "Is the response well-organized and logically structured?",
      "Are complex concepts broken down into understandable parts?",
      "Is the use of mathematical notation consistent and clear?",
      "Are transitions between different parts of the explanation smooth?",
      "Is the language accessible to non-experts?"
    ]
  },
  {
    "criterion": "Use of Examples and Illustrations",
    "weight": 15.0,
    "checklist": [
      "Are examples provided to illustrate key concepts?",
      "Do the examples effectively clarify the explanation?",
      "Is there a visual or intuitive explanation of the projection operation?",
      "Are hypothetical scenarios used to explain the concept of regret?",
      "Do the examples help in understanding the sublinear nature of the regret?"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:

