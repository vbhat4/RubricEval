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
The two main challenges are that (1) we have to ensure that the decisions are in the feasible set $\mathcal{D}$; and (2) the gradients of the function $f_t$ may not be defined.
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

```latex
\begin{equation}\label{eq:eta}
\eta \defeq \frac{D}{M}\sqrt{\frac{1}{T}}
\end{equation}
```

Then we can show that the regret for the above online projected subgradient descent is 

```latex
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
&\leq \sup_{\{f_1, \dots, f_T\}} [ \frac{1}{2\eta} D^2 + \frac{\eta}{2} \| \nabla_t \|^2T ] \label{eq:use_bound_D} & \text{\cref{eq:bound_D}}\\
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
  "Are the assumptions clearly stated, including that \\( \\mathcal{D} \\) is convex, closed, non-empty, and bounded, and that \\( f_t \\) is \\( M \\)-Lipschitz? (High importance)",
  "Is the concept of regret correctly defined and explained, including the formula for \\( \\mathrm{Reg}_T(\\mathcal{A}) \\)? (High importance)",
  "Is the subgradient \\( \\nabla f_t(x_t) \\) correctly defined and used in the context of convex functions? (Moderate importance)",
  "Is the projected subgradient descent algorithm correctly described, including the update rule \\( x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)] \\)? (High importance)",
  "Is the projection operation \\( \\Pi_{\\mathcal{D}} \\) correctly explained as projecting onto the feasible set \\( \\mathcal{D} \\)? (Moderate importance)",
  "Is the learning rate \\( \\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}} \\) correctly derived and justified? (High importance)",
  "Is the proof of sublinear regret \\( \\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T} = o(T) \\) correctly structured and logically sound? (High importance)",
  "Are the steps leading to the inequality \\( \\nabla_t \\cdot (x_t - x^*) \\leq \\frac{1}{2\\eta} (\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2 ) + \\frac{\\eta}{2} \\| \\nabla_t \\|^2 \\) clearly explained? (Moderate importance)",
  "Is the use of telescoping sums in the proof correctly applied and explained? (Moderate importance)",
  "Is the final regret bound \\( MD\\sqrt{T} \\) correctly derived from the assumptions and intermediate steps? (High importance)",
  "Is the importance of achieving sublinear regret \\( o(T) \\) clearly explained in the context of online learning? (Moderate importance)",
  "Are all mathematical notations and symbols used consistently and correctly throughout the explanation? (Moderate importance)",
  "Is the explanation free of unnecessary jargon and accessible to non-experts in convex optimization? (Moderate importance)",
  "Does the solution address potential edge cases or limitations of the projected subgradient descent method? (Low importance)",
  "Is the overall structure of the proof logical and easy to follow, with clear transitions between steps? (Moderate importance)"
]
# <expert_checklist_time_sec>:
1198.0
# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect definition of regret",
    "description": "The regret should be defined as the difference between the cumulative loss of the algorithm and the cumulative loss of the best fixed decision in hindsight. It should be expressed as \\( \\mathrm{Reg}_T(\\mathcal{A}) = \\sup_{\\{f_1, \\dots, f_T\\}} [\\sum_t^T f_t(x_t^{\\mathcal{A}}) - \\min_{x \\in \\mathcal{X}} \\sum_t^T f_t(x)] \\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing assumption of convex, closed, non-empty, and bounded set \\(\\mathcal{D}\\)",
    "description": "The solution must state that \\(\\mathcal{D}\\) is convex, closed, non-empty, and bounded, with a constant \\(D\\) such that \\(\\| x - x' \\| \\leq D\\) for all \\(x,x' \\in \\mathcal{D}\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing assumption of \\(M\\)-Lipschitz function",
    "description": "The solution must assume that each function \\(f_t\\) is \\(M\\)-Lipschitz, meaning \\(\\| \\nabla f_t(x_t) \\| \\leq M\\) for all \\(t\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect learning rate definition",
    "description": "The learning rate \\(\\eta\\) should be defined as \\(\\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}}\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect projection step in subgradient descent",
    "description": "The projection step should be \\(x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)]\\), where \\(\\Pi_{\\mathcal{D}}\\) projects onto the closest point in \\(\\mathcal{D}\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect use of subgradient inequality",
    "description": "The subgradient inequality should be used correctly: \\(f_t(x') - f_t(x_t) \\geq \\nabla f_t(x_t) \\cdot (x' - x_t)\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect telescoping sum application",
    "description": "The telescoping sum should be applied correctly to simplify the regret bound: \\(\\sum_t^T (\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2)\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect final regret bound",
    "description": "The final regret bound should be \\(\\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T}\\), showing sublinear growth.",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing explanation of sublinear regret",
    "description": "The solution should explain that sublinear regret means the average per-round regret goes to zero as \\(T\\) increases, i.e., \\(\\mathrm{Reg}_T(\\mathcal{A}) = o(T)\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Unclear or incomplete proof steps",
    "description": "The proof should clearly show each step leading to the regret bound, including assumptions and inequalities used.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect or missing projection property",
    "description": "The projection property \\(\\| \\Pi_{\\mathcal{D}}[x'] - x \\|^2 \\leq \\| x' - x \\|^2\\) should be used correctly in the proof.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect or missing notation",
    "description": "The notation should be consistent and correct throughout the solution, e.g., using \\(x_t^{\\mathcal{A}}\\) for the learner's decision.",
    "delta_score": -0.25
  },
  {
    "error_name": "Irrelevant or unnecessary details",
    "description": "The solution should avoid including irrelevant or unnecessary details that do not contribute to the proof or understanding of the regret bound.",
    "delta_score": -0.25
  },
  {
    "error_name": "Unclear answer or formatting",
    "description": "The answer should be clear and well-formatted, making it easy to follow for non-experts.",
    "delta_score": -0.25
  }
]
# <expert_list_error_rubric_time_sec>:
1283.0
# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Proof of Sublinear Regret Bound",
    "weight": 70.0,
    "checklist": [
      "The answer should correctly prove that the regret is sublinear without any mistakes. One possible proof is the following:\n1. Using convexity to upper bound the regret $\\mathrm{Reg}_T(\\mathcal{A}) \\defeq \\sup_{\\{f_1, \\dots, f_T\\}} [\\sum_t^T f_t(x_t) - f_t(x^*) ] \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [\\sum_t^T \nabla_t (x - x^*) ]$\n2. Analyzing the distance between consecutive decisions $\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2$\n3. Using properties of projections into convex bodies (i.e. $\\| \\Pi_D[x'] - x \\|^2 \\leq \\| x' - x \\|^2$ for any $x' \\in \\mathbb{R}^d$ and $x \\in \\mathcal{D}$) to bound $\nabla_t \\cdot (x_t - x^*) \\leq \frac{1}{2\\eta} (\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2 ) + \frac{\\eta}{2} \\| \nabla_t \\|^2$\n4. Telescoping sums to conclude $\\mathrm{Reg}_T \\leq \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [ \frac{1}{2\\eta} (\\| x_1 -x^* \\|^2 - \\| x_{T+1} - x^* \\|^2) + \frac{\\eta}{2} \\| \nabla_t \\|^2T ]$\n5. Applying the bounds on the decision set diameter $D$ (compactness) and subgradients $M$ (boundedness) to show that $\\mathrm{Reg}_T \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [ \frac{1}{2\\eta} D^2 + \frac{\\eta}{2}  M^2T ]$\n6. Plugging in the learning rate $\\eta = \frac{D}{M}\\sqrt{\frac{1}{T}}$ to conclude that $\\mathrm{Reg}_T \\leq MD\\sqrt{T} = o(T)$\nOther proofs may be possible, but they likely require similar steps and should be just as correct/complete. (Highest importance)",
      "Does the answer correctly state the final bound, it should be something along the lines of $\\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T} = o(T)$ where D is the diameter of the decision set and M is the bound on the subgradients. (Low importance)",
      "Does the answer clearly state what should be proven (Low importance).",
      "The answer fails to state the learning rate used in the proof (Low importance) despite that being useful. It should be something along the lines of $\\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}}$. (Medium inmportance)"
    ]
  },
  {
    "criterion": "Understanding of Online Convex Optimization and Assumptions",
    "weight": 20.0,
    "checklist": [
      "The proof doesn't state the assumptions on the functions $f_t$ necessary and used for the proof. The exact assumptions depend on the proof but they are typically convexity, and lipschitz continuity or boundedness of the norm of the subgradients. (High importance)",
      "The proof doesn't state the assumptions on the decision set necessary and used for the proof. The exact assumptions depend on the proof but they typically convexity, closedness and boundedness. (High importance)",
      "The assumptions are at reasonable rather than at odds with standard results in convex analysis. (Medium importance)",
      "Does the proof use really unnecessary assumptions? (Medium importance)"
    ]
  },
  {
    "criterion": "Follows the assignment instructions",
    "weight": 5.0,
    "checklist": [
      "Does the answer correctly focus on projected subgradient descent algorithm? (Moderate importance)",
      "When needed, does the answer incorporate improtant information from the assignment instructions? (Moderate importance)",
      "Does the answer avoid dicussing unimportant or irrelevant details? (Low importance)"
    ]
  },
  {
    "criterion": "Clarity and Proof Writing Skills",
    "weight": 5.0,
    "checklist": [
      "Are mathematical notations and symbols used consistently and correctly? (Low importance)",
      "Are complex concepts broken down into understandable parts? (Low importance)",
      "Are all mathematical steps clearly justified and logically connected? (Low importance)",
      "Are all the steps clear and discuss or reference the properties previously proved results? (Low importance)"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:
1480.0
# <expert_rubric>:
[
  {
    "criterion": "Proof of Sublinear Regret Bound",
    "weight": 70.0,
    "performance_to_description": {
      "excellent": "The proof correctly demonstrates that the regret is sublinear without any mistakes. It includes the following steps: 1) Using convexity to upper bound the regret as \\(\\mathrm{Reg}_T(\\mathcal{A}) \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [\\sum_t^T \\nabla_t (x - x^*) ]\\). 2) Analyzing the distance between consecutive decisions \\(\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2\\). 3) Using properties of projections into convex bodies to bound \\(\\nabla_t \\cdot (x_t - x^*) \\leq \\frac{1}{2\\eta} (\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2 ) + \\frac{\\eta}{2} \\| \\nabla_t \\|^2\\). 4) Telescoping sums to conclude \\(\\mathrm{Reg}_T \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [ \\frac{1}{2\\eta} (\\| x_1 -x^* \\|^2 - \\| x_{T+1} - x^* \\|^2) + \\frac{\\eta}{2} \\| \\nabla_t \\|^2T ]\\). 5) Applying bounds on the decision set diameter \\(D\\) and subgradients \\(M\\) to show \\(\\mathrm{Reg}_T \\leq \\sup_{\\{f_1, \\dots, f_T\\}} [ \\frac{1}{2\\eta} D^2 + \\frac{\\eta}{2}  M^2T ]\\). 6) Plugging in the learning rate \\(\\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}}\\) to conclude \\(\\mathrm{Reg}_T \\leq MD\\sqrt{T} = o(T)\\).",
      "good": "The proof is mostly correct but has a minor mistake, such as a small computational error in one of the steps or a slight misstatement of the final bound. For example, the proof might correctly follow the steps but miscalculate a term in the telescoping sum.",
      "fair": "The proof has a moderate mistake, such as missing one of the key steps or incorrectly applying a property of projections. For example, it might not correctly use the projection property \\(\\| \\Pi_D[x'] - x \\|^2 \\leq \\| x' - x \\|^2\\), leading to an incorrect bound.",
      "poor": "The proof contains major mistakes or multiple moderate ones, such as failing to demonstrate sublinear regret or incorrectly setting up the problem. For example, it might not use convexity to upper bound the regret or completely miss the telescoping sum step."
    }
  },
  {
    "criterion": "Understanding of Online Convex Optimization and Assumptions",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The proof clearly states all necessary assumptions: 1) The functions \\(f_t\\) are convex and \\(M\\)-Lipschitz, meaning \\(\\| \\nabla f_t(x_t) \\| \\leq M\\). 2) The decision set \\(\\mathcal{D}\\) is convex, closed, non-empty, and bounded, with a constant \\(D\\) such that \\(\\| x - x' \\| \\leq D\\) for all \\(x,x' \\in \\mathcal{D}\\). These assumptions are reasonable and align with standard results in convex analysis.",
      "good": "The proof states most of the necessary assumptions but omits a minor detail, such as not explicitly stating the boundedness of the decision set \\(\\mathcal{D}\\).",
      "fair": "The proof has a moderate mistake in stating assumptions, such as missing one of the key assumptions like the \\(M\\)-Lipschitz condition for \\(f_t\\).",
      "poor": "The proof contains major mistakes in stating assumptions, such as failing to mention the convexity of \\(f_t\\) or the decision set \\(\\mathcal{D}\\), or using assumptions that are at odds with standard results."
    }
  },
  {
    "criterion": "Follows the Assignment Instructions",
    "weight": 5.0,
    "performance_to_description": {
      "excellent": "The answer correctly focuses on the projected subgradient descent algorithm and incorporates all important information from the assignment instructions. It avoids discussing irrelevant details and stays on topic.",
      "good": "The answer mostly follows the assignment instructions but includes a minor irrelevant detail or slightly deviates from the main focus on projected subgradient descent.",
      "fair": "The answer somewhat follows the assignment instructions but includes a moderate amount of irrelevant information or fails to focus on the projected subgradient descent algorithm.",
      "poor": "The answer does not follow the assignment instructions, includes many irrelevant details, or fails to address the projected subgradient descent algorithm."
    }
  },
  {
    "criterion": "Clarity and Proof Writing Skills",
    "weight": 5.0,
    "performance_to_description": {
      "excellent": "The proof is clear, concise, and easy to follow. Mathematical notations and symbols are used consistently and correctly. Complex concepts are broken down into understandable parts, and all mathematical steps are clearly justified and logically connected.",
      "good": "The proof is mostly clear and concise, with a minor issue such as a slight inconsistency in notation or a small gap in the logical flow.",
      "fair": "The proof is somewhat clear but has a moderate issue, such as inconsistent notation or a confusing presentation of steps that makes it difficult to follow.",
      "poor": "The proof is unclear, verbose, or difficult to follow. It may include many irrelevant details, miss crucial points, or present information in a disorganized manner."
    }
  }
]
# <expert_rubric_time_sec>:
nan
