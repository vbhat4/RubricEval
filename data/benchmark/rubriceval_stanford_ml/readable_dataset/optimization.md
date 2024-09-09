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
  "Does the solution correctly define the online projected subgradient descent algorithm, including the update rule x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)]? (High importance)",
  "Are the assumptions clearly stated, such as \\mathcal{D} being convex, closed, non-empty, and bounded, and f_t being M-Lipschitz? (High importance)",
  "Is the learning rate \\eta correctly set as \\frac{D}{M}\\sqrt{\\frac{1}{T}}? (High importance)",
  "Does the solution correctly derive the regret bound \\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T} = o(T)? (High importance)",
  "Is the concept of sublinear regret explained, specifically that \\mathrm{Reg}_T(\\mathcal{A}) = o(T)? (Moderate importance)",
  "Does the solution correctly use the property of projections into convex bodies: \\| \\Pi_D[x'] - x \\|^2 \\leq \\| x' - x \\|^2? (Moderate importance)",
  "Is the telescoping sum correctly applied in the derivation of the regret bound? (Moderate importance)",
  "Does the solution correctly identify and use the subgradient \\nabla f_t(x_t) in the update rule? (Moderate importance)",
  "Is the inequality \\nabla_t \\cdot (x_t - x^*) \\leq \\frac{1}{2\\eta} (\\| x_t -x^* \\|^2 - \\| x_{t+1} - x^* \\|^2 ) + \\frac{\\eta}{2} \\| \\nabla_t \\|^2 correctly derived and used? (Moderate importance)",
  "Are the bounds \\| x - x' \\| \\leq D and \\| \\nabla f_t(x_t) \\| \\leq M correctly used in the proof? (Moderate importance)",
  "Is the final regret bound MD\\sqrt{T} correctly calculated and explained? (High importance)",
  "Does the solution provide a clear explanation of why the regret is sublinear and its implications? (Moderate importance)",
  "Is the role of the projection operator \\Pi_{\\mathcal{D}} in ensuring x_{t+1} \\in \\mathcal{D} clearly explained? (Moderate importance)",
  "Are all mathematical notations and symbols used correctly and consistently throughout the solution? (Low importance)",
  "Is the solution free from unnecessary or irrelevant details that do not contribute to the proof? (Low importance)",
  "Does the solution follow a logical structure, making it easy to follow the derivation of the regret bound? (Moderate importance)",
  "Is the solution concise and to the point, avoiding overly complex explanations? (Low importance)",
  "Does the solution demonstrate a clear understanding of online convex optimization and its challenges? (High importance)"
]
# <expert_checklist_time_sec>:

# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect definition of regret",
    "description": "The regret should be defined as the difference between the cumulative loss of the algorithm and the cumulative loss of the best fixed decision in hindsight. It should be expressed as \\( \\mathrm{Reg}_T(\\mathcal{A}) = \\sup_{\\{f_1, \\dots, f_T\\}} [\\sum_t^T f_t(x_t^{\\mathcal{A}}) - \\min_{x \\in \\mathcal{X}} \\sum_t^T f_t(x)] \\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing assumption of convexity of \\(\\mathcal{D}\\)",
    "description": "The solution should state that \\(\\mathcal{D}\\) is convex, closed, non-empty, and bounded. This is crucial for the projection step in the algorithm.",
    "delta_score": -0.5
  },
  {
    "error_name": "Missing Lipschitz assumption for \\(f_t\\)",
    "description": "The solution should assume that each function \\(f_t\\) is \\(M\\)-Lipschitz, i.e., \\(\\| \\nabla f_t(x_t) \\| \\leq M\\). This is necessary for bounding the regret.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect learning rate definition",
    "description": "The learning rate \\(\\eta\\) should be defined as \\(\\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}}\\). This is critical for achieving sublinear regret.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect projection step",
    "description": "The projection step should be defined as \\(x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)]\\), where \\(\\Pi_{\\mathcal{D}}\\) projects onto the feasible set \\(\\mathcal{D}\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect subgradient definition",
    "description": "A subgradient \\(\\nabla f_t(x_t)\\) should satisfy \\(f_t(x') - f_t(x_t) \\geq \\nabla f_t(x_t) \\cdot (x' - x_t)\\) for all \\(x' \\in \\mathcal{D}\\).",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect regret bound derivation",
    "description": "The regret bound should be derived as \\(\\mathrm{Reg}_T(\\mathcal{A}) \\leq D M \\sqrt{T}\\). The derivation should include the use of telescoping sums and the properties of projections.",
    "delta_score": -1
  },
  {
    "error_name": "Missing telescoping sum argument",
    "description": "The derivation should use a telescoping sum to simplify the expression for regret. This is a key step in the proof.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect use of projection property",
    "description": "The property \\(\\| \\Pi_{\\mathcal{D}}[x'] - x \\|^2 \\leq \\| x' - x \\|^2\\) should be used correctly in the derivation to ensure the regret bound holds.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect final regret expression",
    "description": "The final expression for regret should be \\(\\mathrm{Reg}_T(\\mathcal{A}) = MD\\sqrt{T}\\). Any deviation from this indicates a mistake in the derivation.",
    "delta_score": -0.5
  },
  {
    "error_name": "Unclear explanation or formatting",
    "description": "The explanation should be clear and well-formatted, making it easy for non-experts to follow the logic and steps of the proof.",
    "delta_score": -0.5
  },
  {
    "error_name": "Irrelevant or unnecessary details",
    "description": "The solution should avoid including irrelevant or unnecessary details that do not contribute to the understanding of the proof.",
    "delta_score": -0.5
  }
]
# <expert_list_error_rubric_time_sec>:

# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Understanding of Online Convex Optimization",
    "weight": 20.0,
    "checklist": [
      "Does the response correctly define the setting of online convex optimization, including the roles of the learner and nature? (High importance)",
      "Is the concept of regret clearly explained, including how it is calculated and its significance? (High importance)",
      "Does the response mention the goal of minimizing cumulative loss compared to the best fixed decision in hindsight? (Moderate importance)",
      "Is the explanation of sublinear regret and its importance in online learning provided? (Moderate importance)"
    ]
  },
  {
    "criterion": "Correctness of Assumptions",
    "weight": 15.0,
    "checklist": [
      "Are the assumptions about the convexity, closedness, and boundedness of the set \\(\\mathcal{D}\\) correctly stated? (High importance)",
      "Is the assumption that \\(f_t\\) is \\(M\\)-Lipschitz correctly stated and explained? (High importance)",
      "Is the learning rate \\(\\eta\\) correctly defined as \\(\\frac{D}{M}\\sqrt{\\frac{1}{T}}\\)? (Moderate importance)",
      "Does the response explain why these assumptions are necessary for proving the regret bound? (Moderate importance)"
    ]
  },
  {
    "criterion": "Proof of Sublinear Regret Bound",
    "weight": 30.0,
    "checklist": [
      "Is the online projected subgradient descent algorithm correctly described, including the update rule \\(x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)]\\)? (High importance)",
      "Does the proof correctly use the properties of convex functions and subgradients? (High importance)",
      "Is the telescoping sum technique correctly applied to derive the regret bound? (High importance)",
      "Does the response correctly conclude that the regret is \\(MD\\sqrt{T}\\) and explain why this is sublinear? (High importance)",
      "Are all mathematical steps clearly justified and logically connected? (Moderate importance)"
    ]
  },
  {
    "criterion": "Clarity and Structure of Explanation",
    "weight": 15.0,
    "checklist": [
      "Is the explanation of the proof clear and logically structured? (Moderate importance)",
      "Are mathematical notations and symbols used correctly and consistently? (Moderate importance)",
      "Does the response avoid unnecessary jargon and explain terms that may not be familiar to non-experts? (Moderate importance)",
      "Are key concepts and steps highlighted to aid understanding? (Moderate importance)"
    ]
  },
  {
    "criterion": "Mathematical Rigor and Precision",
    "weight": 20.0,
    "checklist": [
      "Are all mathematical expressions and equations correctly formatted and accurate? (High importance)",
      "Does the response demonstrate a strong understanding of mathematical concepts involved in the proof? (High importance)",
      "Is the use of inequalities and bounds in the proof precise and justified? (Moderate importance)",
      "Are any approximations or assumptions clearly stated and justified? (Moderate importance)"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:

# <expert_rubric>:
[
  {
    "criterion": "Understanding of Online Convex Optimization",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The response clearly defines the setting of online convex optimization, accurately describing the roles of the learner and nature. It provides a thorough explanation of regret, including its calculation and significance, and clearly states the goal of minimizing cumulative loss compared to the best fixed decision in hindsight. The concept of sublinear regret is explained in detail, including its importance in online learning.",
      "good": "The response correctly defines the setting of online convex optimization and explains the concept of regret, but may lack depth in explaining the significance of regret or the goal of minimizing cumulative loss. The explanation of sublinear regret is present but not detailed.",
      "fair": "The response provides a basic definition of online convex optimization and mentions regret, but lacks clarity or detail in explaining these concepts. The goal of minimizing cumulative loss or the concept of sublinear regret may be mentioned but not well explained.",
      "poor": "The response fails to accurately define online convex optimization or explain the concept of regret. It does not mention the goal of minimizing cumulative loss or the concept of sublinear regret."
    }
  },
  {
    "criterion": "Correctness of Assumptions",
    "weight": 15.0,
    "performance_to_description": {
      "excellent": "The response correctly states and explains all necessary assumptions, including the convexity, closedness, and boundedness of the set \\(\\mathcal{D}\\), the \\(M\\)-Lipschitz condition of \\(f_t\\), and the learning rate \\(\\eta = \\frac{D}{M}\\sqrt{\\frac{1}{T}}\\). It clearly explains why these assumptions are necessary for proving the regret bound.",
      "good": "The response correctly states most of the necessary assumptions and provides some explanation for their necessity. Minor details or explanations may be missing.",
      "fair": "The response mentions some assumptions but lacks clarity or completeness. It may not fully explain why these assumptions are necessary for the proof.",
      "poor": "The response fails to correctly state the necessary assumptions or explain their importance in the proof."
    }
  },
  {
    "criterion": "Proof of Sublinear Regret Bound",
    "weight": 30.0,
    "performance_to_description": {
      "excellent": "The response provides a complete and accurate description of the online projected subgradient descent algorithm, including the update rule \\(x_{t+1} = \\Pi_{\\mathcal{D}} [x_t - \\eta \\nabla f_t(x_t)]\\). The proof correctly uses properties of convex functions and subgradients, applies the telescoping sum technique, and concludes with the correct regret bound \\(MD\\sqrt{T}\\), explaining why this is sublinear. All mathematical steps are clearly justified and logically connected.",
      "good": "The response describes the online projected subgradient descent algorithm and provides a mostly correct proof of the sublinear regret bound. Some minor errors or omissions in the mathematical steps or justifications may be present.",
      "fair": "The response attempts to describe the algorithm and prove the regret bound but contains significant errors or omissions. The logical flow of the proof may be unclear or incomplete.",
      "poor": "The response fails to correctly describe the algorithm or prove the sublinear regret bound. It contains major errors or lacks a coherent proof structure."
    }
  },
  {
    "criterion": "Clarity and Structure of Explanation",
    "weight": 15.0,
    "performance_to_description": {
      "excellent": "The explanation is clear and logically structured, with correct and consistent use of mathematical notations and symbols. The response avoids unnecessary jargon and explains terms that may not be familiar to non-experts. Key concepts and steps are highlighted to aid understanding.",
      "good": "The explanation is generally clear and structured, with mostly correct use of mathematical notations. Some jargon may be present, and explanations of terms could be improved.",
      "fair": "The explanation lacks clarity or logical structure. Mathematical notations may be inconsistent or incorrect, and key concepts are not well highlighted.",
      "poor": "The explanation is unclear and poorly structured, with incorrect use of mathematical notations and a lack of explanation for key terms and concepts."
    }
  },
  {
    "criterion": "Mathematical Rigor and Precision",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "All mathematical expressions and equations are correctly formatted and accurate. The response demonstrates a strong understanding of the mathematical concepts involved in the proof, with precise and justified use of inequalities and bounds. Any approximations or assumptions are clearly stated and justified.",
      "good": "Most mathematical expressions are correct, and the response shows a good understanding of the concepts. Some minor errors or lack of precision in the use of inequalities or bounds may be present.",
      "fair": "The response contains several errors in mathematical expressions or lacks precision in the use of inequalities and bounds. The understanding of mathematical concepts may be incomplete.",
      "poor": "The response demonstrates a lack of understanding of the mathematical concepts, with numerous errors in expressions and unjustified use of inequalities and bounds."
    }
  }
]
# <expert_rubric_time_sec>:

