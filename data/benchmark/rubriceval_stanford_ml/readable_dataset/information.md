# <category>:
Stats & ML
# <instruction>:
Shannon's information theory has been a cornerstone in the development of modern communication systems. However, it is now used much more broadly in machine learning, statistics, and other fields. One issue with Shannon's theory is that it studies the existence of information rather than its quality or usability. Specifically, Shannon’s information misses two key aspects of the downstream task. First, it is agnostic to how the actions or predictions from the decision maker (DM) will be evaluated. Second, it does not depend on the computational constraints of the DM, e.g., a polynomial-time complexity algorithm or a linear predictor. As a result, its applications in decision-making or machine learning can give rise to false claims. For example, the data processing inequality would imply that features of deeper layers in a neural network are less informative than earlier ones. Similarly, as mutual information is invariant to bijections, encrypting a message should not alter its informativeness for any DM, which is, of course, not true in practice.

A potential solution to this problem is to generalize Shannon's information theory by taking a utilitarian perspective that considers both the potential actions (i.e., its computational constraints) of the decision maker and the loss function that will be used to evaluate the actions. We call this utilitarian information theory. Here's an explanation of it.

## Background: Predictions, Bayes Risk, and Bayes Decision Theory with Computationally Bounded Agents

In standard Bayes decision theory, a decision maker (DM) wants to minimize some expected loss $\ell$ when given samples from an underlying distribution $\mathbb{P}(Y)$.
Specifically, the DM decides on an action $a \in \mathcal{A}$ and then incurs an expected loss $\operatorname{E}_{\mathbb{P}(Y)}[\ell(Y,a)]$. In many scenarios, the DM can choose the optimal action after seeing outcomes from a related random variable $X$. The DM then incurs the conditional expected loss $\operatorname{E}_{\mathbb{P}(X)}\left[\inf_{a \in \mathcal{A}} \operatorname{E}_{\mathbb{P}(Y \mid X)}[\ell(Y,a)]\right]$.
Real DMs, however, are often computationally bounded, which raises the question: what if finding the optimal action $a$ is computationally intractable? What if not every action can be taken for any outcome $X$? What if the DM is constrained to taking related actions? These questions about computationally bounded DMs are important in practice but are rarely discussed in Bayes decision theory. 
To answer these questions, it is useful to take a predictive perspective, whereby the DM has to choose a predictor $f$ from inputs $X$ to actions in $\mathcal{A}$.
The conditional expected loss can then be equivalently rewritten as $\inf_{f \in \mathcal{U}} \operatorname{E}_{\mathbb{P}(Y,X)}[\ell(Y,f(X))]$, where the infimum is over all possible predictors $\mathcal{U}=\mathcal{A}^\mathcal{X}$. The previous questions then simply amount to asking what if the DM is restricted to selecting from a subset of predictors $\mathcal{V} \subseteq \mathcal{U}$ that are usually "simple".
The resulting loss incurred by the bounded DM then simply corresponds to the Bayes risk for predictive family (also called hypothesis class) $\mathcal{V}$ and loss $\ell$, i.e., $\inf_{f \in \mathcal{V}} \operatorname{E}_{\mathbb{P}(Y,X)}[\ell(Y,f(X))]$, which is well studied in statistical learning theory.
The predictors that achieve the Bayes risk are then called Bayes predictors $\mathcal{V}_{p,\ell}^*$.
Our framework of Utilitarian Information Theory is the study of how useful some information is for this restricted DM whose actions are evaluated by general $\ell$.

## Utilitarian Information Theory

The intuition behind Shannon's conditional entropy is to measure how much uncertainty one has about some random variable $Y$ if one knows about another random variable $X$, i.e., $\operatorname{H}(Y \mid X) \defeq \inf_{f \in \mathcal{U}} \operatorname{E}_{\mathbb{P}(Y,X)}[\ell(Y,q(X))] = \inf_{f \in \mathcal{U}} \operatorname{E}_{\mathbb{P}(Y,X)}[\ell(Y,q(X))]$.
From the perspective of our restricted DM, it is then very natural to measure such uncertainty by the minimal loss that she would incur when predicting $Y$ given $X$, i.e., the Bayes risk. 
Our \textit{utilitarian conditional entropy} is then naturally defined as $\operatorname{H}_{\ell,\mathcal{V}}(Y \mid X) \defeq \inf_{f \in \mathcal{V}} \operatorname{E}_{\mathbb{P}(Y,X)}[\ell(Y,f(X))]$.

Similarly, Shannon's (marginal) entropy $\operatorname{H}(Y)$ measures how much uncertainty one has about a random variable $Y$ prior to seeing any other information.
Our \textit{utilitarian (marginal) entropy} is then naturally defined by the uncertainty the DM has about $Y$ without additional knowledge, i.e., the unconditional Bayes risk $\operatorname{H}_{\ell}(Y) \defeq \inf_{a \in \mathcal{A}}\operatorname{E}_{\mathbb{P}(Y)}[\ell(Y,a)]$. Note that the DM's boundedness only enters decision theory when she has access to additional knowledge $X$, so our marginal entropy does not depend on $\mathcal{V}$.

The intuition behind Shannon's mutual information is to measure how much the knowledge of a random variable $X$ decreases your uncertainty about another $Y$, i.e., $\operatorname{I}(Y;X) = \operatorname{H}(Y) - \operatorname{H}(Y \mid X)$.
The same construction naturally gives rise to a notion of \textit{utilitarian information} $\operatorname{I}_{\ell,\mathcal{V}}(X;Y) = \operatorname{H}_{\ell}(Y) - \operatorname{H}_{\ell,\mathcal{V}}(Y \mid X)$, which measures the decrease in uncertainty for the constrained DM evaluated by $\ell$, i.e., how much better the DM can predict $Y$ if she has access to $X$.

Formally:

\begin{definition} [Action Predictive Family] Let $\Omega_{\text{action}} = \lbrace f: \mathcal{X} \cup \lbrace \emptyset \rbrace \to \mathcal{A} \rbrace$, we say that $\mathcal{V} \subset \Omega_{\text{action}}$ is a predictive family (for $\mathcal{A}$) if it satisfies
\begin{align}
\forall f \in \mathcal{V}, \forall a \in \mathrm{range}(f), \exists f' \in \mathcal{V}, \forall x\in\mathcal{X}, f'(x) = a, f'(\emptyset) = a
\end{align}
\end{definition}

Predictive families essentially mean that there always exists a function that can predict any constant value for any input. This is a very mild assumption that is satisfied by most hypothesis classes in practice. We also call this property Optional Ignorance.

\begin{definition} [Utilitarian Entropy] For any loss function $\ell$ and predictive family $\mathcal{V}$, the utilitarian entropy of a random variable $Y$ (potentially conditioned on another random variable $X$) is defined as
\begin{align}
    H_{\ell, \mathcal{V}}(Y) &:= \inf_{f \in \mathcal{V}} \mathbb{E}\left[ \ell(Y, f[\emptyset]) \right] \\
    H_{\ell, \mathcal{V}}(Y|X) &:= \inf_{f \in \mathcal{V}} \mathbb{E}\left[ \ell(Y, f[X]) \right]
\end{align}
\end{definition}
Intuitively, the Utilitarian Entropy is the Bayes risk, i.e., the risk of the best prediction function.

\begin{definition}[Utilitarian Mutual Information] For any loss function $\ell$, predictive family $\mathcal{V}$, and random variables $X$ and $Y$, the utilitarian mutual information is defined as 
\begin{align} 
I_{\ell, \mathcal{V}}(X \to Y) := H_{\ell, \mathcal{V}} (Y) - H_{\ell, \mathcal{V}}(Y \mid X)
\end{align} 
\end{definition} 

## Questions

For each of the following statements, prove whether they are true or false. The proof should be detailed.

1. Utilitarian information is always symmetric, i.e., $\operatorname{I}_{\ell,\mathcal{V}}(X; Y) = \operatorname{I}_{\ell,\mathcal{V}}(Y; X)$.
2. Utilitarian information is always non-negative.
3. Utilitarian information recovers Shannon's information theory for some choices of $\mathcal{V}$ and $\ell$.
4. Utilitarian information satisfies the Data Processing Inequality, i.e., the Markov Chain $Z \to X \to Y$ implies $\operatorname{I}_{\ell,\mathcal{V}}(X; Y) \leq \operatorname{I}_{\ell,\mathcal{V}}(Z; Y)$.
5. Utilitarian information between two random variables $X$ and $Y$ that are independent is always zero.
6. Utilitarian information is always invariant to bijections, i.e., $\operatorname{I}_{\ell,\mathcal{V}}(X; Y) = \operatorname{I}_{\ell,\mathcal{V}}(b_X(X); Y)$ for any bijection $b_X$.

For each of those questions also provide an explanation / illustration of why that makes sense for machine learning.
# <expert_solution>:
## Symmetry

**False.** Utilitarian information is not symmetric.

*Counterexample:*  
Consider random variables \(X\) taking values in \(\{0,1,2\}\) and \(Y\) taking values in \(\{0,2\}\). Assume the following conditional probabilities:  
- \(\mathbb{P}(Y = 0 \mid X = 0) = 1\)  
- \(\mathbb{P}(Y = 2 \mid X = 1) = 1\)  
- \(\mathbb{P}(Y = 2 \mid X = 2) = 1\)  
And the marginal probabilities:  
- \(\mathbb{P}(X = 0) = 0.5\)  
- \(\mathbb{P}(X = 1) = 0.25\)  
- \(\mathbb{P}(X = 2) = 0.25\)  

Consider the Mean Absolute Error (MAE) as the loss function \(\ell\) and the predictive family \(\mathcal{V}\) consisting of the following functions:  
- \(c_0(\emptyset) = 0\), \(c_1(\emptyset) = 1\), \(c_2(\emptyset) = 2\)  
- \(f(X = 0) = 0, f(X = 1) = 2, f(X = 2) = 2\)  
- \(g(\cdot) = 1\)  

The conditional entropy \(\operatorname{H}_{\ell, \mathcal{V}}(Y \mid X) = 0\) because the optimal predictor \(f\) can perfectly predict \(Y\) given \(X\). However, \(\operatorname{H}_{\ell, \mathcal{V}}(X \mid Y) = 0.25\) since \(f\) cannot predict \(X = 1\) when conditioned on \(Y = 2\). The marginal entropies are:  
- \(\operatorname{H}_{\ell, \mathcal{V}}(Y) = 1\)  
- \(\operatorname{H}_{\ell, \mathcal{V}}(X) = 0.75\)  

Therefore, the utilitarian mutual information is:  
\[
\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = 1 - 0 = 1, \quad \text{but} \quad \operatorname{I}_{\ell, \mathcal{V}}(Y; X) = 0.75 - 0.25 = 0.5.
\]
Since \(\operatorname{I}_{\ell, \mathcal{V}}(X; Y) \neq \operatorname{I}_{\ell, \mathcal{V}}(Y; X)\), utilitarian information is not symmetric.

*Machine Learning Perspective:* In practical machine learning, predicting a target variable from features is often easier than the reverse. For example, in classification tasks, predicting the class label (a lower-dimensional output) from features (often high-dimensional and complex) is generally more straightforward than predicting the features given the class label. This asymmetry is captured by utilitarian information, reflecting the practical differences in predictive power between features and targets.

## Non-Negativity

**True.** Utilitarian information is always non-negative.

*Proof:*  
By definition, a predictive family \(\mathcal{V}\) is constructed to include constant predictors such that for every action \(a \in \mathcal{A}\), there exists a constant predictor \(f_a \in \mathcal{V}\) satisfying:
\[
f_a(X) = a, \quad \forall X.
\]
This implies:
\[
\operatorname{H}_{\ell, \mathcal{V}}(Y \mid X) = \inf_{f \in \mathcal{V}} \operatorname{E}_{\mathbb{P}(Y, X)}[\ell(Y, f(X))] \leq \inf_{a \in \mathcal{A}} \operatorname{E}_{\mathbb{P}(Y)}[\ell(Y, a)] = \operatorname{H}_{\ell}(Y).
\]
Thus:
\[
\operatorname{I}_{\ell,\mathcal{V}}(X; Y) = \operatorname{H}_{\ell}(Y) - \operatorname{H}_{\ell,\mathcal{V}}(Y \mid X) \geq 0.
\]

*Machine Learning Perspective:* In machine learning, having access to additional features or information can never worsen a model's performance since it always has the option not to use that information. This property ensures that feature selection or addition can only improve or maintain the current predictive performance, never decrease it. The non-negativity of utilitarian information aligns with this fundamental principle in machine learning.

## Recovery of Shannon's Information Theory

**True.** Utilitarian information can recover Shannon's information theory for specific choices of \(\mathcal{V}\) and \(\ell\).

*Proof:*  
Consider the logarithmic loss (cross-entropy loss), which is common in information theory and machine learning. For a predicted probability distribution \(q(Y)\) and the true distribution \(p(Y)\), the logarithmic loss is:
\[
\ell(Y, q) = -\log q(Y).
\]
The expected logarithmic loss over a distribution \(\mathbb{P}(Y)\) is:
\[
\operatorname{E}_{\mathbb{P}(Y)}[\ell(Y, q)] = -\operatorname{E}_{\mathbb{P}(Y)}[\log q(Y)].
\]
The Bayes risk for this loss function is achieved when \(q = p\), the true conditional distribution of \(Y\) given \(X\). This Bayes risk corresponds to the conditional entropy:
\[
\operatorname{H}(Y \mid X) = -\operatorname{E}_{\mathbb{P}(Y, X)}[\log \mathbb{P}(Y \mid X)].
\]

*Choice of Predictive Family \(\mathcal{V}\):*  
Let \(\mathcal{V}\) be the set of all possible conditional distributions \(q(Y \mid X)\), i.e., the set of all functions \(f: \mathcal{X} \to \Delta_{\mathcal{Y}}\) where \(\Delta_{\mathcal{Y}}\) is the set of all probability distributions over \(\mathcal{Y}\). This choice of \(\mathcal{V}\) includes all probabilistic mappings from \(X\) to \(Y\), and is effectively unrestricted.

Utilitarian mutual information with logarithmic loss and predictive family \(\mathcal{V}\) is:
\[
\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = \operatorname{H}_{\ell}(Y) - \operatorname{H}_{\ell, \mathcal{V}}(Y \mid X).
\]
Given the logarithmic loss as a proper scoring rule, the infimum of expected loss is attained when the predicted distribution matches the true distribution. Thus:
\[
\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = \operatorname{H}(Y) - \operatorname{H}(Y \mid X) = \operatorname{I}(X; Y).
\]
Thus, by selecting appropriate \(\ell\) and \(\mathcal{V}\), utilitarian information theory recovers Shannon's information theory.

*Machine Learning Perspective:* When unconstrained (having full access to all features and complexity), utilitarian information theory aligns with Shannon's framework, retaining all its properties.

## Data Processing Inequality

**False.** Utilitarian information does not necessarily satisfy the Data Processing Inequality.

*Counterexample:*  
Consider random variables \(X\) taking values in \(\{0,2\}\) and \(Y\) taking values in \(\{0,1,2\}\). Assume:  
- \(\mathbb{P}(Y = 0 \mid X = 0) = 1\)  
- \(\mathbb{P}(Y = 2 \mid X = 2) = \mathbb{P}(Y = 1 \mid X = 2) = 0.5\)  
- \(\mathbb{P}(X = 0) = \mathbb{P}(X = 2) = 0.5\)  

Using MAE loss \(\ell\) and the predictive family \(\mathcal{V}\) consisting of functions:  
- \(c_0(\emptyset) = 0, c_1(\emptyset) = 1, c_2(\emptyset) = 2\)  
- \(f(X = 0) = 0, f(X = 1) = 1, f(X = 2) = 11\)  
- \(g(\cdot) = 2\)  

The utilitarian marginal and conditional entropies are both 1. Predicting with \(f\) yields a higher risk since it cannot predict well when \(X = 2\). The risk using \(f\) is \(0 + 10 \times 0.25 + 11 \times 0.25 = 5.25\). Predicting with \(c_0\) yields a risk of 0.75. Thus, \(\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = 0.75 - 0.75 = 0\). However, if a function \(h\) exists such that \(h(2) = 1\) and is the identity otherwise, then \(f\) applied to \(h(X)\) becomes a good predictor, making \(\operatorname{I}_{\ell, \mathcal{V}}(h(X); Y) = 0.75 - 0.25 = 0.5\). Given \(X - h(X) - Y\) is a Markov chain, \(\operatorname{I}_{\ell, \mathcal{V}}(h(X); Y) = 0.5 > 0 = \operatorname{I}_{\ell, \mathcal{V}}(X; Y)\), violating the Data Processing Inequality.

*Machine Learning Perspective:* Feature preprocessing can enhance the "extractability" of information by a predictor, even if a preprocessing function is bijective. This reflects practical scenarios where feature engineering or transformations can increase predictive power. In machine learning, it's common to apply transformations to input features (e.g., normalization, scaling, or non-linear transformations) that can significantly improve model performance, even though these transformations don't add new information in the Shannon sense.

## Independence

**True.** Utilitarian information between two independent random variables \(X\) and \(Y\) is always zero.

*Proof:*  
Using independence:
\[
\operatorname{H}_{\ell, \mathcal{V}}(Y \mid X) = \inf_{f \in \mathcal{V}} \operatorname{E}_{(X, Y) \sim \mathbb{P}(X, Y)}[\ell(Y, f(X))] = \inf_{f \in \mathcal{V}} \operatorname{E}_{X \sim \mathbb{P}(X)} \operatorname{E}_{Y \sim \mathbb{P}(Y)}[\ell(Y, f(X))].
\]
By Jensen's inequality and the assumption of Optional Ignorance, we get:
\[
\operatorname{H}_{\ell, \mathcal{V}}(Y \mid X) \geq \operatorname{E}_{X \sim \mathbb{P}(X)} \left[ \inf_{f \in \mathcal{V}} \operatorname{E}_{Y \sim \mathbb{P}(Y)}[\ell(Y, f(X))] \right] = \operatorname{H}_{\ell, \mathcal{V}}(Y).
\]
Thus:
\[
\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = \operatorname{H}_{\ell, \mathcal{V}}(Y) - \operatorname{H}_{\ell, \mathcal{V}}(Y \mid X) \leq 0.
\]
Combining with non-negativity, we have \(\operatorname{I}_{\ell, \mathcal{V}}(X; Y) = 0\).

*Machine Learning Perspective:* When two variables are independent, they provide no information about each other. This is fundamental in machine learning, where independent features or labels do not inform predictions.

## Invariance to Bijections

**False.** Utilitarian information is not always invariant to bijections.

*Counterexample:*  
Consider the previous example where a function \(h\) was a bijection that improved mutual information. Applying a function can increase utilitarian information by enabling the constrained predictor to better predict the target variable. This makes sense for machine learning because preprocessing features can often enhance predictive accuracy, even if the transformation is bijective.

**Conclusion:** Utilitarian information theory provides a more nuanced approach to information theory in the context of machine learning by incorporating computational constraints and task-specific loss functions. This aligns well with practical scenarios where perfect information is unattainable, and decision-makers must operate within computational and informational limits.
# <expert_checklist>:
[
  "Does the response correctly identify whether utilitarian information is symmetric and provide a valid counterexample if it is not? (High importance)",
  "Is the explanation of why utilitarian information is not symmetric clear and related to practical machine learning scenarios? (Moderate importance)",
  "Does the response correctly prove that utilitarian information is always non-negative? (High importance)",
  "Is the explanation of non-negativity in utilitarian information related to the principle that additional information cannot worsen model performance? (Moderate importance)",
  "Does the response correctly demonstrate that utilitarian information can recover Shannon's information theory for specific choices of \\(\\mathcal{V}\\) and \\(\\ell\\)? (High importance)",
  "Is the explanation of how utilitarian information recovers Shannon's theory clear and related to unconstrained predictive scenarios in machine learning? (Moderate importance)",
  "Does the response correctly address whether utilitarian information satisfies the Data Processing Inequality and provide a valid counterexample if it does not? (High importance)",
  "Is the explanation of the Data Processing Inequality violation related to feature preprocessing in machine learning? (Moderate importance)",
  "Does the response correctly prove that utilitarian information between two independent random variables is always zero? (High importance)",
  "Is the explanation of utilitarian information being zero for independent variables related to the lack of predictive power in machine learning? (Moderate importance)",
  "Does the response correctly address whether utilitarian information is invariant to bijections and provide a valid counterexample if it is not? (High importance)",
  "Is the explanation of invariance to bijections related to feature transformations in machine learning? (Moderate importance)",
  "Does the response provide detailed proofs or counterexamples for each statement as required? (High importance)",
  "Are the machine learning perspectives provided for each statement clear and relevant to the concepts being discussed? (Moderate importance)",
  "Is the overall explanation of utilitarian information theory and its implications in machine learning clear and comprehensive? (High importance)",
  "Does the response avoid unnecessary or irrelevant details that do not contribute to the understanding of utilitarian information theory? (Low importance)",
  "Is the response structured logically, with clear sections for each statement and its proof or counterexample? (Moderate importance)",
  "Does the response use clear and precise language, making it accessible to non-expert evaluators? (Moderate importance)"
]
# <expert_checklist_time_sec>:
1178.0
# <expert_list_error_rubric>:
[
  {
    "error_name": "Incorrect proof of symmetry",
    "description": "The proof for whether utilitarian information is symmetric should clearly demonstrate that \\(\\operatorname{I}_{\\ell,\\mathcal{V}}(X; Y)\\) is not equal to \\(\\operatorname{I}_{\\ell,\\mathcal{V}}(Y; X)\\) using a valid counterexample. Check if the example provided correctly shows the asymmetry in utilitarian information.",
    "delta_score": -1
  },
  {
    "error_name": "Incorrect proof of non-negativity",
    "description": "The proof for non-negativity should show that \\(\\operatorname{I}_{\\ell,\\mathcal{V}}(X; Y) \\geq 0\\) by demonstrating that the conditional entropy is always less than or equal to the marginal entropy. Verify if the explanation aligns with the principle that additional information cannot worsen predictive performance.",
    "delta_score": -1
  },
  {
    "error_name": "Incorrect recovery of Shannon's information theory",
    "description": "The explanation should show how utilitarian information theory can recover Shannon's information theory by choosing appropriate \\(\\mathcal{V}\\) and \\(\\ell\\). Check if the example uses logarithmic loss and an unrestricted predictive family to demonstrate this recovery.",
    "delta_score": -1
  },
  {
    "error_name": "Incorrect proof of Data Processing Inequality",
    "description": "The proof should demonstrate whether utilitarian information satisfies the Data Processing Inequality using a valid counterexample. Verify if the example correctly shows a scenario where preprocessing can increase utilitarian information, violating the inequality.",
    "delta_score": -1
  },
  {
    "error_name": "Incorrect proof of independence",
    "description": "The proof should show that utilitarian information between two independent variables is zero by demonstrating that the conditional entropy equals the marginal entropy. Check if the explanation aligns with the principle that independent variables provide no information about each other.",
    "delta_score": -1
  },
  {
    "error_name": "Incorrect proof of invariance to bijections",
    "description": "The proof should demonstrate whether utilitarian information is invariant to bijections using a valid counterexample. Verify if the example correctly shows how a bijection can alter utilitarian information, reflecting practical scenarios in machine learning.",
    "delta_score": -1
  },
  {
    "error_name": "Lack of machine learning perspective",
    "description": "Each proof should include an explanation or illustration of why the result makes sense for machine learning. Check if the response provides a clear and relevant machine learning perspective for each statement.",
    "delta_score": -0.5
  },
  {
    "error_name": "Unclear or incomplete proofs",
    "description": "The proofs should be clear, detailed, and complete. Check if any proof lacks clarity or omits important steps or explanations.",
    "delta_score": -0.5
  },
  {
    "error_name": "Irrelevant or unnecessary details",
    "description": "The response should be concise and focused on the task. Check if there are any irrelevant or unnecessary details that do not contribute to the understanding of the proofs.",
    "delta_score": -0.5
  },
  {
    "error_name": "Incorrect or unclear definitions",
    "description": "The definitions provided for utilitarian entropy, mutual information, and predictive families should be correct and clear. Check if any definitions are incorrect or unclear.",
    "delta_score": -0.5
  }
]
# <expert_list_error_rubric_time_sec>:
1200.0
# <expert_brainstormed_rubric>:
[
  {
    "criterion": "Correctness of statements",
    "weight": 30.0,
    "checklist": [
      "Does the response correctly state that utilitarian information is not always symmetric? (i.e. original statement is false)",
      "Does the response correctly state that utilitarian information is always non-negative? (i.e. original statement is true)",
      "Does the response correctly state that utilitarian information can recover Shannon's information theory? (i.e. original statement is true)",
      "Does the response correctly state that utilitarian information does not always satisfy the Data Processing Inequality (DPI)? (i.e. original statement is false)",
      "Does the response correctly state that utilitarian information between independent variables is always zero? (i.e. original statement is true)",
      "Does the response correctly state that utilitarian information is not always invariant to bijections? (i.e. original statement is false)"
    ]
  },
  {
    "criterion": "Accuracy of Proofs and Counterexamples",
    "weight": 40.0,
    "checklist": [
      "Is there a correct proof or counterexample showing that invariance to symmetry does not hold? (A valid counterexample could involve a scenario where the predictive family is very limited and can predict much better X to Y than Y to X. Other valid approaches may exist.)",
      "Is the proof for non-negativity correct? (The proof must use the definition of action predictive family, i.e., Optional Ignorance. This property is crucial as non-negativity wouldn't hold otherwise.)",
      "Is the proof for recovery of Shannon's information theory correct? (The proof should consider the setting where the predictive family includes all measurable functions and the loss function is the log-loss, i.e., \u2113(y, \u0177) = -log \u0177(y). It could then use the properness of log-loss to show that the Bayes predictor is the conditional distribution so that the utilitarian conditional entropy is the same as in Shannon's conditional entropy.)",
      "Is there a correct proof or counterexample showing that the Data Processing Inequality (DPI) does not always hold? (A valid counterexample could involve a preprocessing step that makes prediction easier, violating the DPI. For instance, consider a scenario where the predictive family is very limited and doesn't contain the Bayes predictor from X to Y, but there exists another function g so that the composition g \u2218 f (for an f in the predictive family) would perfectly predict Y. Other valid approaches may exist.)",
      "Is the proof that independence of random variables gives zero utilitarian information correct? (One valid approach would be to use Jensen's inequality and the assumption of Optional Ignorance to show that information is non-positive. Then, given the non-negativity of utilitarian information, this would imply it is zero. Other valid proofs may exist.)",
      "Is there a correct proof or counterexample showing that invariance to bijections does not always hold? (A valid counterexample could show how a bijective transformation of X can change the utilitarian information by making Y easier or harder to predict. For example, this could occur if the function mapping X to Y is in a limited predictive family, and choosing a bijection g to Y such that g^-1 \u2218 f is not in the predictive family. Then the utilitarian information would have decreased, violating invariance. Other valid approaches may exist.)",
      "Are the proofs comprehensive rather than purely intuitive? (Look for detailed mathematical steps and reasoning, not just high-level explanations)",
      "Are the counterexamples provided with numerical values rather than just high-level intuition? (Look for specific numerical examples where applicable)",
      "Are all the assumptions and properties being used clearly stated in the proofs? (Each proof should explicitly mention the assumptions and properties it relies on)"
    ]
  },
  {
    "criterion": "Relevance to Machine Learning",
    "weight": 20.0,
    "checklist": [
      "Does the answer provide an intuition as to why information should be non-symmetric in machine learning settings? (For example, explaining that it's often easier to predict Y from X than X from Y in real-world scenarios)",
      "Does the answer provide an intuition as to why information should be non-negative in machine learning settings? (For example, explaining that adding features can never worsen a model's performance if used optimally)",
      "Does the answer provide an intuition as to why information should be able to recover Shannon's information theory in machine learning settings? (For example, explaining that when the predictive family is unconstrained, utilitarian information theory aligns with Shannon's framework, retaining all its properties)",
      "Does the answer provide an intuition as to why information should not always satisfy the DPI in machine learning settings? (For example, explaining how feature preprocessing such as normalization or embedding can enhance the 'extractability' of information by a predictor)",
      "Does the answer provide an intuition as to why information should be zero for independent variables in machine learning settings? (For example, explaining that adding irrelevant features won't add any predictive power or information gain)",
      "Does the answer provide an intuition as to why information should not be invariant to bijections in machine learning settings? (For example, explaining how feature preprocessing such as normalization or embedding can enhance the 'extractability' of information by a predictor even if it's bijective)",
      "Does the response connect utilitarian information theory to practical machine learning scenarios? (Look for concrete examples or applications in ML that illustrate the theoretical concepts)",
      "Are the machine learning perspectives on each statement insightful and relevant? (Look for clear connections between the theoretical concepts and ML practices, showing how utilitarian information theory applies to real-world ML problems)"
    ]
  },
  {
    "criterion": "Clarity and conciseness of explanation",
    "weight": 10.0,
    "checklist": [
      "Is the response well-organized and easy to follow? (Look for a clear structure with logical flow of ideas, possibly with section headers or numbered points)",
      "Does the response avoid irrelevant or unnecessary details not asked in the question? (The answer should focus on the properties of utilitarian information theory and their relevance to machine learning, without digressing into unrelated topics)",
      "Is the response concise and to the point? (Look for succinct explanations that convey the necessary information without unnecessary elaboration)",
      "Are the key concepts and calculations clearly explained? (Look for clear definitions of terms like 'utilitarian information', 'Data Processing Inequality', etc., and step-by-step explanations of any calculations or proofs)",
      "Does the answer provide some intuition behind why each statement is true or false? (Look for explanations that go beyond just stating facts, offering insights into the underlying reasons for each property of utilitarian information theory)"
    ]
  }
]
# <expert_brainstormed_rubric_time_sec>:
1138.0
# <expert_rubric>:
[
  {
    "criterion": "Correctness of Statements",
    "weight": 30.0,
    "performance_to_description": {
      "excellent": "The response correctly states the truth value of each statement: 1) Utilitarian information is not always symmetric (false), 2) Utilitarian information is always non-negative (true), 3) Utilitarian information can recover Shannon's information theory (true), 4) Utilitarian information does not always satisfy the Data Processing Inequality (false), 5) Utilitarian information between independent variables is always zero (true), 6) Utilitarian information is not always invariant to bijections (false). Each statement is addressed accurately and comprehensively.",
      "good": "The response correctly states the truth value of most statements, with one minor mistake. For example, it might incorrectly state that utilitarian information is symmetric but correctly address the other statements. Minor mistakes are those that do not significantly alter the understanding of utilitarian information theory.",
      "fair": "The response correctly states the truth value of some statements, but has a moderate mistake or two minor ones. For example, it might incorrectly state that utilitarian information is symmetric and invariant to bijections. Moderate mistakes are those that show a misunderstanding of key concepts in utilitarian information theory.",
      "poor": "The response has major mistakes in stating the truth value of the statements, such as incorrectly addressing multiple statements or failing to provide any correct answers. Major mistakes indicate a fundamental misunderstanding of utilitarian information theory."
    }
  },
  {
    "criterion": "Accuracy of Proofs and Counterexamples",
    "weight": 40.0,
    "performance_to_description": {
      "excellent": "The response provides correct proofs or counterexamples for each statement: 1) A valid counterexample for non-symmetry, 2) A proof using Optional Ignorance for non-negativity, 3) A proof showing recovery of Shannon's theory with log-loss and an unrestricted predictive family, 4) A counterexample for the Data Processing Inequality, 5) A proof using independence and Optional Ignorance for zero information, 6) A counterexample for non-invariance to bijections. All proofs are comprehensive, detailed, and include numerical examples where applicable.",
      "good": "The response provides mostly correct proofs or counterexamples, with one minor error. For example, it might have a small mistake in the proof for non-negativity but correctly address the other statements. Minor errors are those that do not significantly affect the validity of the proofs.",
      "fair": "The response provides some correct proofs or counterexamples, but has a moderate error or two minor ones. For example, it might incorrectly prove the Data Processing Inequality or fail to provide a numerical example where needed. Moderate errors indicate a lack of understanding of some aspects of utilitarian information theory.",
      "poor": "The response has major errors in the proofs or counterexamples, such as failing to provide valid arguments for multiple statements or lacking any detailed reasoning. Major errors indicate a fundamental misunderstanding of the concepts involved."
    }
  },
  {
    "criterion": "Relevance to Machine Learning",
    "weight": 20.0,
    "performance_to_description": {
      "excellent": "The response provides clear and insightful connections between utilitarian information theory and machine learning for each statement: 1) Explains non-symmetry with practical ML scenarios, 2) Relates non-negativity to feature addition, 3) Connects recovery of Shannon's theory to unconstrained predictive scenarios, 4) Discusses DPI violation with feature preprocessing, 5) Relates zero information to independent features, 6) Explains non-invariance to bijections with feature transformations. Each explanation is relevant and enhances understanding of utilitarian information theory in ML contexts.",
      "good": "The response provides mostly relevant connections to machine learning, with one minor issue. For example, it might not fully explain the relevance of non-symmetry but addresses the other statements well. Minor issues are those that do not significantly detract from the overall understanding.",
      "fair": "The response provides some relevant connections to machine learning, but has a moderate issue or two minor ones. For example, it might fail to relate DPI violation to ML or provide unclear explanations for some statements. Moderate issues indicate a lack of clarity in connecting theory to practice.",
      "poor": "The response lacks relevant connections to machine learning, with major issues such as failing to relate multiple statements to ML or providing irrelevant explanations. Major issues indicate a lack of understanding of the practical implications of utilitarian information theory."
    }
  },
  {
    "criterion": "Clarity and Conciseness of Explanation",
    "weight": 10.0,
    "performance_to_description": {
      "excellent": "The response is well-organized, concise, and easy to follow. It avoids irrelevant details, focuses on key concepts, and presents information logically. Key terms and calculations are clearly explained, and the response provides intuition behind each statement's truth value.",
      "good": "The response is mostly clear and concise, with a minor issue. For example, it might include a few unnecessary details but is generally easy to follow and focuses on key points. Minor issues do not significantly affect the clarity of the explanation.",
      "fair": "The response is somewhat clear but has a moderate issue or two minor ones. For example, it might include irrelevant details or present information in a confusing order. Moderate issues indicate a need for better organization and clarity.",
      "poor": "The response is unclear, verbose, or difficult to follow, with major issues such as disorganized information or missing key explanations. Major issues significantly impact the clarity and quality of the explanation."
    }
  }
]
# <expert_rubric_time_sec>:
1200.0
