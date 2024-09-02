# Problem


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

# Potential Solution

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