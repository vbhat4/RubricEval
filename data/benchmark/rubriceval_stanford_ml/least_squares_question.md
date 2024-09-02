# Problem

The Least Squares objective is denoted by:

\begin{align*}
    \argmin_{w} \quad ( Y - Xw)^T(Y - Xw)
\end{align*}

where $Y \in \mathbb{R}^n$, $w \in \mathbb{R}^d$ are the parameters, and $X \in \mathbb{R}^{n \times d}$ is the feature matrix.\\

The Ridge-regression regularized objective is:
\begin{align*}
\argmin_{w} \quad (Y - Xw)^T(Y - Xw) + \lambda w^Tw
\end{align*}

Show that $||w^*_r ||_2^2 \leq ||w^*_l ||_2^2$. Where $w^*_l$ and $w^*_r$ respectively denote the optimal solution for the least squares objective and the ridge regression objective.

# Potential Solution
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

