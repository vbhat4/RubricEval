# Problem

You must now predict the value of a random variable, $Y$, which is \textbf{continuous} valued so as to minimize the expected loss over the distribution of $Y$, i.e. $\mathbb{E}_{Y}[L(\hat{Y}, Y)]$. You know the entire probability density function of $Y$, i.e. $p(Y)$ is fully known to you. You can also assume that $Y$ has a Lebesgue density.  

In the case where you are using the following loss function as your objective: 

```latex
\begin{align*}
    L(\hat{Y}, Y) &= \left| \left( Y - \hat{Y} \right) \left(0.16- \mathbf{1}\{ Y > \hat{Y} \} \right)
\end{align*}
``` 

what is the optimal prediction for $\hat{Y}$? Prove it.



# Potential Solution

To solve for the minimum of the loss function

```latex
\[
\min_{\hat{Y}} E_{p(Y)} \left[ \left| \left( Y - \hat{Y} \right) \left(0.16 - \mathbf{1}\{ Y > \hat{Y} \} \right) \right| \right],
\]
```

we need to determine the optimal estimator \(\hat{Y}\) that minimizes the expected loss.

After a bit of simplification, we can rewrite the task as:

```latex
\[
\arg\min_{\hat{Y}} E_{p(Y)}[L(\hat{Y}, Y)] = \int_{-\infty}^{\hat{Y}} 0.16 (\hat{Y} - Y) p(Y) \, dY + \int_{\hat{Y}}^{\infty} 0.84 (Y - \hat{Y}) p(Y) \, dY.
\]
```

To find the optimal \(\hat{Y}\), take the derivative with respect to \(\hat{Y}\) and set it to zero:

```latex
\begin{align}
 0&=\frac{d}{d\hat{Y}} E_{p(Y)}[L(\hat{Y}, Y)]\\
 &=  \frac{d}{d\hat{Y}} \left( \int_{-\infty}^{\hat{Y}} 0.16 (\hat{Y} - Y) p(Y) \, dY + \int_{\hat{Y}}^{\infty} 0.84 (Y - \hat{Y}) p(Y) \, dY \right) \\
    &= 0.16 \int_{-\infty}^{\hat{Y}} p(Y) \, dY - 0.84 \int_{\hat{Y}}^{\infty} p(Y) \, dY & \text{by Leibniz's rule} \\
     &= 0.16 P(Y \leq \hat{Y}) - 0.84 P(Y > \hat{Y}) \\
 &=  0.16 F(\hat{Y}) - 0.84 (1 - F(\hat{Y})) & P(Y \leq \hat{Y}) + P(Y > \hat{Y}) \\
 0.84 &= 0.16 F(\hat{Y}) + 0.84 F(\hat{Y})  \\
    \hat{Y}) &= F^{-1}(0.84).
```


   where \(F(\hat{Y}) = P(Y \leq \hat{Y})\) is the cumulative distribution function (CDF) of \(Y\) so \(F^{-1}(0.84)\) is the 84th percentile of the distribution of \(Y\).
