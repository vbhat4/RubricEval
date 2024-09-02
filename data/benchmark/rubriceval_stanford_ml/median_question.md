# Problem

You must now predict the value of a random variable, $y$, which is \textbf{continuous} valued so as to minimize the expected loss over the distribution of $y$, i.e. $\mathbb{E}_{y}[L(\hat{y}, y)]$. You know the entire probability density function of $y$, i.e. $p(y)$ is fully known to you. In the case where you are using the absolute value loss function as your objective: 

\begin{align*}
    L(\hat{y}, y) &= | \hat{y} - y |
\end{align*}, 

what is the optimal prediction for $\hat{y}$? Provide a one line equation after a proof.


# Potential Solution

To calculate the variance of the delivery times, we can use the following formula for the variance of a mixture of distributions:

```latex
\begin{align}
\operatorname{Var}[X] & = \sigma^2 \\
& = \operatorname{E}[X^2] - \mu^{2} \\
& = \sum_{i=1}^n w_i(\sigma_i^2 + \mu_i^{2} )- \sum_{i=1}^n w_i\mu_i^{2}
\end{align}
```

so we just need to calculate the variance and mean of each distribution.

For the Gaussian distribution we have $\mu_g=5$ and variance $\sigma_g^2=0.01$.

For the Trapezoidal distribution you can compute $E[X]$ and $E[X^2]$ by integrating the distribution function on the entire range (i.e. 3 simple intergrals). Or you can use
the following formula for the $k$-th moment of the Trapezoidal distribution:

```latex
\begin{align}
E[X^k] = \frac{2}{d+c-b-a}\frac{1}{(k+1)(k+2)}\left(\frac{d^{k+2} - c^{k+2}}{d - c} - \frac{b^{k+2} - a^{k+2}}{b - a}\right)
\end{align}
```

Using this formula with k=1, a=0, b=1, c=3, and d=4 

```latex
\begin{align}
\mu_t &= E[X] \\  
&= 
\frac{1}{3(d+c-b-a)}\left(\frac{d^3 - c^3}{d - c} - \frac{b^3 - a^3}{b - a}\right)\\ 
& = 2
\end{align}
```

and 

```latex
\begin{align}
\sigma_t^2
&= E[X^2] - \mu_t^2\\
&= \frac{1}{6(d+c-b-a)}\left(\frac{d^4 - c^4}{d - c} - \frac{b^4 - a^4}{b - a}\right) - 4\\
&= \frac{29}{6}- 4^2\\
&= \frac{5}{6}
\end{align}
```

Putting all together we have that the mean of the mixture is 

```latex    
\begin{align}
\mu  &= \sum_{i=1}^n w_i\mu_i\\
& = 0.3 * 5 + 0.7 * 2\\
& = 2.9
\end{align}
``` 

and the variance of the mixture is 

```latex
\begin{align}
\sigma^2  &= \sum_{i=1}^n w_i(\sigma_i^2 + \mu_i^{2} )- \mu^2 \\
& = 0.3 * (0.01 + 5^2) + 0.7 * 29/6 - 2.9^2\\
& = 2.4763
\end{align}
```



