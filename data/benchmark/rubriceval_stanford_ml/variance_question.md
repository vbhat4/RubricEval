# Problem

You are a data analyst working for a tech company that manages a fleet of autonomous delivery drones. These drones deliver packages in two distinct environments: a densely packed urban area and a sprawling suburban area. Your task is to predict the delivery times of these drones to optimize scheduling and improve customer satisfaction.

**Urban Area**: In the urban environment, the drone delivery times are primarily influenced by a range of factors such as tall buildings, varying wind speeds due to narrow alleyways, and frequent stops. This results in a highly variable delivery time that is longer in some areas and shorter in others, but generally follows a pattern that could be represented by a Trapezoidal Distribution. The delivery times are likely to increase linearly up to a certain point due to navigating traffic and wind, remain relatively stable when in open spaces, and then decrease linearly as they approach more predictable pathways near delivery points. We assume that for the Trapezoidal distribution a=0, b=1, c=3, and d=4. As a reminder the Trapezoidal Distribution is defined as follows:

```latex
\begin{align}
f_{urban}(x)=
\begin{cases}
\frac{2}{d+c-a-b}\frac{x-a}{b-a}  & \text{for } a\le x < b \\
\frac{2}{d+c-a-b}  & \text{for } b\le x < c \\
\frac{2}{d+c-a-b}\frac{d-x}{d-c}  & \text{for } c\le x \le d
\end{cases}
\end{align}
```

**Suburban Area**: In the suburban environment, drone delivery times are less variable due to more consistent and open flying conditions. The drones generally fly at a constant speed over open areas with fewer obstacles. This scenario is well-modeled by a Normal Distribution with a specific mean and variance, where most delivery times cluster around the mean due to the predictable nature of flying over suburban landscapes. We assume that the mean delivery time is 5 hours with a standard deviation of 0.1 hours. Namely:

```latex
\begin{align}
f_{suburban}(x) = \frac{1}{\sqrt{2\pi} \cdot 0.5} \cdot e^{-\frac{(x - 2)^2}{2 \cdot 0.5^2}}
\end{align}
```

Assume that 70% of the drones operate in the urban area and 30% in the suburban area. So the delivery times are a mixture of the two distributions. What is the mean and variance of the delivery times?

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
