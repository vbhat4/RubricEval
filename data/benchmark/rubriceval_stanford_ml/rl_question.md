# Problem

Consider a simple grid world environment. The environment is a 3x3 grid, where each cell represents a state \( s \in \{s_1, s_2, \ldots, s_9\} \), going top-to-bottom and left-to-right.
The top-left corner of the grid is $s_1$, and the bottom-right is $s_9$.
The top-middle is $s_2$, the top-right is $s_3$, etc.

The agent can perform four possible actions in each state: up, down, left, and right. Actions that would move the agent off the grid have no effect (the agent stays in the same state).

The agent starts in the top-left corner of the grid (\( s_1 \)) and aims to reach the bottom-right corner (\( s_9 \)). The reward function \( R(s, a) \) is defined as follows:

\begin{itemize}
    \item \( R(s, a) = 1 \) if \( s = s_9 \)
    \item \( R(s, a) = -0.1 \) for all other states \( s \neq s_9 \)
\end{itemize}

Assume the agent follows an optimal policy and the discount factor $\gamma = 0.9$

Determine the optimal action \( a^*(s) \) for the agent in state \( s_5 \) (the center of the grid) after performing one step of value iteration (assume we initialize $V(s_i) = 0$ for all states). What is the expected reward of taking this action?

# Potential Solution
The environment is deterministic, so our value iteration update rule is:
$$ V'(s) = \max_a  R(s, a) + \gamma V(s') $$

Which gives us:
\begin{align*}
    V'(s_2) &= \max(-0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0) = -0.1\\
    V'(s_4) &= \max(-0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0) = -0.1\\
    V'(s_6) &= \max(-0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0) = -0.1\\
    V'(s_8) &= \max(-0.1 + 0.9 \cdot 0, -0.1 + 0.9 \cdot 0, 1 + 0.9 \cdot 0) = 1
\end{align*}

Possible actions from \( s_5 \):
\begin{itemize}
    \item Up (to \( s_2 \)): \( R(s_5, \text{up}) + \gamma V'(s_2) = -0.1 + 0.9 \cdot -0.1 = -0.19 \)
    \item Left (to \( s_4 \)): \( R(s_5, \text{left}) + \gamma V'(s_4) = -0.1 + 0.9 \cdot -0.1 = -0.19 \)
    \item Right (to \( s_6 \)): \( R(s_5, \text{right}) + \gamma V'(s_6) = -0.1 + 0.9 \cdot -0.1 = -0.19 \)
    \item Down (to \( s_8 \)): \( R(s_5, \text{down}) + \gamma V'(s_8) = -0.1 + 0.9 \cdot 1 = 0.8 \)
\end{itemize}

The optimal action \( a^*(s_5) \) is "down" with the value of 0.8.
