# Greek-Calculator

Mathematical Foundations

Stochastic Calculus Framework
The system implements the Black-Scholes-Merton model under the following assumptions:

(i) Geometric Brownian Motion for underlying asset price:

$$ dS = μS dt + σS dW $$

S: Underlying asset price
μ: Drift rate (risk-neutral: r - q)
σ: Volatility (constant)
dW: Wiener process increment

(ii) Risk-Neutral Valuation:

$$ E^Q[V(S,T)] = e^{-rT} E[V(S,T)] $$

