# Greek-Calculator

## Mathematical Foundations

### Stochastic Calculus Framework
The system implements the Black-Scholes-Merton model under the following assumptions:

**(i) Geometric Brownian Motion for underlying asset price:**
$$ dS = \mu S dt + \sigma S dW $$

Where:
- $S$: Underlying asset price
- $\mu$: Drift rate (risk-neutral: $r - q$)
- $\sigma$: Volatility (constant)
- $dW$: Wiener process increment

**(ii) Risk-Neutral Valuation:**
$$ \mathbb{E}^Q[V(S,T)] = e^{-rT} \mathbb{E}[V(S,T)] $$

Where $Q$ is the risk-neutral measure.

### Partial Differential Equation Framework
The Black-Scholes PDE:
$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q)S \frac{\partial V}{\partial S} - rV = 0 $$

**Call Option:**
- $V(0,t) = 0$
- $V(S,t) \to S - Ke^{-r(T-t)}$ as $S \to \infty$
- $V(S,T) = \max(S-K, 0)$

**Put Option:**
- $V(0,t) = Ke^{-r(T-t)}$
- $V(S,t) \to 0$ as $S \to \infty$
- $V(S,T) = \max(K-S, 0)$

## Architecture Overview

### System Component Diagram

┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Parser    │ →  │ Batch Processor  │ →  │ Results Writer  │
│                 │    │                  │    │                 │
│ - Data Validation│    │ - Thread Pool    │    │ - Comprehensive │
│ - Error Handling │    │ - Work Chunking  │    │   Reporting     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ↓
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Math Utilities │ ←  │ Greek Calculator │ →  │ Risk Analyzer   │
│                 │    │                  │    │                 │
│ - BS Pricing    │    │ - 1st/2nd Order  │    │ - Portfolio VaR │
│ - Implied Vol   │    │ - Sensitivities  │    │ - ES Calculation │
└─────────────────┘    └──────────────────┘    └─────────────────┘



### Class Dependencies
- `BatchProcessor` → `MathUtils`, `GreekCalculator`, `RiskAnalyzer`
- `GreekCalculator` → `MathUtils`
- `RiskAnalyzer` → `GreekCalculator`, `MathUtils`
- `CSVParser` → `OptionData`
- `ResultsWriter` → `OptionResult`, `PortfolioRisk`

## Core Mathematical Models

### Black-Scholes Closed-Form Solutions

**Call Option Pricing:**
$$ C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2) $$

**Put Option Pricing:**
$$ P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1) $$

**Intermediate Variables:**
$$ d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}} $$
$$ d_2 = d_1 - \sigma\sqrt{T} $$

Where:
- $N(x)$: Cumulative standard normal distribution
- $S_0$: Current underlying price
- $K$: Strike price
- $T$: Time to expiration (years)
- $r$: Risk-free interest rate (continuous)
- $q$: Dividend yield (continuous)
- $\sigma$: Volatility (annualized)

## First-Order Greeks ($\Delta$, $\Gamma$, $\Theta$, $\nu$, $\rho$)

### Delta ($\Delta$) - Price Sensitivity to Underlying
$$ \Delta_{call} = \frac{\partial C}{\partial S} = e^{-qT} N(d_1) $$
$$ \Delta_{put} = \frac{\partial P}{\partial S} = e^{-qT} [N(d_1) - 1] $$

**Mathematical Derivation:**
$$ \frac{\partial C}{\partial S} = e^{-qT} N(d_1) + S_0 e^{-qT} \frac{\partial N(d_1)}{\partial S} - K e^{-rT} \frac{\partial N(d_2)}{\partial S} $$

Using chain rule and the fact that $S_0 e^{-qT} \phi(d_1) = K e^{-rT} \phi(d_2)$, we get the simplified form.

### Gamma ($\Gamma$) - Delta Sensitivity to Underlying
$$ \Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{e^{-qT} \phi(d_1)}{S_0 \sigma \sqrt{T}} $$

**Mathematical Properties:**
- $\Gamma$ is identical for calls and puts
- Maximum when option is at-the-money
- Approaches 0 as option moves deep in/out of money

### Theta ($\Theta$) - Time Decay (Per Day)
$$ \Theta_{call} = \frac{\partial C}{\partial t} / 365 = \left[ -\frac{S_0 e^{-qT} \phi(d_1) \sigma}{2\sqrt{T}} + q S_0 e^{-qT} N(d_1) - r K e^{-rT} N(d_2) \right] / 365 $$
$$ \Theta_{put} = \frac{\partial P}{\partial t} / 365 = \left[ -\frac{S_0 e^{-qT} \phi(d_1) \sigma}{2\sqrt{T}} - q S_0 e^{-qT} N(-d_1) + r K e^{-rT} N(-d_2) \right] / 365 $$

### Vega ($\nu$) - Volatility Sensitivity (Per 1% Change)
$$ \nu = \frac{\partial V}{\partial \sigma} \times 0.01 = S_0 e^{-qT} \phi(d_1) \sqrt{T} \times 0.01 $$

### Rho ($\rho$) - Interest Rate Sensitivity (Per 1% Change)
$$ \rho_{call} = \frac{\partial C}{\partial r} \times 0.01 = K T e^{-rT} N(d_2) \times 0.01 $$
$$ \rho_{put} = \frac{\partial P}{\partial r} \times 0.01 = -K T e^{-rT} N(-d_2) \times 0.01 $$

## Second-Order Greeks

### Vanna - $\Delta$ Sensitivity to Volatility
$$ \text{Vanna} = \frac{\partial \Delta}{\partial \sigma} = \frac{\partial^2 V}{\partial S \partial \sigma} = -e^{-qT} \phi(d_1) \frac{d_2}{\sigma} $$

**Financial Interpretation:** Measures how delta changes with volatility
- Positive for out-of-money options
- Negative for in-the-money options

### Charm - $\Delta$ Sensitivity to Time
$$ \text{Charm} = \frac{\partial \Delta}{\partial t} = \frac{\partial^2 V}{\partial S \partial t} = -e^{-qT} \phi(d_1) \frac{2(r-q)T - d_2 \sigma \sqrt{T}}{2T \sigma \sqrt{T}} $$

### Vomma - $\nu$ Sensitivity to Volatility
$$ \text{Vomma} = \frac{\partial \nu}{\partial \sigma} = \frac{\partial^2 V}{\partial \sigma^2} = S_0 e^{-qT} \phi(d_1) \sqrt{T} \frac{d_1 d_2}{\sigma} $$

### Speed - $\Gamma$ Sensitivity to Underlying
$$ \text{Speed} = \frac{\partial \Gamma}{\partial S} = \frac{\partial^3 V}{\partial S^3} = -\frac{\Gamma}{S} \times \left( 1 + \frac{d_1}{\sigma \sqrt{T}} \right) $$

### Zomma - $\Gamma$ Sensitivity to Volatility
$$ \text{Zomma} = \frac{\partial \Gamma}{\partial \sigma} = \frac{\partial^3 V}{\partial S^2 \partial \sigma} = \Gamma \times \frac{d_1 d_2 - 1}{\sigma} $$

## Risk Exposure Metrics

### Dollar Delta
$$ \$\Delta = \Delta \times S_0 \times \text{Quantity} $$

### Dollar Gamma
$$ \$\Gamma = \frac{1}{2} \times \Gamma \times S_0^2 \times \text{Quantity} $$

**Interpretation:** P&L impact from gamma effects:
$$ \text{P&L}_{\text{Gamma}} \approx \frac{1}{2} \times \Gamma \times (\Delta S)^2 = \$\Gamma \times \left( \frac{\Delta S}{S_0} \right)^2 $$

## Numerical Methods

### Cumulative Normal Distribution
**High-Precision Approximation:**


static double norm_cdf(double x) {
    if (x < -8.0) return 0.0;
    if (x > 8.0) return 1.0;
    double sum = x;
    double term = x;
    for (int i = 1; i < 100; i++) {
        term *= x * x / (2 * i + 1);
        sum += term;
    }
    return 0.5 + (sum * 0.3989422804014327); // 1/√(2π)
}




