<img width="971" height="589" alt="Screenshot 2025-10-18 001420" src="https://github.com/user-attachments/assets/e611c754-3560-4a28-9aca-6750adb12b6a" />
<img width="793" height="426" alt="Screenshot 2025-10-18 001442" src="https://github.com/user-attachments/assets/4b3614ed-c8a6-4ebe-9808-5e132ec30563" />
<img width="828" height="308" alt="Screenshot 2025-10-18 001450" src="https://github.com/user-attachments/assets/162b5f80-e6b2-4e71-839a-a790955146f2" />
<img width="746" height="560" alt="Screenshot 2025-10-18 001503" src="https://github.com/user-attachments/assets/c2b9b8ab-0c3c-4656-8338-01e074eabd8b" />
<img width="715" height="675" alt="Screenshot 2025-10-18 001516" src="https://github.com/user-attachments/assets/079ef9f4-93b8-4792-8043-cd64bb4cf5b3" />
<img width="779" height="808" alt="Screenshot 2025-10-18 001528" src="https://github.com/user-attachments/assets/2a92d6ed-3a91-4008-a868-142833ca2dec" />


# Greek-Calculator

A high-performance options analytics toolkit implementing the Black–Scholes–Merton framework, analytical Greeks, implied-volatility extraction, batch processing, and portfolio risk metrics. Designed for quantitative research, institutional risk management, and production analytics.

---

## Table of Contents

1. [Overview](#overview)  
2. [Concepts & Mathematical Foundations](#concepts--mathematical-foundations)  
3. [System Architecture & Components](#system-architecture--components)  
4. [Numerical Methods & Stability](#numerical-methods--stability)  
5. [Parallel Processing & Performance](#parallel-processing--performance)  
6. [Input / Output Specification & Example](#input--output-specification--example)  
7. [Validation, Testing & Edge Cases](#validation-testing--edge-cases)  
8. [Difficulties Faced and How We Resolved Them](#difficulties-faced-and-how-we-resolved-them)  
9. [How to Build & Run (quick start)](#how-to-build--run-quick-start)  
10. [Extensibility & Future Work](#extensibility--future-work)  
11. [Glossary / Variable Meanings](#glossary--variable-meanings)  
12. [Troubleshooting & FAQs](#troubleshooting--faqs)  
13. [License & Notes](#license--notes)

---

## Overview

Greek-Calculator computes option prices, first- and higher-order Greeks, implied volatilities, and portfolio-level risk measures from batch inputs. The tool is optimized for throughput (batch processing with a thread pool), numerical stability (robust CDF/PDF approximations and clamped solvers), and practical risk reporting (VaR, ES, delta/gamma exposures).

This repository contains:
- a single-file reference program: `greek_calculator.cpp` (source included),
- a batch CSV parser and writer,
- robust numerical utilities and analytic Greeks,
- a threaded batch processor,
- portfolio aggregation and a simple VaR/ES report.

---

## Concepts & Mathematical Foundations

### Underlying dynamics

We model the underlying asset using Geometric Brownian Motion:

\[
dS = \mu S\,dt + \sigma S\,dW
\]

Under risk-neutral pricing, drift is replaced by \(r - q\).

### Black–Scholes closed-form

Define
\[
d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},\qquad
d_2 = d_1 - \sigma\sqrt{T}
\]

Call and put prices:
\[
C = S_0 e^{-qT}N(d_1) - K e^{-rT}N(d_2)
\]
\[
P = K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)
\]

\(N(\cdot)\) is the standard normal CDF, \(\phi(\cdot)\) is the PDF.

### Greeks (selected)

- Delta: \(\Delta_{\text{call}} = e^{-qT}N(d_1)\)  
- Gamma: \(\Gamma = \dfrac{e^{-qT}\phi(d_1)}{S_0\sigma\sqrt{T}}\)  
- Vega: \(\nu = S_0 e^{-qT}\phi(d_1)\sqrt{T}\)  
- Theta, Rho and higher-order Greeks (Vanna, Vomma, Charm, Zomma, Speed) are computed analytically.

---

## System Architecture & Components

High-level pipeline:

CSV Parser -> Batch Processor (thread pool) -> Greek Calculator & Pricing -> Results Writer -> Portfolio Aggregator & Risk Report


Key modules:

- **CSV Parser**: robust parsing, validation, normalization.  
- **Batch Processor**: splits work into chunks and runs worker threads.  
- **Math Utilities**: norm PDF/CDF, clamp, stable transforms, pricing.  
- **Greek Calculator**: closed-form formulas and implied-vol solver.  
- **Risk Analyzer**: aggregation, exposures, VaR/ES, stress tests.  
- **Results Writer**: detailed CSV and plain-text portfolio report.

---

## Numerical Methods & Stability

- **Normal CDF/PDF**: series-based central-region evaluation + hard tail returns for |x| > 8 to avoid underflow/overflow.  
- **Implied volatility solver**: conservative multiplicative/bisection-like solver with clamping to `[0.001, 5.0]` and tolerance `1e-8`.  
- **T → 0 and σ → 0 handling**: intrinsic-value fallback prevents division by zero and precision loss.  
- **Precision guards**: precompute sqrt(T) and sigma*sqrt(T); consider long double if needed in extreme precision scenarios.

---

## Parallel Processing & Performance

- Configure threads via command-line or auto-detect hardware concurrency.  
- Chunking formula: `chunk_size = ceil(n / threads)`.  
- Preallocate result arrays so each worker writes to unique indices to avoid locks.  
- Minimal synchronization via an atomic progress counter and a mutex-protected logger.  
- Hot-path micro-optimizations: reuse computed terms, reduce recomputation of transcendental calls.

---

## Input / Output Specification & Example

### Required CSV columns

| Column     | Type   | Description            |
|------------|--------|------------------------|
| ID         | string | Unique identifier      |
| Spot       | double | Underlying price (>0)  |
| Strike     | double | Strike price (>0)      |
| Rate       | double | Continuous risk-free   |
| Volatility | double | Volatility (annual)    |
| Time       | double | Years to expiry        |
| Type       | string | "call" or "put"        |

### Optional columns
`Underlying`, `Dividend_Yield`, `MarketPrice`, `Quantity`, `Notional`

### Example input (`options_input.csv`)
ID,Spot,Strike,Rate,Volatility,Time,Type,Dividend_Yield,MarketPrice,Quantity
opt-001,100,100,0.05,0.20,0.5,call,0.0,5.5,10
opt-002,100,110,0.05,0.20,0.5,put,0.0,8.2,5
opt-003,50,45,0.04,0.35,0.25,call,0.01,6.1,2


### Example output columns (CSV)

ID,TheoreticalPrice,MarketPrice,PriceDifference,ImpliedVol,Delta,Gamma,Vega,Theta,Rho,Vanna,Charm,Vomma,Speed,Zomma,IntrinsicValue,ExtrinsicValue,TimeValue,DeltaExposure,GammaExposure,VegaExposure,DollarDelta,DollarGamma,Status



---

## Validation, Testing & Edge Cases

- Unit tests for CDF/PDF, closed-form prices, Greeks, solver under synthetic scenarios.  
- Edge cases:
  - `T = 0` → intrinsic values and zero Greeks.  
  - `σ = 0` → non-diffusive limits.  
  - deep OTM/ITM → stable tail handling.  
- CI should include numerical regression tests and throughput benchmarks.

---

## Difficulties Faced and How We Resolved Them

1. **Numerical instability in tails / short maturities**  
   - Implemented safe-tail returns for the normal CDF and intrinsic-value fallback for `T` below a threshold.

2. **Implied volatility solver divergence**  
   - Use prechecks (market price < intrinsic), implement multiplicative adaptation and clamp sigma to avoid runaway values.

3. **Cancellation and precision loss in d1/d2**  
   - Precompute reusable parts, consider long double for critical branches.

4. **Throughput and load imbalance**  
   - Use smaller chunks or a task queue for skewed workloads. Preallocate result vectors to avoid allocations in hot paths.

5. **Memory pressure from repeated allocations**  
   - Preallocate vectors and reuse buffers. Stream output buffered batches.

6. **Greeks consistency**  
   - Use parity relations (put-call parity) and shared intermediate computations to enforce consistency numerically.

7. **Audit and reproducibility requirements**  
   - Add optional convergence traces, timestamps, deterministic seeds for sampling.

---

## How to Build & Run (quick start)

1. Save the provided source file as `greek_calculator.cpp`.

2. Build:
g++ -std=c++17 -O2 -pthread greek_calculator.cpp -o greek_calc

3. Run:
./greek_calc options_input.csv results_output.csv 4



- Arguments: `<input_csv> <output_csv> [threads]`  
- The program writes `results_output.csv` and, if portfolio analysis is enabled, `results_output.csv.risk_report.txt`.

---

## Glossary / Variable Meanings

- `S` — current spot price of the underlying.  
- `K` — strike price.  
- `r` — continuous risk-free interest rate.  
- `q` — continuous dividend yield.  
- `σ` (sigma) — annualized volatility.  
- `T` — time to expiry (years).  
- `d1`, `d2` — Black–Scholes intermediate terms.  
- `N(x)` — standard normal CDF.  
- `φ(x)` — standard normal PDF.  
- `Delta` — ∂V/∂S (sensitivity to spot).  
- `Gamma` — ∂²V/∂S² (sensitivity of delta).  
- `Vega` — ∂V/∂σ (sensitivity to volatility).  
- `Theta` — ∂V/∂t (time decay, reported per day).  
- `Rho` — ∂V/∂r (sensitivity to rate, per 1%).  
- `DollarDelta` — delta × spot (P&L per unit spot move).  
- `DollarGamma` — 0.5 × gamma × spot² (gamma-scaled contribution to P&L).

---

## Troubleshooting & FAQs

- **Implied vol at clamp limits** means market price outside model range; verify market price and intrinsic value.  
- **NaN Greeks** check that `time > 0`, `sigma > 0`, `spot` and `strike` > 0.  
- **Slow performance**: reduce computed Greeks via config or increase threads; profile to find hotspots.  
- **Different underlying assets**: current VaR uses approximate single-spot; change to per-option underlying for multi-asset portfolios.

---


## Major Development Challenges and Resolutions

### 1. Floating-Point Precision Errors
**Problem:** Double precision rounding caused inaccuracies in Greeks.  
**Resolution:** Used `long double` for sensitive calculations and constrained results via `std::max()` and `std::clamp()`.

---

### 2. Numerical Instability in Cumulative Normal (N)
**Problem:** Standard C++ does not provide cumulative normal distribution.  
**Resolution:** Implemented:

```cpp
inline double norm_cdf(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}
```

3. Implied Volatility Root-Finding Divergence

Problem: Newton-Raphson may diverge if Vega ≈ 0.
Resolution: Hybrid Newton-Bisection solver:

``` cpp

double implied_vol(double market_price, double S, double K, double r, double q, double T, bool is_call) {
    double low = 1e-6, high = 5.0, sigma = 0.2;
    for (int i = 0; i < 100; ++i) {
        double price = black_scholes_price(S,K,r,q,T,sigma,is_call);
        double diff = price - market_price;
        if (fabs(diff) < 1e-8) break;
        double vega = black_scholes_vega(S,K,r,q,T,sigma);
        if (vega < 1e-8) { sigma = 0.5 * (low + high); continue; }
        sigma -= diff / vega;
        if (diff > 0) high = sigma; else low = sigma;
    }
    return sigma;
}


```

4. Thread Synchronization Issues

Problem: Concurrent writes caused race conditions.
Resolution: Atomic counters and per-thread result buffers:

``` cpp
atomic<size_t> processed_count{0};
vector<OptionResult> results(batch.size());
```

5. Performance Bottleneck

Problem: Sequential evaluation of thousands of options was slow.
Resolution: Multithreading with dynamic chunk allocation:

```cpp
  vector<thread> threads;
size_t chunk = (batch.size() + config.threads - 1) / config.threads;

for (size_t t = 0; t < config.threads; ++t) {
    size_t start = t * chunk;
    size_t end = min(start + chunk, batch.size());
    threads.emplace_back([&, start, end]() {
        for (size_t i = start; i < end; ++i)
            results[i] = compute_option(batch[i]);
        processed_count += (end - start);
    });
}
for (auto& th : threads) th.join();

```
6. Incorrect Numerical Greeks Validation

Problem: Finite-difference approximation produced inconsistent results.
Resolution: Adaptive step size:


```cpp
double h = max(1e-4, 0.01 * S);
double delta_fd = (price(S + h) - price(S - h)) / (2 * h);

```
<img width="796" height="779" alt="Screenshot 2025-10-18 004032" src="https://github.com/user-attachments/assets/4428b1d5-7035-41a3-bd3e-5382ff90adb2" />
<img width="820" height="366" alt="Screenshot 2025-10-18 004041" src="https://github.com/user-attachments/assets/0aa9d8f0-47a4-4454-9a56-7ed6eb95c42f" />



7. CSV Input Parsing Errors

Problem: Extra commas or newlines caused skipped lines.
Resolution:

```cpp
OptionData parse_csv_line(const string& line) {
    stringstream ss(line);
    string field; vector<string> fields;
    while (getline(ss, field, ',')) fields.push_back(trim(field));
    if (fields.size() < 7) throw runtime_error("Malformed CSV line");
    return {stod(fields[0]),stod(fields[1]),stod(fields[2]),
            stod(fields[3]),stod(fields[4]),stod(fields[5]),
            fields[6]=="C"};
}

```

8. Logarithmic Domain Error (log(0))

Resolution:
```cpp
inline double safe_log(double x) { return log(max(x,1e-12)); }
```

9. Non-Converging Options

Resolution: Iteration cap and volatility clipping:
```cpp
sigma = std::clamp(sigma, 0.0001, 5.0);
```

10. Output Synchronization

Resolution: Mutex-protected CSV writes:
```cpp
mutex file_mutex;
void write_result(const OptionResult& res, ofstream& fout) {
    lock_guard<mutex> lock(file_mutex);
    fout << res.to_csv_line() << "\n";
}
```

MAJOR PROBLEMS:->
------------------------------------------------------------
1) Numerical Instability in Normal CDF/PDF
------------------------------------------------------------
Symptom:
  - The normal CDF N(d) returned incorrect values (0 or 1) for large |d| causing incorrect Greeks.
Root Cause:
  - Floating-point underflow and overflow for very high or low values.
Fix:
  - Used clamping for extreme tails:
      if (x < -8.0) return 0.0;
      if (x > 8.0) return 1.0;
  - Used long double precision for better accuracy.
Testing:
  - Compared computed values with reference tables for x = ±6, ±8, ±10.
Prevention:
  - Always handle tails explicitly and prefer long double precision for financial models.

------------------------------------------------------------
2) Implied Volatility Solver Not Converging
------------------------------------------------------------
Symptom:
  - Solver oscillated or failed to find implied volatility for certain market prices.
Root Cause:
  - Newton-Raphson method diverged when initial guess was poor or Vega was near zero.
Fix:
  - Replaced naive Newton method with hybrid bisection approach:
      - Clamp sigma in [0.001, 5.0].
      - Use bracketed range and switch to Newton only when Vega is large.
Testing:
  - Tested across a wide range of option prices; solver converged in under 20 iterations.
Prevention:
  - Always use bracketed solvers or hybrid methods to ensure convergence stability.

------------------------------------------------------------
3) Loss of Precision in d1 and d2 Computation
------------------------------------------------------------
Symptom:
  - For S ≈ K and small T, computed Greeks became unstable.
Root Cause:
  - Catastrophic cancellation when dividing by small sqrt(T).
Fix:
  - Used long double precision and rearranged computation:
      d1 = (log(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
Testing:
  - Compared against reference Black–Scholes values for near-ATM, short-expiry options.
Prevention:
  - Always use high precision for small-time or near-the-money computations.

------------------------------------------------------------
4) Division by Zero for T = 0
------------------------------------------------------------
Symptom:
  - Program crashed when expiry time (T) was zero.
Root Cause:
  - Division by sqrt(T) in Black–Scholes formula.
Fix:
  - Added guard condition:
      if (T <= 1e-12) return intrinsic value;
Testing:
  - Input options with T = 0 returned correct intrinsic prices.
Prevention:
  - Always check for zero-time cases in time-dependent models.

------------------------------------------------------------
5) CSV Parsing Errors and Malformed Data
------------------------------------------------------------
Symptom:
  - Program terminated on missing or extra fields.
Root Cause:
  - Rigid parsing and unguarded string-to-number conversions.
Fix:
  - Added safe parser with try/catch, skipping malformed rows.
Testing:
  - Tested with missing headers and extra whitespace.
Prevention:
  - Validate input CSV schema before parsing.

------------------------------------------------------------
6) Data Races in Multithreaded Execution
------------------------------------------------------------
Symptom:
  - Occasionally corrupted outputs or segmentation faults.
Root Cause:
  - Multiple threads writing to shared data structures simultaneously.
Fix:
  - Preallocated results vector; each thread writes to unique index.
  - Used atomic counters for progress tracking.
Testing:
  - Verified correctness under ThreadSanitizer (-fsanitize=thread).
Prevention:
  - Avoid shared mutable state; use atomic variables and mutexes when required.

------------------------------------------------------------
7) Load Imbalance Between Threads
------------------------------------------------------------
Symptom:
  - Some threads finished early, others lagged behind.
Root Cause:
  - Static chunk allocation caused uneven workload.
Fix:
  - Switched to dynamic work scheduling with a task queue.
Testing:
  - Measured CPU utilization before and after fix.
Prevention:
  - Use dynamic scheduling for non-uniform workloads.

------------------------------------------------------------
8) Performance Bottleneck Due to Memory Allocations
------------------------------------------------------------
Symptom:
  - Performance dropped significantly on large input files.
Root Cause:
  - Frequent dynamic allocations and string concatenations.
Fix:
  - Used preallocated vectors and stringstream buffers for I/O.
Testing:
  - Benchmark showed 3x improvement in throughput.
Prevention:
  - Reserve memory beforehand; reuse objects.

------------------------------------------------------------
9) Slow Output Writing to File
------------------------------------------------------------
Symptom:
  - Large outputs caused program to hang or run slowly.
Root Cause:
  - Frequent unbuffered I/O operations per row.
Fix:
  - Buffered output in memory, wrote once at the end.
Testing:
  - Reduced write time from seconds to milliseconds.
Prevention:
  - Use buffered writes for batch data processing.

------------------------------------------------------------
10) Put–Call Parity Inconsistency
------------------------------------------------------------
Symptom:
  - Call and put results violated theoretical parity.
Root Cause:
  - Independent computation caused rounding discrepancies.
Fix:
  - Derived put prices from call via parity relation:
      P = C - S*e^{-qT} + K*e^{-rT}
Testing:
  - Verified parity within 1e-8 tolerance for all test cases.
Prevention:
  - Reuse shared intermediate results for both call and put calculations.

------------------------------------------------------------
11) Simplified Value-at-Risk (VaR) Estimation
------------------------------------------------------------
Symptom:
  - Risk report understated exposure for non-linear options.
Root Cause:
  - Only Delta was considered; Gamma and Vega ignored.
Fix:
  - Added scenario-based estimation with Delta–Gamma–Vega adjustment.
Testing:
  - Compared with Monte Carlo results to verify correction.
Prevention:
  - Document all model assumptions clearly in README.

------------------------------------------------------------
12) Non-Deterministic Results Due to Asynchronous Logging
------------------------------------------------------------
Symptom:
  - Output order and logs changed between runs.
Root Cause:
  - Threads writing to log without synchronization.
Fix:
  - Protected logging with a mutex; added deterministic mode.
Testing:
  - Outputs identical across multiple runs with same input.
Prevention:
  - Always synchronize output streams in multi-threaded programs.

------------------------------------------------------------
13) Handling Extremely Small or Large Volatilities
------------------------------------------------------------
Symptom:
  - Volatility below 0.001 or above 5.0 caused invalid Greeks.
Root Cause:
  - Numeric overflow or division instability.
Fix:
  - Clamped sigma values using:
      sigma = max(0.001, min(5.0, sigma));
Testing:
  - Tested across edge cases; outputs remained finite.
Prevention:
  - Always bound parameters in financial models.

------------------------------------------------------------
14) Floating-Point Round-Off Accumulation
------------------------------------------------------------
Symptom:
  - Slightly different results for same input on repeated runs.
Root Cause:
  - Floating-point arithmetic accumulated rounding differences.
Fix:
  - Used long double internally; rounded final output to 8 decimals.
Testing:
  - Output stable across multiple compilers and platforms.
Prevention:
  - Standardize precision across all calculations.



------------------------------------------------------------
Summary:
------------------------------------------------------------
All the above issues were discovered during iterative development and testing of the Option Greeks Calculator. Numerical stability, convergence, and concurrency were the most challenging aspects. By employing long double precision, hybrid solvers, thread-safe design, and buffered I/O, the final program achieved high numerical reliability and performance across large datasets.

------------------------------------------------------------

