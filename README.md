# Greek-Calculator

**High-performance options analytics** — Black–Scholes pricing, analytic Greeks, implied volatility extraction, batched processing with multithreading, and simple portfolio risk reports. Intended for quantitative research, teaching, and production prototyping.

---

## Table of contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick start (build & run)](#quick-start-build--run)
4. [Command-line usage](#command-line-usage)
5. [Input CSV specification](#input-csv-specification)
6. [Large example input (CSV)](#large-example-input-csv)
7. [Example output (CSV) & sample rows](#example-output-csv--sample-rows)
8. [Mathematical notes](#mathematical-notes)
9. [Numerical stability & edge cases](#numerical-stability--edge-cases)
10. [Architecture & internals](#architecture--internals)
11. [Testing & validation](#testing--validation)
12. [Troubleshooting & FAQs](#troubleshooting--faqs)
13. [Extensibility & future work](#extensibility--future-work)
14. [License](#license)

---

## Overview

Greek-Calculator is a single-file reference implementation and small toolkit for option pricing and sensitivity analysis. It computes theoretical prices under the Black–Scholes–Merton model, analytic Greeks (Delta, Gamma, Vega, Theta, Rho and several second-order Greeks), implied volatility (hybrid root-finder), and basic portfolio-level risk measures (delta/gamma exposures, simple VaR/ES heuristics).

The program is optimized for batch throughput (multithreaded processing), numerical robustness (guarded tail handling and intrinsic fallbacks), and auditable deterministic outputs.

---

## Features

* Black–Scholes closed-form pricing (calls & puts) with continuous dividend yield.
* First- and higher-order Greeks (Delta, Gamma, Vega, Theta, Rho, Vanna, Vomma, Charm, Zomma, Speed).
* Implied-vol solver (hybrid Newton–bisection with clipping and safeguards).
* Buffered CSV input/output and large-batch processing.
* Thread pool / chunked parallelism with lock-free hot path and minimal synchronization.
* Portfolio aggregation: exposures (dollar delta, gamma, vega) and a simple delta–gamma VaR approximation.
* Defensive handling of edge cases (T→0, σ≈0, deep tails).

---

## Quick start (build & run)

**Save** the supplied source file as `greek_calculator.cpp`.

**Build (Linux/macOS):**

```
g++ -std=c++17 -O2 -pthread greek_calculator.cpp -o greek_calc
```

**Run:**

```
./greek_calc <input.csv> <output.csv> [threads]
```

Example:

```
./greek_calc options_input.csv options_results.csv 4
```

* `threads` is optional; if omitted the program uses hardware concurrency.
* By default the program writes `options_results.csv` and `options_results.csv.risk_report.txt` (when portfolio reporting is enabled).

---

## Command-line usage

```
Usage: greek_calc <input_csv> <output_csv> [threads]

Positional arguments:
  input_csv     CSV file with option rows
  output_csv    Processed CSV with prices, greeks and exposures
  threads       Optional: number of worker threads (default = auto)

Options (configurable via header or compile-time flags):
  --no-portfolio    Skip portfolio aggregation and risk report
  --precision=N     Decimal precision for numeric output (default: 8)
  --deterministic   Use single-threaded deterministic mode
```

---

## Input CSV specification

**Required columns (order-insensitive, case-insensitive):**

| Column     | Type   | Description                                                |
| ---------- | ------ | ---------------------------------------------------------- |
| ID         | string | Unique identifier                                          |
| Spot       | number | Spot price (S > 0)                                         |
| Strike     | number | Strike price (K > 0)                                       |
| Rate       | number | Continuous risk-free rate (r)                              |
| Volatility | number | Annual volatility (σ) — can be blank to request IV solving |
| Time       | number | Years to expiry (T) — > 0 for Greeks                       |
| Type       | string | `call` or `put`                                            |

**Optional columns:** `Dividend_Yield`, `MarketPrice` (observed option price — used for implied vol), `Quantity`, `Notional`, `Underlying`.

Notes:

* Missing `Volatility` but present `MarketPrice` will trigger implied-vol computation.
* Rows with malformed or missing required fields are skipped (logged) unless `--fail-on-error` is set.

---

## Large example input (CSV)

Below is a reasonably large illustrative input (10 rows) you can paste into `options_input.csv`.

```
ID,Spot,Strike,Rate,Volatility,Time,Type,Dividend_Yield,MarketPrice,Quantity
opt-001,100,100,0.05,0.20,0.5,call,0.00,5.50,10
opt-002,100,110,0.05,0.20,0.5,put,0.00,8.20,5
opt-003,50,45,0.04,0.35,0.25,call,0.01,6.10,2
opt-004,150,130,0.03,0.30,1.0,put,0.00,26.75,1
opt-005,250,260,0.05,0.18,0.75,call,0.00,3.10,20
opt-006,80,80,0.06,,0.1667,call,0.00,2.85,15
opt-007,45,60,0.04,0.50,0.2,put,0.00,5.40,7
opt-008,120,100,0.04,0.22,0.05,call,0.00,21.00,3
opt-009,10,8,0.02,0.60,0.02,call,0.00,2.15,50
opt-010,500,450,0.03,0.25,2.0,put,0.01,78.40,1
```

* `opt-006` intentionally omits `Volatility` but supplies `MarketPrice` to request implied-vol solving.
* Times are in years. Small T values (e.g., 0.02) test short-expiry handling.

---

## Example output (CSV) & sample rows

**Output columns (primary):**

```
ID,TheoreticalPrice,MarketPrice,PriceDifference,ImpliedVol,Delta,Gamma,Vega,Theta,Rho,DollarDelta,DollarGamma,Status
```

**Sample rows (illustrative)**

```
opt-001,5.489321,5.50,-0.010679,0.200000,0.598712,0.012345,12.345678,-0.018432,0.234500,5.98712,50.12345,OK
opt-006,2.845100,2.85,-0.004900,0.195700,0.512345,0.020000,6.543210,-0.005432,0.083210,12.68518,18.00000,IV_FOUND
opt-009,2.120000,2.15,-0.030000,0.614000,0.720123,0.305000,0.450000,-0.010000,0.005000,36.00615,0.51000,SHORT_EXPIRY_OK
```

* `Status` indicates processing outcome (`OK`, `IV_FOUND`, `IV_CLAMPED`, `MALFORMED_ROW`, etc.).
* Numeric columns are truncated/rounded per `--precision`.

---

## Mathematical notes

**Black–Scholes inputs & definitions:**

* Geometric Brownian motion with continuous dividend yield `q`.
* Intermediate terms:

[ d_1 = \frac{\ln(S/K) + (r - q + 0.5\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T} ]

* Call price:

[ C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2) ]

* Put price via parity or closed-form.

**Greeks:** analytic formulas are used where available. Theta is reported as per-day decay (i.e., divided by 365).

**Implied volatility:** hybrid Newton–bisection with clipping to `[1e-6, 5.0]`, adaptive step control and explicit checks against intrinsic bounds.

---

## Numerical stability & edge cases

Key defensive measures implemented:

* Safe normal CDF/PDF with explicit tail clamps (e.g., `if (x > +8) N(x)~1`), and `erfc`-based central evaluation for numerical accuracy.
* Intrinsic-value fallback for `T <= 1e-12` (no division by zero).
* Sigma clipping and iteration cap for implied-vol; if IV cannot be found within iterations, status flags `IV_CLAMPED` or `IV_FAILED` are set.
* Preallocated result buffers and per-thread write indices to avoid data races.

---

## Architecture & internals

Processing pipeline (high-level):

1. Read and validate CSV rows into a lightweight `OptionData` structure.
2. Split work into chunks using `chunk_size = ceil(n / threads)` (or use dynamic queueing for highly skewed loads).
3. Each worker computes theoretical price, Greeks, and — if requested — implied volatility.
4. Results are written into a preallocated vector; final CSV is written in a buffered manner.

Concurrency strategy:

* Each worker writes only to its assigned slice of the results vector (lock-free writes).
* Atomic counters are used for progress reporting; only logging and final file writes use mutex protection.

I/O strategy:

* Buffered file writes to avoid per-row flushes.
* Robust CSV parser that tolerates whitespace and extra blank lines; malformed rows are logged and skipped unless configured to fail.

---

## Testing & validation

Recommended tests:

* Unit tests for `norm_cdf`, `norm_pdf`, closed-form prices, analytic Greeks.
* Regression tests for implied-vol solver across a grid of market prices and times.
* Thread-safety tests under ThreadSanitizer and stress tests with very large input files.
* Numerical regression (store golden outputs) to detect platform/compiler differences.

---

## Troubleshooting & FAQs

**Q: Program prints `NaN` for Greeks.**

* A: Confirm `Time > 0`, `Spot > 0`, `Strike > 0`, and `Volatility > 0` (or that `MarketPrice` is in range for IV). Check `Status` for `MALFORMED_ROW`.

**Q: Implied vol hits clamps (e.g., 5.0).**

* A: Market price is outside theoretical bounds given inputs (or price > forward intrinsic). Verify `MarketPrice` and `Dividend_Yield`.

**Q: Output order changes between runs.**

* A: Use `--deterministic` for single-threaded, deterministic ordering.

**Q: Performance is slow.**

* A: Increase `threads`, ensure `-O2` is used, and use buffered I/O. Profile to find hotspots (transcendental calls often dominate).

---

## Extensibility & future work

* Support for multi-asset portfolios and correlation-aware VaR (Monte Carlo engine).
* Add alternative models: local-vol, Heston, Bachelier.
* On-disk memory-mapped CSV streaming for extremely large inputs.
* JSON output and REST API wrapper for integration with other systems.

---

## Contributing

Small, focused pull requests are welcome. Format code with `clang-format` style and include numerical unit tests for any new formula. Add regression test vectors for any change that affects numerical output.

---

## License

This project is provided under the MIT License — see `LICENSE` for details.

---

*If you want a version tailored for teaching (more comments and simpler single-threaded flow) or a `README` without implementation-level sections, tell me which style and I will create that variation.*
