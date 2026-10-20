# Markowitz Analysis of Oslo Stocks

Empirical mean-variance (Markowitz) portfolio analysis of five stocks on the Oslo Stock Exchange: Equinor, DNB Bank, Telenor, Norwegian Air Shuttle and Norsk Hydro, using daily adjusted closing prices from January 2023 to December 2025.

The project was made as part of a course in mathematical finance at the University of Oslo.

## What the code does

- Computes daily simple returns from adjusted closing prices
- Estimates the sample mean returns, variances, volatilities and covariance matrix
- Computes the Markowitz scalars A, B and C and the global minimum-variance (GMV) portfolio
- Plots the minimum-variance frontier together with the individual stocks and the GMV portfolio
- Computes the tangent portfolio, its Sharpe ratio and the capital market line for a given risk-free rate
- Finds the allocation between the risk-free asset and the tangent portfolio for a target return

All quantities are daily. The portfolio formulas are implemented directly with NumPy, without optimisation libraries.

## Requirements

- Python 3
- pandas, numpy, matplotlib, openpyxl

```
pip install pandas numpy matplotlib openpyxl
```

## Usage

Place the data file `oslo-stocks-2023-2025.xlsx` (sheet `Adjusted Prices`) in the same folder as `main.py`, then run:

```
python main.py
```

The script prints the estimates and checks, and saves the plots as `frontier.png` and `cml.png`.
