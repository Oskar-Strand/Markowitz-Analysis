import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("================= Part a =================")
prices = pd.read_excel('oslo-stocks-2023-2025.xlsx', sheet_name='Adjusted Prices',header = 2, index_col=0)
print (prices.head())
#formula: r{i,k} = (S{i,t{k}}) - S{i-1,t{k}}) / S{i-1,t{k}})
S_prev = prices.shift(1)
returns = (prices - S_prev) / S_prev
returns = returns.dropna()

#Sample mean return for each stock, the sample is the average of the returns for each stock over the time period considered.

sample_mean_return = returns.mean()

sample_variance = returns.var()

sample_volatility = returns.std()

sample_covariance = returns.cov()

V = sample_covariance.to_numpy()

n = len(returns)
print("n =" + str(n))
print("Is V symmetric?", np.allclose(V, V.T))
print("Does diagonal entries agree with sample_variance?", np.allclose(np.diag(V), sample_variance.to_numpy()))
print("Sigma = sqrt(var)?", np.allclose(sample_volatility, np.sqrt(sample_variance)))

print(returns.head())
print(sample_mean_return)
print(sample_variance)
print(sample_volatility)
print(sample_covariance)

# ================= Part b =================
print("================= Part b =================")
# Vector of ones, e = (1, ..., 1)^T
e = np.ones(len(sample_mean_return))
# Sample mean return vector mu_hat
mr = sample_mean_return.to_numpy()
# Inverse covariance matrix times vector of ones: V_hat^{-1} e
Vinv_e = np.linalg.solve(V, e)
# Inverse covariance matrix times mean return vector: V_hat^{-1} mu_hat
Vinv_mr = np.linalg.solve(V, mr)

# Markowitz scalars A, B, C
A = e.T @ Vinv_e      # A = e^T V^{-1} e
B = mr.T @ Vinv_e     # B = mu^T V^{-1} e
C = mr.T @ Vinv_mr    # C = mu^T V^{-1} mu

print("A, B, C:", A, B, C)
# Must be positive
print("AC - B^2:", A*C - B**2)

# Global minimum-variance portfolio weights: w_G = V^{-1} e / A
w_G = Vinv_e / A

# Minimum-variance portfolio with target expected return m
def w_m(m):
    return ((C - m*B)/(A*C - B**2)) * Vinv_e + ((m*A - B)/(A*C - B**2)) * Vinv_mr

# Tangent portfolio for risk-free return r: w_M = V^{-1}(mu - r e) / (B - A r)
def w_M(r):
    return np.linalg.solve(V, mr - r*e) / (B - A*r)

# Risk (volatility) of each stock i: sigma_i = sqrt(V_ii)
sigma_i = np.sqrt(np.diag(V))

# Expected return and risk of the GMV portfolio
mr_G = w_G @ mr                      # mu_G = w_G^T mu
sigma_G = np.sqrt(w_G @ V @ w_G)     # sigma_G = sqrt(w_G^T V w_G)

# GMV weights per stock, and checks against the closed forms B/A and 1/sqrt(A)
print(pd.Series(w_G, index=sample_mean_return.index))
print("Sum of weights:", w_G.sum())
print("mu_G:", mr_G, "  B/A:", B/A)
print("sigma_G:", sigma_G, "  1/sqrt(A):", 1/np.sqrt(A))

#Sanity checks
print("mu_G = B/A?", np.isclose(mr_G, B/A))
print("sigma_G = 1/sqrt(A)?", np.isclose(sigma_G, 1/np.sqrt(A)))
print("A > 0, C > 0?, AC-B^2:", A > 0, C > 0, A*C - B**2)

# ================= Part c =================
print("================= Part c =================")
# 400 target returns m
m_grid = np.linspace(min(mr.min()-0.0005, mr_G), mr.max()+0.0005, 400)

# For each m: build the best portfolio w_m and compute its risk sqrt(w^T V w).
# This gives one point (risk, m) on the frontier per value of m.
sigma_frontier = np.array([np.sqrt(w_m(m) @ V @ w_m(m)) for m in m_grid])

# Create an empty figure
fig, ax = plt.subplots(figsize=(8, 6))

# The frontier: risk on the x-axis, expected return on the y-axis
ax.plot(sigma_frontier, m_grid, color="navy",
        label="Minimum-variance frontier")

# The five stocks as red dots at (sigma_i, mu_i).
ax.scatter(sigma_i, mr, color="darkred", zorder=3)

# Write each stock's name next to its dot
for name, s, m in zip(sample_mean_return.index, sigma_i, mr):
    ax.annotate(name, (s, m), textcoords="offset points", xytext=(6, 4))

# The GMV portfolio as a large green dot
ax.scatter(sigma_G, mr_G, color="green", marker="o", s=100, zorder=3,
           label="GMV portfolio")

# Axis labels
ax.set_xlabel(r"Risk $\hat\sigma$")
ax.set_ylabel(r"Expected return $\hat\mu$")

ax.set_xlim(left=0)

# Light grid lines and a legend explaining the line and the dots
ax.grid(alpha=0.3)
ax.legend()

plt.savefig("frontier.png", dpi=200, bbox_inches="tight")

# ================= Part d =================
print("================= Part d =================")
# Daily risk-free return
r = 0.000117

# Tangent portfolio weights
w_tan = w_M(r)
print(pd.Series(w_tan, index=sample_mean_return.index))
print("Sum of weights:", w_tan.sum())

# Expected return, risk and Sharpe ratio of the tangent portfolio
mr_M = w_tan @ mr
sigma_M = np.sqrt(w_tan @ V @ w_tan)
sharpe_M = (mr_M - r) / sigma_M

print("mu_M:", mr_M)
print("sigma_M:", sigma_M)
print("Sharpe ratio:", sharpe_M)

# Allocation for m1: w0 = weight in risk-free asset, 1 - w0 = weight in tangent
m1 = 0.00055
w0_1 = (mr_M - m1) / (mr_M - r)
print("m1: w0 =", w0_1, " 1 - w0 =", 1 - w0_1)

# Allocation for m2
m2 = 0.00119
w0_2 = (mr_M - m2) / (mr_M - r)
print("m2: w0 =", w0_2, " 1 - w0 =", 1 - w0_2)

# Plot: frontier + capital market line
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(sigma_frontier, m_grid, color="navy", label="Minimum-variance frontier")
ax.scatter(sigma_i, mr, color="darkred")
ax.scatter(sigma_G, mr_G, color="green", s=100, label="GMV portfolio")

# Capital market line: mu = r + Sharpe * sigma
sigma_cml = np.linspace(0, sigma_frontier.max(), 200)
ax.plot(sigma_cml, r + sharpe_M * sigma_cml, color="orange", label="Capital market line")
ax.scatter(sigma_M, mr_M, color="orange", s=100, label="Tangent portfolio")

ax.set_xlabel("Risk")
ax.set_ylabel("Expected return")
ax.legend()
plt.savefig("cml.png", dpi=200)
