#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Residual (mass-transport/coverage) overpotential fit under one-lambda framework.

Model:
  eta_res(BD; eta_con, k_theta) = eta_con + b*log10( BD / (BD - j*k_theta) )

We enforce BD - j*k_theta > 0 for all data by bounding k_theta:
  0 < k_theta < (1 - eps) * min(BD)/j

Input : 'input_fit_res_nls.xlsx' (two cols: BD, eta_res; no header)
Output: 'result_fit_res_nls.xlsx' with sheets: params, data_fit, metrics, settings
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

# ----- Settings -----
INPUT_PATH  = "input_fit_res_nls.xlsx"
RESULT_PATH = "result_fit_res_nls.xlsx"
j   = 4.0        # A/cm^2
# Use intrinsic b from kinetics fit (fit_kin_nls.py). Keep a sensible default:
b   = 0.10       # V/dec
eps = 0.2       # safety margin for positivity (BD - j*k_theta >= eps*BD)
use_relative_residuals = False
# Initial guesses (eta_con in volts; k_theta will be clipped into bounds)
eta_con_init = 0.05
k_theta_init = 0.05
# --------------------

def eta_res_model(BD, eta_con, k_theta):
    denom = BD - j*k_theta
    return eta_con + b * np.log10(BD / denom)

def fit_params(BD, y, eta0, k0, lb_k, ub_k):
    # least_squares with bounds on (eta_con, k_theta)
    # eta_con is unbounded; k_theta is [lb_k, ub_k]
    x0 = np.array([eta0, np.clip(k0, lb_k + 1e-12, ub_k - 1e-12)], dtype=float)

    def resid_fun(theta):
        eta_con, k_theta = theta
        if k_theta <= lb_k or k_theta >= ub_k:
            return 1e6*np.ones_like(y)
        denom = BD - j*k_theta
        # numerical guard (should be positive by construction)
        if np.any(denom <= 0):
            return 1e6*np.ones_like(y)
        yhat = eta_res_model(BD, eta_con, k_theta)
        r = y - yhat
        return r/np.maximum(np.abs(y),1e-9) if use_relative_residuals else r

    lb = np.array([-np.inf, lb_k], dtype=float)
    ub = np.array([ np.inf,  ub_k], dtype=float)
    return least_squares(resid_fun, x0=x0, bounds=(lb, ub), method="trf")

def main():
    # Load data
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_res.")
    BD = df.iloc[:,0].to_numpy(dtype=float)
    y  = df.iloc[:,1].to_numpy(dtype=float)
    n  = y.size

    # Compute data-driven bound
    BD_min = float(np.min(BD))
    if BD_min <= 0:
        raise ValueError("All BD must be positive.")
    k_upper = (1.0 - eps) * (BD_min / j)
    k_lower = 1e-12
    if k_upper <= k_lower:
        raise ValueError("Computed k_theta upper bound is non-positive. Check BD and j.")

    # Full-data fit
    sol = fit_params(BD, y, eta_con_init, k_theta_init, k_lower, k_upper)
    eta_con_hat, k_theta_hat = map(float, sol.x)

    # Covariance & SE from LM Jacobian on absolute residuals
    r_abs = y - eta_res_model(BD, eta_con_hat, k_theta_hat)
    p = 2; dof = max(n - p, 1)
    sigma2 = float(r_abs @ r_abs) / dof
    J = sol.jac  # (n,2)
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
        cov = sigma2 * JTJ_inv
        se_eta, se_k = np.sqrt(np.diag(cov)).astype(float)
    except np.linalg.LinAlgError:
        cov = np.full((2,2), np.nan)
        se_eta, se_k = np.nan, np.nan

    # 95% CIs (normal approx)
    tval = 1.96
    ci_eta = (eta_con_hat - tval*se_eta,  eta_con_hat + tval*se_eta)
    ci_k   = (k_theta_hat - tval*se_k,    k_theta_hat + tval*se_k)

    # R^2 on original scale
    yhat = eta_res_model(BD, eta_con_hat, k_theta_hat)
    R2 = 1.0 - float(np.sum((y - yhat)**2))/float(np.sum((y - np.mean(y))**2))

    # Q^2 via LOO refitting (respecting bounds)
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        BD_tr, y_tr = BD[idx], y[idx]
        try:
            sol_i = fit_params(BD_tr, y_tr, eta_con_hat, k_theta_hat, k_lower, k_upper)
            eta_i, k_i = map(float, sol_i.x)
            yhat_loo[i] = eta_res_model(np.array([BD[i]]), eta_i, k_i)[0]
        except Exception:
            yhat_loo[i] = np.nan
    ok = np.isfinite(yhat_loo)
    if ok.sum() >= 3:
        ybar = float(np.nanmean(y))
        ss_res_loo = float(np.nansum((y - yhat_loo)**2))
        ss_tot = float(np.nansum((y - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save outputs
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["eta_con","k_theta","b_used","j","k_theta_lb","k_theta_ub","eps"],
            "value": [eta_con_hat, k_theta_hat, b, j, k_lower, k_upper, eps],
            "se":    [se_eta,      se_k,        np.nan, np.nan, np.nan, np.nan, np.nan],
            "ci_low":[ci_eta[0],   ci_k[0],     np.nan, np.nan, np.nan, np.nan, np.nan],
            "ci_high":[ci_eta[1],  ci_k[1],     np.nan, np.nan, np.nan, np.nan, np.nan],
            "note":  ["NLS (95% CI)","NLS (95% CI)","","","data-driven bound","",""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "eta_res_obs": y, "eta_res_fit": yhat, "residual": y - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

        pd.DataFrame({
            "setting": ["use_relative_residuals","eps","b_source"],
            "value":   [use_relative_residuals, eps, "Use b from kinetics fit"]
        }).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
