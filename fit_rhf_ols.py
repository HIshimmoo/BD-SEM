#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OLS for R_HF vs 1/BD, then λ_RHF via delta method.

Input : 'input_fit_rhf_ols.xlsx' with two cols: BD, eta_RHF (V), no header
Output: 'result_fit_rhf_ols.xlsx' with params, data_fit, metrics
"""
import pandas as pd
import numpy as np

INPUT_PATH  = "input_fit_rhf_ols.xlsx"
RESULT_PATH = "result_fit_rhf_ols.xlsx"
j          = 4.0       # A/cm^2
R_HF0      = 0.45     # Ω cm^2

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_RHF.")
    BD  = df.iloc[:,0].to_numpy(dtype=float)     # (n,)
    eta = df.iloc[:,1].to_numpy(dtype=float)     # (n,)
    R_HF = eta / j                               # (n,)
    X = 1.0 / BD                                 # (n,)
    n = R_HF.size

    # Full OLS with intercept using 1-D arrays throughout
    Xd = np.column_stack([np.ones(n), X])        # (n,2)
    beta, *_ = np.linalg.lstsq(Xd, R_HF, rcond=None)   # beta is (2,)
    a_hat, b_hat = float(beta[0]), float(beta[1])

    yhat = Xd @ beta                             # (n,)
    resid = R_HF - yhat                          # (n,)

    ybar = float(R_HF.mean())
    ss_res = float(resid @ resid)
    ss_tot = float(((R_HF - ybar)**2).sum())
    R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Covariance & SEs
    p = 2
    dof = max(n - p, 1)
    sigma2 = ss_res / dof
    XtX = Xd.T @ Xd                               # (2,2)
    try:
        XtX_inv = np.linalg.inv(XtX)
        cov = sigma2 * XtX_inv                    # (2,2)
    except np.linalg.LinAlgError:
        cov = np.full((2,2), np.nan)

    se = np.sqrt(np.diag(cov))                    # (2,)
    se_a, se_b = float(se[0]), float(se[1])

    # λ_RHF via delta method
    lam_hat = (a_hat - R_HF0) / b_hat
    # gradient as 1-D vector => scalar variance without (1,1) arrays
    g = np.array([1.0/b_hat, -lam_hat/b_hat])     # (2,)
    var_lambda = float(g @ cov @ g) if np.all(np.isfinite(cov)) else np.nan
    se_lambda  = float(np.sqrt(var_lambda)) if np.isfinite(var_lambda) else np.nan

    # 95% CIs (normal approx)
    tval = 1.96
    ci_a = (a_hat - tval*se_a, a_hat + tval*se_a)
    ci_b = (b_hat - tval*se_b, b_hat + tval*se_b)
    ci_l = (lam_hat - tval*se_lambda, lam_hat + tval*se_lambda) if np.isfinite(se_lambda) else (np.nan, np.nan)

    # LOO-CV Q² with proper refitting; keep 1-D shapes to avoid warnings
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        Xi = X[idx]                             # (n-1,)
        Yi = R_HF[idx]                          # (n-1,)
        Xdi = np.column_stack([np.ones(Xi.size), Xi])  # (n-1,2)
        try:
            bet_i, *_ = np.linalg.lstsq(Xdi, Yi, rcond=None)  # (2,)
            yhat_loo[i] = float(bet_i[0] + bet_i[1]*X[i])
        except Exception:
            yhat_loo[i] = np.nan

    ok = np.isfinite(yhat_loo)
    if ok.sum() >= 3:
        ybar = float(R_HF.mean())
        ss_res_loo = float(np.nansum((R_HF - yhat_loo)**2))
        ss_tot = float(np.sum((R_HF - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["a","b","lambda_RHF","j","R_HF0"],
            "value": [a_hat, b_hat, lam_hat, j, R_HF0],
            "se":    [se_a,  se_b,  se_lambda, np.nan, np.nan],
            "ci_low":[ci_a[0], ci_b[0], ci_l[0], np.nan, np.nan],
            "ci_high":[ci_a[1], ci_b[1], ci_l[1], np.nan, np.nan],
            "note": ["OLS (95% CI)", "OLS (95% CI)", "delta-method CI", "", ""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD,
            "inv_BD": X,
            "R_HF_obs": R_HF,
            "R_HF_fit": yhat,
            "residual": resid
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

if __name__ == "__main__":
    main()
