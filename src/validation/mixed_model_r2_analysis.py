import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# ==============================
# LOAD DATA
# ==============================

cpue = pd.read_csv("cpue_with_global_regime.csv")

cpue["log_cpue"] = np.log1p(cpue["CPUE"])
cpue["log_effort"] = np.log1p(cpue["EFFORT"])

cpue["regime"] = cpue["regime"].astype("category")
cpue["MONTH"] = cpue["MONTH"].astype("category")
cpue["YEAR"] = cpue["YEAR"].astype("category")

# ==============================
# MODEL WITHOUT REGIME
# ==============================

model_base = smf.mixedlm(
    "log_cpue ~ MONTH + log_effort",
    cpue,
    groups=cpue["YEAR"]
)

result_base = model_base.fit()

# ==============================
# MODEL WITH REGIME
# ==============================

model_full = smf.mixedlm(
    "log_cpue ~ regime + MONTH + log_effort",
    cpue,
    groups=cpue["YEAR"]
)

result_full = model_full.fit()

# ==============================
# FUNCTION TO COMPUTE R2
# ==============================

def compute_r2(model, result, data):

    # Fixed effect predictions
    fixed_pred = result.predict(data)

    var_fixed = np.var(fixed_pred)

    # Random effect variance
    var_random = result.cov_re.iloc[0,0]

    # Residual variance
    var_residual = result.scale

    r2_marginal = var_fixed / (var_fixed + var_random + var_residual)
    r2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)

    return r2_marginal, r2_conditional


r2_base = compute_r2(model_base, result_base, cpue)
r2_full = compute_r2(model_full, result_full, cpue)

print("\nMODEL WITHOUT REGIME")
print("Marginal R2:", round(r2_base[0], 4))
print("Conditional R2:", round(r2_base[1], 4))

print("\nMODEL WITH REGIME")
print("Marginal R2:", round(r2_full[0], 4))
print("Conditional R2:", round(r2_full[1], 4))

print("\nΔ Marginal R2 (regime contribution):",
      round(r2_full[0] - r2_base[0], 4))