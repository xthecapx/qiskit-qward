# Statistical Modeling Reference (statsmodels)

## Linear Regression (OLS)

```python
import statsmodels.api as sm

X = sm.add_constant(X_data)  # ALWAYS add constant
model = sm.OLS(y, X).fit()
print(model.summary())

# Key results
print(f"R-squared: {model.rsquared:.4f}")
print(f"Coefficients: {model.params}")
print(f"P-values: {model.pvalues}")

# Predictions with CIs
predictions = model.get_prediction(X_new)
pred_df = predictions.summary_frame()
```

## Formula API (R-style)

```python
import statsmodels.formula.api as smf

# OLS
results = smf.ols('y ~ x1 + x2 + x1:x2', data=df).fit()

# Categorical variables
results = smf.ols('y ~ x1 + C(category)', data=df).fit()

# Polynomial
results = smf.ols('y ~ x + I(x**2)', data=df).fit()

# Logistic regression
results = smf.logit('y ~ x1 + x2 + C(group)', data=df).fit()
```

## Generalized Linear Models

```python
# Poisson regression (count data)
model = sm.GLM(y_counts, X, family=sm.families.Poisson()).fit()

# Check overdispersion
overdispersion = model.pearson_chi2 / model.df_resid
if overdispersion > 1.5:
    # Use Negative Binomial
    from statsmodels.discrete.count_model import NegativeBinomial
    model = NegativeBinomial(y_counts, X).fit()
```

### Family selection
- **Gaussian**: continuous, normal errors (= OLS)
- **Binomial**: binary outcomes (logistic regression)
- **Poisson**: count data
- **Negative Binomial**: overdispersed counts
- **Gamma**: positive continuous, right-skewed

## Logistic Regression

```python
from statsmodels.discrete.discrete_model import Logit
import numpy as np

X = sm.add_constant(X_data)
model = Logit(y_binary, X).fit()
print(model.summary())

# Odds ratios
odds_ratios = np.exp(model.params)
print(f"Odds ratios: {odds_ratios}")

# Predicted probabilities
probs = model.predict(X)

# Marginal effects
marginal = model.get_margeff()
print(marginal.summary())
```

## Time Series (ARIMA)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Check stationarity
adf = adfuller(y_series)
print(f"ADF p-value: {adf[1]:.4f}")

# ACF/PACF to identify p, q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(y_series, lags=40, ax=ax1)
plot_pacf(y_series, lags=40, ax=ax2)

# Fit ARIMA
model = ARIMA(y_series, order=(1, 1, 1)).fit()
print(model.summary())

# Forecast
forecast = model.get_forecast(steps=10)
forecast_df = forecast.summary_frame()

# Diagnostics
model.plot_diagnostics(figsize=(12, 8))
```

## Model Comparison

```python
import pandas as pd

# AIC/BIC (lower = better)
comparison = pd.DataFrame({
    'AIC': {name: res.aic for name, res in models.items()},
    'BIC': {name: res.bic for name, res in models.items()},
}).sort_values('AIC')

# Likelihood ratio test (nested models only)
from scipy import stats
lr_stat = 2 * (full_model.llf - reduced_model.llf)
df_diff = full_model.df_model - reduced_model.df_model
p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
```

## Diagnostics

```python
# Heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
bp = het_breuschpagan(model.resid, X)
print(f"Breusch-Pagan p={bp[1]:.4f}")

# Autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(model.resid)

# Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Normality of residuals
from statsmodels.stats.stattools import jarque_bera
jb = jarque_bera(model.resid)

# Influence
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model)
cooks_d = influence.cooks_distance[0]
```

## Robust Standard Errors

```python
# Heteroscedasticity-consistent (HC)
model_robust = model.get_robustcov_results(cov_type='HC3')

# HAC (Newey-West) for time series
model_hac = model.get_robustcov_results(cov_type='HAC', maxlags=5)

# Cluster-robust
model_cluster = model.get_robustcov_results(
    cov_type='cluster', groups=cluster_ids
)
```

## Common Pitfalls

1. Forgetting `sm.add_constant()` (no intercept)
2. Using OLS for binary outcomes (use Logit)
3. Using Poisson with overdispersion (use NegBin)
4. Ignoring residual diagnostics
5. Not using robust SEs when heteroscedasticity present
6. Fitting ARIMA on non-stationary data
7. Comparing non-nested models with LR test (use AIC/BIC)
