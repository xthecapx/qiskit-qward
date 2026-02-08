# Statistical Analysis Reference

## Test Selection Guide

### Comparing Two Groups
| Data Type | Normal | Non-normal |
|-----------|--------|------------|
| Independent | Independent t-test | Mann-Whitney U |
| Paired | Paired t-test | Wilcoxon signed-rank |
| Binary | Chi-square / Fisher's exact | -- |

### Comparing 3+ Groups
| Data Type | Normal | Non-normal |
|-----------|--------|------------|
| Independent | One-way ANOVA | Kruskal-Wallis |
| Paired | Repeated measures ANOVA | Friedman |

### Relationships
- Two continuous: Pearson (normal) or Spearman (non-normal)
- Continuous outcome + predictors: Linear regression
- Binary outcome + predictors: Logistic regression

## Assumption Checking

### Normality
```python
from scipy import stats

# Shapiro-Wilk test (n < 5000)
stat, p = stats.shapiro(data)
print(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4f}")

# Q-Q plot
import matplotlib.pyplot as plt
stats.probplot(data, dist="norm", plot=plt)
plt.show()
```

### Homogeneity of Variance
```python
# Levene's test
stat, p = stats.levene(group1, group2)
print(f"Levene's: F={stat:.4f}, p={p:.4f}")
```

### When Assumptions Fail
- Mild normality violation + n > 30: proceed with parametric
- Moderate violation: use non-parametric alternative
- Heteroscedasticity: use Welch's t-test or robust SEs

## Running Tests

### T-Tests
```python
import pingouin as pg

# Independent t-test
result = pg.ttest(group_a, group_b, correction='auto')
t_stat = result['T'].values[0]
p_val = result['p-val'].values[0]
d = result['cohen-d'].values[0]
```

### ANOVA
```python
# One-way ANOVA
aov = pg.anova(dv='score', between='group', data=df, detailed=True)

# Post-hoc (if significant)
if aov['p-unc'].values[0] < 0.05:
    posthoc = pg.pairwise_tukey(dv='score', between='group', data=df)
```

### Correlation
```python
# Spearman (recommended for QWARD metrics)
corr = pg.corr(x, y, method='spearman')
r = corr['r'].values[0]
p = corr['p-val'].values[0]
```

### Regression
```python
import statsmodels.api as sm

X = sm.add_constant(X_data)  # Always add constant
model = sm.OLS(y, X).fit()
print(model.summary())

# Diagnostics
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, X)
print(f"Breusch-Pagan p={bp_test[1]:.4f}")
```

## Effect Sizes

| Test | Effect Size | Small | Medium | Large |
|------|-------------|-------|--------|-------|
| T-test | Cohen's d | 0.20 | 0.50 | 0.80 |
| ANOVA | eta-squared | 0.01 | 0.06 | 0.14 |
| Correlation | r | 0.10 | 0.30 | 0.50 |
| Regression | R-squared | 0.02 | 0.13 | 0.26 |

Always report effect sizes with confidence intervals.

## Power Analysis

```python
from statsmodels.stats.power import tt_ind_solve_power

# Required sample size for d=0.5
n = tt_ind_solve_power(effect_size=0.5, alpha=0.05, power=0.80)
print(f"Required n per group: {n:.0f}")

# Detectable effect with n=50
d = tt_ind_solve_power(effect_size=None, nobs1=50, alpha=0.05, power=0.80)
print(f"Detectable d >= {d:.2f}")
```

## Multiple Comparisons

```python
from statsmodels.stats.multitest import multipletests

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')

# FDR (Benjamini-Hochberg)
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
```

## APA Reporting Templates

### T-Test
```
Group A (n = 48, M = 75.2, SD = 8.5) scored significantly higher than
Group B (n = 52, M = 68.3, SD = 9.2), t(98) = 3.82, p < .001, d = 0.77,
95% CI [0.36, 1.18].
```

### ANOVA
```
A one-way ANOVA revealed a significant main effect of condition on scores,
F(2, 147) = 8.45, p < .001, eta-squared = .10.
```

### Regression
```
The model was significant, F(3, 146) = 45.2, p < .001, R-squared = .48.
Study hours (B = 1.80, beta = .35, p < .001) and GPA (B = 8.52,
beta = .28, p < .001) were significant predictors.
```

## Bayesian Alternatives

```python
import pymc as pm
import arviz as az

with pm.Model() as model:
    mu1 = pm.Normal('mu_group1', mu=0, sigma=10)
    mu2 = pm.Normal('mu_group2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    y1 = pm.Normal('y1', mu=mu1, sigma=sigma, observed=group_a)
    y2 = pm.Normal('y2', mu=mu2, sigma=sigma, observed=group_b)
    diff = pm.Deterministic('difference', mu1 - mu2)
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

print(az.summary(trace, var_names=['difference']))
prob = (trace.posterior['difference'].values > 0).mean()
print(f"P(mu1 > mu2 | data) = {prob:.3f}")
```

## Common Pitfalls

1. P-hacking: don't test multiple ways until significant
2. Confusing significance with importance
3. Not reporting effect sizes
4. Ignoring assumptions
5. Multiple comparisons without correction
6. Post-hoc power analysis (use sensitivity analysis instead)
7. Misinterpreting p-values as P(hypothesis is true)
