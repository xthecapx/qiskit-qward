# Machine Learning Reference (scikit-learn)

## Complete Pipeline Pattern (Always Use)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

numeric_features = ['depth', 'gate_count', 'gate_density']
categorical_features = ['backend', 'optimization_level']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Data Splitting

```python
from sklearn.model_selection import train_test_split

# Stratified split for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Time series: use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

## Supervised Learning

### Classification
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Regression
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
```

### Algorithm Selection
- **Linear models**: interpretable, fast, good baseline
- **Tree-based**: handle non-linearity, no scaling needed
- **SVM**: works well with small-medium datasets
- **KNN**: simple, no training, slow prediction

## Unsupervised Learning

### Clustering
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Find optimal k
scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

optimal_k = range(2, 11)[np.argmax(scores)]
```

### Dimensionality Reduction
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# t-SNE (for visualization only)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_scaled)
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200, 500],
    'classifier__max_depth': [5, 10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.4f}")
best_model = grid.best_estimator_
```

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
```

## Feature Importance

```python
# Tree-based models
importances = model.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_names[sorted_idx][:15], importances[sorted_idx][:15])
ax.set_xlabel('Feature Importance')
```

## Preprocessing Reference

### Scaling (required for SVM, KNN, neural networks)
- `StandardScaler`: zero mean, unit variance
- `MinMaxScaler`: bounded [0, 1]
- `RobustScaler`: robust to outliers

### NOT required for tree-based models
- Decision Trees, Random Forest, Gradient Boosting, XGBoost

### Encoding
- `OneHotEncoder`: nominal categories
- `OrdinalEncoder`: ordered categories
- `LabelEncoder`: target variable only

### Missing Values
- `SimpleImputer`: mean, median, most_frequent
- `KNNImputer`: k-nearest neighbors

## Best Practices

1. **Always use Pipelines** to prevent data leakage
2. **Never fit on test data**: `scaler.fit_transform(X_train)`, `scaler.transform(X_test)`
3. **Stratified splits** for classification
4. **Set random_state** for reproducibility
5. **Cross-validate** before reporting metrics
6. **Choose metrics wisely**: balanced accuracy for imbalanced data

## Common Issues

### ConvergenceWarning
```python
model = LogisticRegression(max_iter=1000)  # increase max_iter
```

### Memory with large datasets
```python
from sklearn.linear_model import SGDClassifier  # stochastic gradient
from sklearn.cluster import MiniBatchKMeans  # mini-batch
```
