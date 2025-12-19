## Updates for Classification Problem 

### 1. Imports

**(a) Metrics import**

```python
from sklearn.metrics import mean_absolute_error
```

→

```python
from sklearn.metrics import f1_score
```

---

**(b) Model classes**

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
```

→

```python
from sklearn.linear_model import RidgeClassifier   # or LogisticRegression, etc.
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
```

And later, all usages of these regressor classes must match their classifier counterparts (see sections 3–4).

---

### 2. Target variable / classification label

Anywhere you define `y_train` and `y_test`:

```python
y_train = train_set["median_house_value"].copy()
y_test = test_set["median_house_value"].copy()
```

→ replace with your **classification label** column:

```python
y_train = train_set["<your_class_label_col>"].copy()
y_test = test_set["<your_class_label_col>"].copy()
```

(If you’re turning `median_house_value` itself into classes, you’d first bin it into categories and then use that categorical column here.)

---

### 3. Optuna objective functions (NO PCA)

Everywhere you:

1. Build a regressor  
2. Call `cross_val_score(... scoring="neg_mean_absolute_error" ...)`  
3. Return `-scores.mean()`

you must adjust to **classifier + F1**:

#### 3.1 `objective_ridge`

**Model construction**

```python
pipeline = make_pipeline(preprocessing_clone, Ridge(alpha=alpha))
```

→

```python
pipeline = make_pipeline(preprocessing_clone, RidgeClassifier(alpha=alpha))
```

**Scoring + return**

```python
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
)
return -scores.mean()
```

→

```python
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="f1",  # or "f1_macro", "f1_weighted" for multiclass
    n_jobs=-1
)
return scores.mean()
```

*(Same change pattern for all other objectives below.)*

---

#### 3.2 `objective_hgb`

**Model**

```python
HistGradientBoostingRegressor(
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42
)
```

→

```python
HistGradientBoostingClassifier(
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42
)
```

**Scoring + return**

```python
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
)
return -scores.mean()
```

→

```python
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="f1", n_jobs=-1
)
return scores.mean()
```

---

#### 3.3 `objective_xgb`

**Model**

```python
XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    tree_method="hist",
    n_jobs=-1,
)
```

→ for binary classification (example):

```python
XGBClassifier(
    objective="binary:logistic",   # or "multi:softprob" for multiclass
    random_state=42,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    tree_method="hist",
    n_jobs=-1,
)
```

**Scoring + return**: same pattern as above.

---

#### 3.4 `objective_lgbm`

**Model**

```python
LGBMRegressor(
    random_state=42,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    n_jobs=-1,
    verbose=-1,
)
```

→

```python
LGBMClassifier(
    random_state=42,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    n_jobs=-1,
    verbose=-1,
)
```

**Scoring + return**: same pattern as above.

---

### 4. Optuna “direction” and reporting (NO PCA loop)

In the loop over `model_names`:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    study_name=f"{name}_study"
)
```

→

```python
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name=f"{name}_study"
)
```

**Print statements (metric name + formatting):**

```python
print(f"nBest {name.upper()} CV MAE: ${study.best_value:,.2f}")
```

→

```python
print(f"nBest {name.upper()} CV F1: {study.best_value:.4f}")
```

---

**Final model construction for each name**

All these lines:

```python
Ridge(alpha=best_params["ridge__alpha"])
HistGradientBoostingRegressor(...)
XGBRegressor(...)
LGBMRegressor(...)
```

must mirror the classifier changes from section 3, e.g.:

```python
RidgeClassifier(alpha=best_params["ridge__alpha"])
HistGradientBoostingClassifier(...)
XGBClassifier(...)
LGBMClassifier(...)
```

---

**Test-set evaluation + variable names**

```python
y_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"{name} (no PCA) Test MAE: ${test_mae:,.2f}")

results[name] = {"pipeline": final_model, "test_mae": test_mae}
```

→

```python
y_pred = final_model.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average="f1")  # or "macro"/"weighted"

print(f"{name} (no PCA) Test F1: {test_f1:.4f}")

results[name] = {"pipeline": final_model, "test_f1": test_f1}
```

(Choose `average` according to your problem.)

---

**MLflow logging for no-PCA models**

```python
mlflow.log_metric("test_MAE", test_mae)
mlflow.log_metric("cv_MAE", study.best_value)
```

→

```python
mlflow.log_metric("test_F1", test_f1)
mlflow.log_metric("cv_F1", study.best_value)
```

---

### 5. Objective functions for PCA models (STEP 6)

Repeat the same pattern in each `_pca` objective:

- Use classifier classes instead of regressors.  
- Change scoring to `"f1"` (or variant).  
- Return `scores.mean()` instead of `-scores.mean()`.

Examples:

#### 5.1 `objective_ridge_pca`

```python
pipeline = make_pipeline(preprocessing_clone, PCA(n_components=pca_components), Ridge(alpha=alpha))

scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
)
return -scores.mean()
```

→

```python
pipeline = make_pipeline(preprocessing_clone, PCA(n_components=pca_components), RidgeClassifier(alpha=alpha))

scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=3, scoring="f1", n_jobs=-1
)
return scores.mean()
```

Similar replacements for:

- `HistGradientBoostingRegressor` → `HistGradientBoostingClassifier`  
- `XGBRegressor` → `XGBClassifier` (with classification objective)  
- `LGBMRegressor` → `LGBMClassifier`

---

### 6. Optuna “direction” and reporting for PCA models (STEP 7)

In the PCA loop:

```python
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    study_name=f"{name}_study"
)
```

→

```python
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    study_name=f"{name}_study"
)
```

**Print and test-eval parts:**

```python
print(f"nBest {name.upper()} CV MAE: ${study.best_value:,.2f}")
...
y_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"{name} Test MAE: ${test_mae:,.2f}")

pca_results[name] = {"pipeline": final_model, "test_mae": test_mae}
```

→

```python
print(f"nBest {name.upper()} CV F1: {study.best_value:.4f}")
...
y_pred = final_model.predict(X_test)
test_f1 = f1_score(y_test, y_pred, average="f1")  # or "macro"/"weighted"
print(f"{name} Test F1: {test_f1:.4f}")

pca_results[name] = {"pipeline": final_model, "test_f1": test_f1}
```

**MLflow logging:**

```python
mlflow.log_metric("test_MAE", test_mae)
mlflow.log_metric("cv_MAE", study.best_value)
```

→

```python
mlflow.log_metric("test_F1", test_f1)
mlflow.log_metric("cv_F1", study.best_value)
```

---

### 7. PCA final model creation (classifier classes)

Same as STEP 5: in the PCA loop, every regressor in:

```python
Ridge(...)
HistGradientBoostingRegressor(...)
XGBRegressor(...)
LGBMRegressor(...)
```

→ must be the classifier equivalents:

```python
RidgeClassifier(...)
HistGradientBoostingClassifier(...)
XGBClassifier(...)
LGBMClassifier(...)
```

---

### 8. Global best model selection (STEP 8)

You currently collect:

```python
results[name] = {"pipeline": final_model, "test_mae": test_mae}
pca_results[name] = {"pipeline": final_model, "test_mae": test_mae}
...
global_best_name = min(all_results, key=lambda k: all_results[k]["test_mae"])
global_best_mae = all_results[global_best_name]["test_mae"]
...
print(f"Global best Test MAE:  ${global_best_mae:,.2f}")
```

These lines must become:

```python
results[name] = {"pipeline": final_model, "test_f1": test_f1}
pca_results[name] = {"pipeline": final_model, "test_f1": test_f1}
...
global_best_name = max(all_results, key=lambda k: all_results[k]["test_f1"])
global_best_f1 = all_results[global_best_name]["test_f1"]
...
print(f"Global best Test F1:   {global_best_f1:.4f}")
```

Also adjust all later references from `global_best_mae` → `global_best_f1`.

---

### 9. Save / load / comparison function (STEP 9)

**(a) `compare_model_predictions` metric logic**

Currently:

```python
mae_mem = mean_absolute_error(y, preds_mem)
mae_load = mean_absolute_error(y, preds_load)

print(f"nTest MAE (memory): ${mae_mem:,.2f}")
print(f"nTest MAE (loaded): ${mae_load:,.2f}")
print(f"Difference in MAE:  ${abs(mae_mem - mae_load):,.2f}")
...
return {
    "mae_memory": mae_mem,
    "mae_loaded": mae_load,
    ...
}
```

→ for F1 (example with macro averaging):

```python
f1_mem = f1_score(y, preds_mem, average="macro")
f1_load = f1_score(y, preds_load, average="macro")

print(f"nTest F1 (memory): {f1_mem:.4f}")
print(f"Test F1 (loaded): {f1_load:.4f}")
print(f"Difference in F1:  {abs(f1_mem - f1_load):,.6f}")
...
return {
    "f1_memory": f1_mem,
    "f1_loaded": f1_load,
    ...
}
```

**(b) Text/variable names where MAE is mentioned**

At the very end:

```python
print("nDone:")
print(f"- GLOBAL best model key: {global_best_name}")
print(f"- GLOBAL best Test MAE:  ${global_best_mae:,.2f}")
print(f"- Saved & loaded global best; predictions match: {comparison['are_identical']}")
```

→

```python
print("nDone:")
print(f"- GLOBAL best model key: {global_best_name}")
print(f"- GLOBAL best Test F1:   {global_best_f1:.4f}")
print(f"- Saved & loaded global best; predictions match: {comparison['are_identical']}")
```

---

### 10. Any remaining “MAE” strings / currency formatting

Search and update all *remaining* occurrences of:

- `MAE` in print strings / metric names → `F1`  
- Dollar formatting `${...:,.2f}` (used for MAE) → plain numeric formatting like `{...:.4f}`.