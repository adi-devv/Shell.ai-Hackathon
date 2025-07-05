We are trying to predict 10 blend properties (multi-output regression) from blend composition and component properties using machine learning models.

### 1. Overall Approach

My goal was to predict 10 blend properties, a multi-output regression problem. I used **Random Forest Regressor** wrapped in `sklearn.multioutput.MultiOutputRegressor` for its non-linear handling and tabular data performance, training one model per target.

Workflow: Load data -> Separate features/targets -> Train `MultiOutputRegressor` -> Predict on test data -> Generate `submission.csv`.


### 2. Feature Engineering Details

My initial approach used all raw input columns directly: 5 blend composition columns and 50 component property columns (`ComponentX_PropertyY`).

**Reasoning:** Random Forests can inherently learn complex non-linear interactions from raw features, capturing synergistic effects without explicit manual creation.

**Future Considerations (not implemented):** More advanced feature engineering could improve performance. This includes: Interaction Features (e.g., `Component1_vol * Component1_Property1`), Ratios/Differences between properties, Polynomial Features, Statistical Aggregations (mean, min, max, std dev) of component properties, and Domain-Specific Features if knowledge were available.


### 3. Tools Used

Solution developed in Python 3.x, using:
* **Pandas:** Data loading and manipulation.
* **NumPy:** Numerical operations.
* **Scikit-learn (`sklearn`):** `RandomForestRegressor`, `MultiOutputRegressor`, `train_test_split`, `mean_absolute_percentage_error`.


### 4. Source Files

My submission includes:
* `main.py`: The Python script that performs the solution.