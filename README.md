# Linear Regression: Bike Sharing Demand

Coursework that models daily bike rentals (`cnt`) with linear regression, then compares it against regularised and tree-based models.

## What the notebook does

1. **Preparation**: parses dates, maps coded season and weather values to labels, and one-hot encodes the categorical columns
2. **Exploration**: plots a correlation heatmap and box plots of demand by season, weather, month and weekday
3. **Baseline model**: fits a linear regression on scaled features and reports R² and RMSE
4. **Feature selection**: uses Recursive Feature Elimination (RFE) to pick the top 10 features and refits the model
5. **Model comparison**: tries polynomial interaction features, Ridge, Lasso and Random Forest
6. **Tuning**: uses GridSearchCV for the Ridge/Lasso `alpha` and RandomizedSearchCV for Random Forest, then compares the best models

## Data

Uses `day.csv` from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). It isn't included in this repo. Download it, then update the path in the first `pd.read_csv(...)` cell.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook "Linear regression Assignment.ipynb"
```
