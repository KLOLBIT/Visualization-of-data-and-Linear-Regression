#  USA Housing Price Prediction – Linear Regression

This project analyzes the **USA Housing dataset** and builds a **Linear Regression model** to predict **house prices** based on demographic and housing features.

---

##  Project Structure

```
.
├── USA_Housing.csv          # Dataset
├── housing_regression.py    # Main script
└── README.md                # Project documentation
```

---

##  Requirements

Install the dependencies with:

```
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

##  Dataset Overview

The dataset (`USA_Housing.csv`) contains **5000 entries** and the following columns:

| Column Name                    | Description                             |
| ------------------------------ | --------------------------------------- |
| `Avg. Area Income`             | Average income of residents in the area |
| `Avg. Area House Age`          | Average age of houses in the area       |
| `Avg. Area Number of Rooms`    | Average number of rooms per house       |
| `Avg. Area Number of Bedrooms` | Average number of bedrooms per house    |
| `Area Population`              | Population of the area                  |
| `Price`                        | House price (Target variable)           |
| `Address`                      | Address of the house                    |

---

##  Exploratory Data Analysis (EDA)

1. **Pairplot** to visualize relationships:

   ```python
   sns.pairplot(df)
   ```

2. **Distribution of Price**:

   ```python
   sns.histplot(df['Price'], kde=True)
   ```

3. **Correlation Heatmap**:

   ```python
   sns.heatmap(df.corr(), annot=True)
   ```

Key Findings:

* House **Price** correlates strongly with **Avg. Area Income** (`0.64`) and **Avg. Area House Age** (`0.45`).

---

##  Linear Regression Model

### Feature & Target Split

```python
X = df[['Avg. Area Income', 'Avg. Area House Age',
        'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
        'Area Population']]
y = df['Price']
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.4,
                                                    random_state=101)
```

### Model Training

```python
from sklearn.linear_model import LinearRegression

lrm = LinearRegression()
lrm.fit(X_train, y_train)
```

### Model Coefficients

| Feature                      | Coefficient |
| ---------------------------- | ----------- |
| Avg. Area Income             | **21.53**   |
| Avg. Area House Age          | **164,883** |
| Avg. Area Number of Rooms    | **122,368** |
| Avg. Area Number of Bedrooms | **2,234**   |
| Area Population              | **15.15**   |

Interpretation:

* For every additional **\$1** in average area income, house price increases by \~**\$21.5**, keeping other factors constant.

---

##  Model Evaluation

### Predictions

```python
predictions = lrm.predict(X_test)
```

### Visualization

* **Actual vs Predicted Prices**:

  ```python
  plt.scatter(y_test, predictions)
  ```
* **Residual Distribution**:

  ```python
  sns.histplot((y_test - predictions), kde=True)
  ```

### Error Metrics

| Metric                  | Value          |
| ----------------------- | -------------- |
| Mean Absolute Error     | 82,288         |
| Mean Squared Error      | 10,460,958,907 |
| Root Mean Squared Error | 102,278        |

The model achieves an **RMSE ≈ \$102,000**, which is reasonable given average prices in the dataset (\~\$1.2M).

---

##  How to Run

1. Place `USA_Housing.csv` in the working directory.
2. Run the script:

   ```
   python housing_regression.py
   ```
3. The script outputs model coefficients, intercept, predictions, and evaluation metrics.

---

##  Note on Error

At the end of your script, you attempted:

```python
y = df['Yearly Amount Spent']
```

 This column does **not exist** in `USA_Housing.csv` (it belongs to another dataset, often used in E-Commerce projects). Remove or update this part to avoid a **KeyError**.

---

## Future Work

* Try **Polynomial Regression** or **Regularization (Ridge/Lasso)** for performance improvement.
* Perform **feature selection** (drop `Avg. Area Number of Bedrooms`, which has weak correlation).
* Build a **web app (Flask/Streamlit)** to input housing features and get price predictions.

---

Do you want me to also **add code for saving the trained model with `joblib` or `pickle`** so it can be reused without retraining?
