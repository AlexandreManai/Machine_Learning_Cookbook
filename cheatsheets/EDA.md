# EDA Cheat Sheet

## 1. Data Overview

```python
# Get the number of rows and columns
print('Shape:', data.shape)

# Display the first few rows of the dataset
print('First few entries:')
print(data.head())
```

## 2. Descriptive Statistics

```python
# Get descriptive statistics
print('Descriptive Statistics:')
print(data.describe())
```

## 3. Missing Values

```python
# Check for missing values
print('Missing Values:')
print(data.isnull().sum())
```

## 4. Data Types

```python
# Check data types of each column
print('Data Types:')
print(data.dtypes)
```

## 5. Unique Values

```python
# Get unique values for each column (useful for categorical data)
print('Unique values:')
for col in data.columns:
    print(col, ':', len(data[col].unique()))
```

## 6. Correlation Matrix

```python
# Generate a correlation matrix
print('Correlation Matrix:')
print(data.corr())
```

## 7. Visualizing Data Distribution

```python
# Histograms or Box-plots for continuous variables
data['column_name'].hist(bins=30)  # Change 'column_name' with your column

# Bar charts for categorical variables
data['column_name'].value_counts().plot(kind='bar')  # Change 'column_name' with your column
```

## 8. Visualizing Relationships in Data

```python
# Scatter plots for relationship between continuous variables
data.plot(kind='scatter', x='column1', y='column2')  # Change 'column1' and 'column2' with your columns

# Box plots for relationship between categorical and continuous variables
data.boxplot(column='continuous_column', by='categorical_column')  # Change with your columns
```

## 9. Outlier Detection

### Z-score

```python
# Z-score method - Doesn't handle NaN values
from scipy.stats import zscore
z_scores = zscore(data['column_name'])  # Change 'column_name' with your column
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
new_data = data[filtered_entries]
```

### IQR

```python
outliers_percentage = {}
for col in data.columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Count outliers
    outliers_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
    
    # Calculate percentage
    outliers_percentage[col] = (outliers_count / len(data[col])) * 100
```

## 10. Skewness

```python
data.skew() # only for numeric data
```

## 12. Multicollinearity

**Definition:** Multicollinearity arises when two or more independent variables in the regression model are highly correlated. This can make it difficult to determine the individual effect of predictors on the response variable.

### Detecting with VIF (Variance Inflation Factor):

- VIF < 5: Generally okay
- 5 =< VIF =< 10: Moderate
- VIF > 10: High

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

data_ = data.copy()
data_ = data_.drop(['target_feature'], axis=1)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["Feature"] = data_.columns
vif_data["VIF"] = [variance_inflation_factor(data_.values, i) for i in range(data_.shape[1])]
```

### Handling Multicollinearity

1. Remove highly correlated features.
2. Use regularization (e.g., Ridge or Lasso regression).
3. Apply dimensionality reduction techniques (like PCA).
4. Opt for models that handle multicollinearity well (e.g., Decision Trees or Random Forests).

Note: Ensure that the target variable is not included in VIF computations.

## 12. Feature Engineering ( as required)

**Remark**: Remember, the exact EDA process can differ based on the specifics of your dataset and what you're interested in investigating. This is just a generic cheat sheet to get started. Be sure to replace 'column_name', 'column1', and 'column2' with your actual column names.