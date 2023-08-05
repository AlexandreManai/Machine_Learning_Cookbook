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

```python
# Z-score method - Doesn't handle NaN values
from scipy.stats import zscore
z_scores = zscore(data['column_name'])  # Change 'column_name' with your column
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3)
new_data = data[filtered_entries]
```

## 10. Feature Engineering ( as required)

**Remark**: Remember, the exact EDA process can differ based on the specifics of your dataset and what you're interested in investigating. This is just a generic cheat sheet to get started. Be sure to replace 'column_name', 'column1', and 'column2' with your actual column names.