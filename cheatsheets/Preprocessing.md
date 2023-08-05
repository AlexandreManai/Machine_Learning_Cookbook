# Pre-Processing Cheat Sheet

## 1. Handling Missing Values

### Continuous Variables

```python
# Mean imputation - Sensible to skewed data
data['column_name'].fillna(data['column_name'].mean(), inplace=True)

# Median imputation
data['column_name'].fillna(data['column_name'].median(), inplace=True)
```

### Categorical Variables

```python
# Mode imputation
data['column_name'].fillna(data['column_name'].mode()[0], inplace=True)
```

### Model-Based Imputation

Model-based imputation leverages predictive models to estimate and fill missing values based on other observed features. It is particularly useful when missingness is not completely at random.

#### K-NN Imputation
K-NN (K-Nearest Neighbors) Imputation uses the K nearest observations to impute missing values. It is distance-based and works well for numerical or categorical data.
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

#### Regression Imputation

Regression Imputation uses linear regression to predict and fill missing values. Suitable for numerical features with linear relationships.

```python
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# For each feature with missing values, fit a linear model
for feature in features_with_missing:
    # Split data into those with and without missing values for the feature
    train_data = data[data[feature].notnull()]
    test_data = data[data[feature].isnull()]

    # Separate feature to predict and other features
    y_train = train_data[feature]
    X_train = train_data.drop(columns=[feature])
    X_test = test_data.drop(columns=[feature])

    # Fit and predict
    model = LinearRegression()
    model.fit(X_train, y_train)
    imputed_values = model.predict(X_test)

    # Fill missing values
    data.loc[data[feature].isnull(), feature] = imputed_values
```

##### Considerations:
- **Complexity**: Model-based imputation adds complexity and may increase the risk of overfitting.
- **Assumptions**: It requires understanding the relationships between features.
- **Validation**: Cross-validation or a holdout set should be used to validate the imputation method.


## 2. Encoding Categorical Variables

### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['column_name'] = le.fit_transform(data['column_name'])
```

### One-Hot Encoding
```python
data = pd.get_dummies(data, columns=['column_name'])
```

## 3. Feature Scaling

### Min-Max Scaling
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data['column_name'] = scaler.fit_transform(data[['column_name']])
```

### Standardization
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data['column_name'] = scaler.fit_transform(data[['column_name']])
```

## 4. Handling Imbalanced Data

### Oversampling
```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)
```

### Undersampling
```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
```

### SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
```

## 5. Text Data Pre-processing

### Bag of words
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
```

### TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
```

## 6. Image Data Pre-processing

### Image Normalization
```python
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load and preprocess an image
image = Image.open('image_path')
image = img_to_array(image)
image = image.astype('float32')
image /= 255.0
```

## 7. Time Series Data Pre-processing

### Differencing
```python
data['column_name_diff'] = data['column_name'].diff()
```

### Seasonal Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data['column_name'], model='additive', freq=seasonal_period)
```

## 8. Handling Outliers
```python
data.clip(0,1) # 0 = lower boundary, 1 = higher boundary
```

## 9. Feature Importance
The importance of a feature can be estimated by looking at coefficient of a feature after training.

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
importance = model.coef_
```

### Decisions Trees / Random Forests
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor().fit(X, y)
importance = model.feature_importances_
```

And others ...

**Note:** Feature importance can be highly dependent on the specific model and its parameters, so it's often a good idea to look at feature importance across multiple models and/or with different model parameters.

## 10. Preprocessing Effectiveness Tests

### Model Imputation methods Test
Tests need to be done to avoid possible overfitting

```python
from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Initialize K-Fold cross-validation
kf = KFold(n_splits=5)

# Define a list to store each fold's error
errors = []

# Iterate through each fold
for train_index, test_index in kf.split(X):
    # Split data into training and validation set
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply imputation on the training set
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    # Apply the same imputation on the validation set
    X_test_imputed = imputer.transform(X_test)

    # Train a model on the imputed training set
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)

    # Predict on the imputed validation set
    y_pred = model.predict(X_test_imputed)

    # Compute an error metric
    error = mean_squared_error(y_test, y_pred)
    errors.append(error)

# Compute the mean error across all K folds
mean_error = np.mean(errors)
```

## Overfitting scenarios:
1. **Fitting the Imputation Model on the Entire Dataset:** If you fit the imputation model (e.g., a regression model used to predict missing values) on the entire dataset, including the validation or test set, you are leaking information from the validation/test set into the training process. This means the imputed values may be overly optimistic and tailored to the particular dataset, resulting in a model that doesn't generalize well to new, unseen data.

2. **Complex Imputation Models:** If the imputation model itself is very complex (e.g., a deep neural network), it may capture noise in the data rather than the underlying pattern. This leads to imputed values that fit the training data very well but may not be realistic or applicable to unseen data.

3. **Highly Correlated Features:** If you're using features that are highly correlated with the target variable to perform imputation, and these features also contain missing values that are being predicted, there could be a recursive biasing effect. The imputation model may create imputed values that are too closely aligned with the target variable, leading the main predictive model to "learn" these artificial patterns that don't hold in unseen data.

4. **Imbalanced Missingness:** If the missingness is not random (i.e., values are missing systematically in some way related to other observed data), using a model to impute these values might introduce biases. If the imputation model captures these biases and the main model learns from them, it may perform poorly on a dataset where the missingness pattern is different.
