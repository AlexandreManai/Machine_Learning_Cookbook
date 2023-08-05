# Metrics Cheat Sheet

## Regression Metrics

### Mean Absolute Error ( MAE )
It measures the average magnitude of the errors in a set of predictions, without considering their direction.

```python 
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_true, y_pred)
```

- Strength: Gives an idea of magnitude of error without considering direction, easy to interpret.
- Weakness: All errors have the same weight regardless of their magnitude. It's not sensitive to outliers as compared to Mean Squared Error.

### Mean Squared Error ( MSE )
It measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
```

- Strength: More useful than MAE when large errors are particularly undesirable.
- Weakness: The squaring makes larger errors more noticeable and smaller ones less noticeable. It is not as interpretable as MAE.

### Root Mean Squared Error ( RMSE )
This is the square root of the mean square error.

```python
import numpy as np
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

- Strength: RMSE is even more sensitive to outliers than MSE. This can be useful in many cases where you want to pay more attention to the outliers.
- Weakness: Just like MSE, larger errors are more prominent and smaller ones are less prominent.

### R² Score (Coefficient of Determination)
This metric provides an indication of the goodness of fit of a set of predictions to the actual values. In the case of linear regression, this metric equals the square of the correlation between the predicted and actual values.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

- Strength: It tells you how much of the variance in the data your model is able to explain.
- Weakness: R² will always increase as you add more features to the model, even if they are unrelated to the output.

## Classification Metrics

### Accuracy Score
This is the ratio of number of correct predictions to the total number of input samples.

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
```

- Strength: Very intuitive, easy to understand.
- Weakness: It doesn't perform well when classes are imbalanced.

### Precision Score
It is the ratio of correctly predicted positive observations to the total predicted positives.

```python
from sklearn.metrics import precision_score
precision = precision_score(y_true, y_pred)
```

- Strength: It tells you how many of the samples predicted as positive are actually positive.
- Weakness: Precision alone doesn’t give a complete picture of the model performance.

### Recall Score
It is the ratio of correctly predicted positive observations to all the actual positives.

```python
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred)
```

- Strength: It tells you how many of the positive samples are captured by the positive predictions.
- Weakness: Just like precision, using recall alone doesn’t give a complete picture of the model performance.

### F1 Score
This is the weighted average of Precision and Recall. This score tries to balance both recall and precision.

```python
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred)
```

- Strength: F1 score is the harmonic mean of precision and recall and tries to balance both.
- Weakness: F1 score might not be a good measure when you care about only precision or only recall.

### Confusion Matrix
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
```

- Strength: Gives a detailed view of classification results.
- Weakness: Not suitable for direct comparisons of multiple models.

### Area Under the Receiver Operating Characteristic Curve (ROC AUC Score)
This metric is equal to the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one.

```python
from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_true, y_score)
```

- Strength: Performance of the classification model at all classification thresholds. AUC-ROC score is 1 for a perfect classifier and .5 for a random one.
- Weakness: AUC-ROC score can be too optimistic for imbalanced datasets.
