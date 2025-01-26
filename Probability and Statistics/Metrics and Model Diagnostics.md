In this page we will learn about model evaluation parameters and model diagnositcs.

# Confusion Matrix

A better way to evaluate the performance of a classifier is using confusion matrix.

## Cells of a confusion Matrix

A confusion matrix is a table that describes the performance of a classification model by showing the counts of true and false predictions versus actual values. Here's what each cell represents:

| True Positive (TP) | False Positive (FP) |
| --- | --- |
| False Negative (FN) | True Negative (TN) |

Where:

- **True Positives (TP):** Cases correctly predicted as positive
- **False Positives (FP):** Cases incorrectly predicted as positive (Type I error)
- **False Negatives (FN):** Cases incorrectly predicted as negative (Type II error)
- **True Negatives (TN):** Cases correctly predicted as negative

This 2x2 matrix forms the basis for calculating many important metrics like accuracy, precision, recall, and F1-score.

```python
from sklearn.metrics import confusion_matrix

conf_mx = confusion_matrix(y_train_5, y_train_pred)
conf_mx
# array([[53892,   687],
#       [ 1891,  3530]])

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
```

## Metrics to measure the validity of the model

### **Accuracy**

the proportion of correct predictions out of the total number of predictions.

$$Accuracy = \frac {\text{(number of correct predictions)}} {\text{(total number of predictions)}}$$

### **Precision**

the proportion of true positives (correctly predicted positives) out of the total number of predicted positives.

$$Precision = \frac{\text{True Positives}}{\text{True Positives + False Positives}}$$

Example → Spam filtering (**false positive should be less**). 

Proportion of how many of the predicted spams are actually spam. 

| Term (result /prediction) | Meaning |
| --- | --- |
| True Positive | Correctly predicted spam |
| False Positive | Legitimate email incorrectly flagged as spam |
| False Negative | Spam email not flagged as spam |
| True Negative | Correctly identified as not spam |

Use when cost of false positive is high.

### **Recall (sensitivity or true positive rate)**

the proportion of true positives (correctly predicted positives) out of the total number of actual positives.

$$Recall = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}$$

Example → Medical diagnosis (**false negative should be less**)

```python
from sklearn.metrics import precision_score, recall_score

print("Precision score: ", precision_score(y_train_5, y_train_pred))
print("Recall score: ", recall_score(y_train_5, y_train_pred))

# Precision score:  0.8370879772350012
# Recall score:  0.6511713705958311
```

### **F1 Score**

a weighted average of precision and recall, where a score of 1.0 represents perfect precision and recall.

The F1 score is the harmonic mean of precision and recall, defined as:

$$F1 = \frac{2}{\frac{1}{precision} + \frac{1}{recall}} = 2 * \frac{precision * recall}{precision + recall}$$

F1 score would be high only when both precision and recall are high.

The F1 score is a useful metric for evaluating the overall performance of a binary classifier when precision and recall are both important.

```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)

# 0.7325171197343846
```

**Precision/Recall tradeoff:**

If you raise the threshold (of model), the false positive (6) becomes true negative → precision increases. [predicted positives decreases]

But one true positive (5) becomes a false negative → recall decreases.

![alt text](images/precision%20recall%20trade%20off.png)

*Fig: Precision/recall trade-off (images are ranked by classifier score). Those above the chosen threshold are considered positive*

### **ROC Curve:** (Receiver operating characteristic)

a graphical representation of the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity).

- **True Positive Rate (TPR) or Sensitivity or recall**
- **False Positive Rate (FPR) or Fall-out:** the proportion of false positives (incorrectly predicted positives) out of the total number of actual negatives.
    
    $$FPR = \frac{\text{False Positives}}{\text{False Positives + True Negatives}}$$
    

These metrics are often used in evaluating binary classifiers and are also used to create ROC curves.

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, 
                                   method="predict_proba")    # SGD has method="decision_function"

y_scores_forest = y_probas_forest[:, 1]   # probability of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# plt.plot(fpr, tpr, linewidth=2, label=label)
```

![Fig: ROC space for a “better” and “worse” classifier.](https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Roc_curve.svg/330px-Roc_curve.svg.png)

*Fig: ROC space for a “better” and “worse” classifier.*

- **AUC:** the area under the ROC curve, which represents the overall performance of the classifier.
    
    ```python
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores_forest)
    # 0.9983436731328145
    ```
    

The choice of evaluation metric depends on the specific problem and the goals of the analysis. 

For example, 

    In medical diagnosis problem, recall (sensitivity) might be more important than precision, since false negatives (missed diagnoses) can have serious consequences. 

    But in spam filtering problem, precision might be more important than recall, since false positives (legitimate emails flagged as spam) are generally less problematic than false negatives (spam emails that are not filtered).

# Residuals

A residual is the difference between an observed value and its corresponding predicted value in a regression model:

$$Residual = Observed\space Value - Predicted\space Value$$

## Examining Residuals

Residuals help us evaluate how well our regression model fits the data. By analyzing residuals, we can:

- **Check Model Assumptions:** Verify if linear regression assumptions are met
- **Identify Patterns:** Detect if there are systematic errors in our predictions
- **Find Outliers:** Spot unusual observations that might influence our model

## Residual Plot Analysis

A residual plot shows residuals on the y-axis and predicted values (or independent variables) on the x-axis. Here's what to look for:

- **Random Scatter:** Ideally, residuals should be randomly scattered around zero, indicating a good model fit
- **No Pattern:** Any visible patterns suggest the model might be missing important aspects of the relationship
- **Constant Spread:** The spread of residuals should be roughly constant (homoscedasticity)

```python
# Creating a residual plot
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

Common patterns in residual plots and their interpretations:

- **Funnel Shape:** Indicates heteroscedasticity (non-constant variance)
- **Curved Pattern:** Suggests non-linear relationship between variables
- **Clustering:** May indicate that important variables are missing from the model

# Effect sizes

Effect size is a statistical measure that quantifies the strength or magnitude of a relationship between variables or the size of a difference between groups.

## Cohen's d

Cohen's d is one of the most common effect size measures. It measures the standardized difference between two means:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

where:

- n₁ and n₂ are the sample sizes of the two groups
- s₁ and s₂ are the standard deviations of the two groups
- s_pooled is the pooled standard deviation.

Cohen suggested the following guidelines for interpreting d:

- **Small effect:** d = 0.2
- **Medium effect:** d = 0.5
- **Large effect:** d = 0.8

```python
from scipy import stats
import numpy as np

# Calculate Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    return (np.mean(group1) - np.mean(group2)) / pooled_se
```

- **Example (calculate pooled standard deviation) ❓**
    
    A statistician for an educational platform wants to evaluate the effect of introducing new question types on test-takers. To do so, the statistician calculates Cohen’s d using the following date:
    
    |  | Size | Scores Mean | Standard deviation |
    | --- | --- | --- | --- |
    | Sample A (control) | 100 | 3.5 | 0.7 |
    | Sample B (treatment) | 100 | 4.5 | 0.8 |
    
    What is the result of Cohen's d in the above scenario?
    
    **Solution ✅**
    
    Let's calculate Cohen's d using the formula:
    
    $$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$
    
    First, calculate the pooled standard deviation:
    
    $s_{\text{pooled}} = \sqrt{\frac{(100 - 1)(0.7)^2 + (100 - 1)(0.8)^2}{100 + 100 - 2}}$
    
    $s_{\text{pooled}} = \sqrt{\frac{99(0.49) + 99(0.64)}{198}} = \sqrt{\frac{48.51 + 63.36}{198}} = \sqrt{0.566} = 0.752$
    
    Now we can calculate Cohen's d:
    
    $d = \frac{4.5 - 3.5}{0.752} = \frac{1.0}{0.752} = 1.33$
    
    The Cohen's d value of 1.33 indicates a large effect size (> 0.8), suggesting that the new question types had a substantial impact on test-taker performance.
    

## R-squared (R²)

R-squared is a statistical measure of how close the data are to the fitted regression line. It is the percentage of the response variable variation that is explained by a linear model.

The formula for R² is:

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Where:

- $y_i$ are the actual values
- $ŷ_i$ are the predicted values
- $ȳ$ is the mean of the actual values

```python
from sklearn.metrics import r2_score

# Calculate R-squared
r2 = r2_score(y_true, y_pred)
print(f"R-squared: {r2:.4f}")
```

When the data points are close to the regression model, the R-Squared value tends to be higher, indicating a good fit for the linear model.

![R-Squared Graph](https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2017/04/flplinear.gif?resize=515%2C343)

*Fig: R-Squared graph*

Interpretation of R²:

- **R² = 1:** Perfect fit, model explains all variability
- **R² = 0:** Model explains none of the variability
- **R² < 0:** Model performs worse than horizontal line

Note that while R² is useful, it should not be used as the sole metric for model evaluation, as it can be misleading in certain situations.