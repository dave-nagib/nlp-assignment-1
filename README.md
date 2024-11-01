# NLP Sentiment Classification Assignment

## Overview
This project involves implementing two text classification models — Naive Bayes and Logistic Regression — to classify movie reviews into five sentiment classes using the Stanford Sentiment Treebank (SST) dataset. The assignment also includes implementing evaluation metrics, including confusion matrix, precision, recall, and F1 score.

## Dataset
The dataset used for this assignment is the **Stanford Sentiment Treebank (SST)**. It contains movie reviews labeled with sentiment scores ranging from 0 to 1. The scores are mapped to five distinct classes as follows:

- **Class 0**: Very Negative (0.0 - 0.2)
- **Class 1**: Negative (0.2 - 0.4)
- **Class 2**: Neutral (0.4 - 0.6)
- **Class 3**: Positive (0.6 - 0.8)
- **Class 4**: Very Positive (0.8 - 1.0)

## Project Structure
This assignment has three main parts:

### Part 1: Naive Bayes
- Implemented from scratch using only NumPy.
- A pipeline is created with `CountVectorizer` to convert text into feature vectors.
- Comparison with scikit-learn’s `MultinomialNB` to validate the implementation.

### Part 2: Logistic Regression
- Built a logistic regression model from scratch using gradient descent with L2 regularization.
- Feature representation is done using bi-grams for each sentence.
- Comparison with scikit-learn’s `LogisticRegression` and `SGDClassifier` (configured for logistic regression) for validation.

### Part 3: Confusion Matrix & Evaluation Metrics
- Implemented functions to compute the confusion matrix, precision, recall, and F1 score.
- Comparison with scikit-learn’s metrics to ensure accuracy.

## Requirements
- Python 3.x
- NumPy
- scikit-learn

Install the requirements via pip:
```bash
pip install numpy scikit-learn
```

## Usage
To run the code:
1. **Preprocess the dataset**: Load and preprocess SST data, including mapping sentiment scores to classes.
2. **Run the models**:
   - Naive Bayes model: Implemented from scratch and validated with `MultinomialNB`.
   - Logistic Regression model: Implemented from scratch and validated with scikit-learn’s logistic regression.
3. **Evaluate the models**: Run evaluation metrics for each model and compare custom and scikit-learn results.

### Example Code
```python
# Train Custom Logistic Regression Model
model = LogisticRegressionCustom(num_features=x_train.shape[1], alpha=0.01)
history = model.train(x_train, y_train, learning_rate=0.01, epochs=100, batch_size=32)

# Evaluate Model
model.evaluate(x_test, y_test)

# Compare with scikit-learn models
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score

# Configure SGDClassifier for Logistic Regression
sgd_classifier = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, tol=1e-3)
sgd_classifier.fit(x_train, y_train)
y_pred_sgd = sgd_classifier.predict(x_test)
print("SGD Classifier Accuracy:", accuracy_score(y_test, y_pred_sgd))
```

## Results
- The custom models were validated against scikit-learn’s implementations to ensure accuracy.
- The following evaluation metrics were used:
  - **Confusion Matrix**
  - **Precision, Recall, and F1 Score** (Macro-averaged)

## Notes
- **Memory Management**: Due to limited resources (especially on Colab), optimizations in data types and manual memory deallocation were employed.
- **Performance**: Regularization (L2) was used in Logistic Regression to address overfitting.

## License
This project is licensed under the MIT License.
