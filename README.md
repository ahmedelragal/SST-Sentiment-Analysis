# Overview
This model leverages logistic regression to perform sentiment analysis on movie reviews, trained using the Stanford Sentiment Treebank (SST) dataset. This dataset contains movie reviews annotated with sentiment scores ranging from 0 to 1.

# Dataset
The SST dataset consists of movie reviews with sentiment scores, where scores closer to 0 indicate negative sentiment and those closer to 1 indicate positive sentiment. For simplification, sentiment scores are mapped into five categories:

0 to 0.2 (inclusive): Very Negative

0.2 to 0.4 (inclusive): Negative

0.4 to 0.6 (inclusive): Neutral

0.6 to 0.8 (inclusive): Positive

0.8 to 1.0 (inclusive): Very Positive
# Model Architecture
Features: Word bi-grams are utilized to represent each sentence in the movie reviews.

Model: Logistic regression is used as the classification model.

Implementation: The model is implemented using NumPy for numerical computations.
# Components
Classification Function (predict): Uses the sigmoid function to compute estimated class probabilities.

Loss Function (compute_loss): Employs binary cross-entropy loss to measure the difference between predicted and actual labels.

Optimization (sgd): Implements stochastic gradient descent (SGD) to update the model weights based on computed gradients.
# Training Process

Training Dataset: The model is trained on the SST training dataset.

Training Algorithm: Stochastic gradient descent (SGD) is used with a learning rate of 0.01 over 100 epochs.

Feature Extraction: Word bi-grams are extracted from text data to create the feature matrix.

Testing and Evaluation

Testing Dataset: The model is evaluated on the SST test dataset.

Performance Metric: Accuracy is used to evaluate the model's classification performance.

Accuracy: The final accuracy achieved on the test set is 0.75.
# Conclusion
Achieved exactly Identical results to scickit-learn's MultiNomialNB.

I used simple bigram features for training and testing As an example, the sentence “I love this movie very much” has 5 word bi-gram features namely (‘I’, ‘love’), (‘love’, ‘this’) and so on. Each sentence is represented with a vector of length equal to the number of 
unique word bi-grams in the whole dataset with 1 at the corresponding index if the bi-gram exists and 0 otherwise. I used Stochastic Gradient descent for optimization and no regularization .
in the logistic_regression_from_scratch_and_SGDcalsifier notebook I compare the results of my implementaion with SGDcLassifier. 


The results are nearly identical with only slight diffrences likely due to numerical precision reasons.
