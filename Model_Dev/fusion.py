
import numpy as pd 


from sklearn.metrics import accuracy_score
import numpy as np

class ADBoost:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.classifier_weights = np.ones(len(classifiers))
        self.errors = np.ones(len(classifiers))

    def fit(self, X, y):
        num_samples = len(X)
        sample_weights = np.ones(num_samples) / num_samples

        for i, clf in enumerate(self.classifiers):
            if i == 0:
                # deep learning model
                clf.fit(X, y, sample_weight=sample_weights)
                predictions = clf.predict(X)
                incorrect = (predictions != y)
            elif i == 1:
                # image+poly_features
                clf.fit(X, y, sample_weight=sample_weights)
                predictions = clf.predict(X)
                incorrect = (predictions != y)
            else:# elm models
                clf.fit(X, y, sample_weight=sample_weights)
                predictions = clf.predict(X)
                incorrect = (predictions != y)
            # Error
            error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
            self.errors[i] = error
            # Boost weight
            boost = np.log((1 - error) / error) + np.log(len(self.classifiers) - 1)
            self.classifier_weights[i] = boost
            # Update sample weights
            sample_weights *= np.exp(boost * incorrect * ((sample_weights > 0) | (boost < 0)))
        self.classifier_weights /= np.sum(self.classifier_weights)

    def predict(self, X):
        classifier_preds = np.array([clf.predict(X) for clf in self.classifiers])
        weighted_preds = np.average(classifier_preds, axis=0, weights=self.classifier_weights)
        return np.sign(weighted_preds)





































