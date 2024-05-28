import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from scipy.linalg import pinv, inv
import time
from datetime import datetime
from collections import Counter
import numpy as np
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, precision_score, recall_score
import numpy as np

# Assuming true_labels and predicted_labels are lists containing true class labels and predicted class labels respectively
# Replace them with your actual true labels and predicted labels




from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score


    

def calculate_metrics(true_labels, predicted_labels, positive_class):

    # Convert labels to binary format for the specific comparison
    true_labels_binary = [1 if label == positive_class else 0 for label in true_labels]
    predicted_labels_binary = [1 if label == positive_class else 0 for label in predicted_labels]
    # Calculate precision
    precision = precision_score(true_labels_binary, predicted_labels_binary)
    
    # Calculate recall
    recall = recall_score(true_labels_binary, predicted_labels_binary)
    
    # Calculate F1 score
    f1 = f1_score(true_labels_binary, predicted_labels_binary)
    
    # Calculate MCC
    mcc = matthews_corrcoef(true_labels_binary, predicted_labels_binary)
        

        
    return precision, recall, f1, mcc

def get_metrics_with_prob(true_labels,predicted_labels,probs):
    # probs = [...]  # Your predicted probabilities for each class
    # Calculate overall accuracy, F1 score, MCC, and AUC
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # 'weighted' for multiclass
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    # Assuming you have probabilities for each class from the classifier, replace 'probs' with your actual probabilities
    # Assuming classes are represented as 0, 1, 2
    
    one_hot_true_labels = np.eye(3)[true_labels]  # Convert true labels to one-hot encoding
    auc = roc_auc_score(one_hot_true_labels, probs, multi_class='ovo')  # One-vs-One AUC

    # Calculate accuracy between classes A, B, C
    def class_accuracy(true_labels, predicted_labels, class_label):
        indices = np.where(np.array(true_labels) == class_label)[0]
        class_true_labels = np.array(true_labels)[indices]
        class_predicted_labels = np.array(predicted_labels)[indices]
        return accuracy_score(class_true_labels, class_predicted_labels)

    accuracy_A_B = class_accuracy(true_labels, predicted_labels, class_label=0)
    accuracy_A_C = class_accuracy(true_labels, predicted_labels, class_label=1)
    accuracy_B_C = class_accuracy(true_labels, predicted_labels, class_label=2)

    # Calculate precision, recall, and F1 score for each class
    precision_A = precision_score(true_labels, predicted_labels, labels=[0], average='micro')
    recall_A = recall_score(true_labels, predicted_labels, labels=[0], average='micro')
    f1_A = f1_score(true_labels, predicted_labels, labels=[0], average='micro')
    precision_B = precision_score(true_labels, predicted_labels, labels=[1], average='micro')
    recall_B = recall_score(true_labels, predicted_labels, labels=[1], average='micro')
    f1_B = f1_score(true_labels, predicted_labels, labels=[1], average='micro')
    precision_C = precision_score(true_labels, predicted_labels, labels=[2], average='micro')
    recall_C = recall_score(true_labels, predicted_labels, labels=[2], average='micro')
    f1_C = f1_score(true_labels, predicted_labels, labels=[2], average='micro')
    
    # Calculate One-vs-One AUC
    one_hot_true_labels = np.array(one_hot_true_labels)
    probs = np.array(probs)
    auc_A_B = roc_auc_score(one_hot_true_labels[:, 0], probs[:, 0])  # A vs B
    auc_A_C = roc_auc_score(one_hot_true_labels[:, 1], probs[:, 1])  # A vs C
    auc_B_C = roc_auc_score(one_hot_true_labels[:, 2], probs[:, 2])  # B vs C

    # Print the evaluation metrics
    print(f"Overall Accuracy: {accuracy}")
    print(f"Overall F1 Score: {f1}")
    print(f"MCC: {mcc}")
    print(f"AUC: {auc}")
    print(f"Accuracy between classes CN and MCI: {accuracy_A_B}")
    print(f"Accuracy between classes CN and AD: {accuracy_A_C}")
    print(f"Accuracy between classes MCI and AD: {accuracy_B_C}")
    print(f"Precision for class CN: {precision_A}")
    print(f"Recall for class CN: {recall_A}")
    print(f"F1 Score for class CN: {f1_A}")
    print(f"Precision for class MCI: {precision_B}")
    print(f"Recall for class MCI: {recall_B}")
    print(f"F1 Score for class MCI: {f1_B}")
    print(f"Precision for class AD: {precision_C}")
    print(f"Recall for class AD: {recall_C}")
    print(f"F1 Score for class AD: {f1_C}")
    print(f"AUC between classes CN and MCI: {auc_A_B}")
    print(f"AUC between classes CN and AD: {auc_A_C}")
    print(f"AUC between classes MCI and AD: {auc_B_C}")
    
    
    # Calculate metrics for CN vs MCI
    precision_cn_mci, recall_cn_mci, f1_cn_mci, mcc_cn_mci = calculate_metrics(true_labels, predicted_labels, 0)
    print("Metrics for CN vs MCI:")
    print("Precision:", precision_cn_mci)
    print("Recall:", recall_cn_mci)
    print("F1 Score:", f1_cn_mci)
    print("MCC:", mcc_cn_mci)


    # Calculate metrics for CN vs AD
    precision_cn_ad, recall_cn_ad, f1_cn_ad, mcc_cn_ad = calculate_metrics(true_labels, predicted_labels, 0)
    print("\nMetrics for CN vs AD:")
    print("Precision:", precision_cn_ad)
    print("Recall:", recall_cn_ad)
    print("F1 Score:", f1_cn_ad)
    print("MCC:", mcc_cn_ad)


    # Calculate metrics for AD vs MCI
    precision_ad_mci, recall_ad_mci, f1_ad_mci, mcc_ad_mci = calculate_metrics(true_labels, predicted_labels, 2)
    print("\nMetrics for AD vs MCI:")
    print("Precision:", precision_ad_mci)
    print("Recall:", recall_ad_mci)
    print("F1 Score:", f1_ad_mci)
    print("MCC:", mcc_ad_mci)
    
    return accuracy,f1,mcc,auc,accuracy_A_B,accuracy_A_C,accuracy_B_C,auc_A_B,auc_A_C,auc_B_C,precision_A,recall_A,f1_A,precision_B,recall_B,f1_B,precision_C,recall_C,f1_C


def get_metrics_without_prob(true_labels,predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # 'weighted' for multiclass
    mcc = matthews_corrcoef(true_labels, predicted_labels)

    # Calculate accuracy between classes A, B, C
    def class_accuracy(true_labels, predicted_labels, class_label):
        indices = np.where(np.array(true_labels) == class_label)[0]
        class_true_labels = np.array(true_labels)[indices]
        class_predicted_labels = np.array(predicted_labels)[indices]
        return accuracy_score(class_true_labels, class_predicted_labels)

    accuracy_A_B = class_accuracy(true_labels, predicted_labels, class_label=0)
    accuracy_A_C = class_accuracy(true_labels, predicted_labels, class_label=1)
    accuracy_B_C = class_accuracy(true_labels, predicted_labels, class_label=2)

    # Calculate precision, recall, and F1 score for each class
    precision_A = precision_score(true_labels, predicted_labels, labels=[0], average='micro')
    recall_A = recall_score(true_labels, predicted_labels, labels=[0], average='micro')
    f1_A = f1_score(true_labels, predicted_labels, labels=[0], average='micro')
    precision_B = precision_score(true_labels, predicted_labels, labels=[1], average='micro')
    recall_B = recall_score(true_labels, predicted_labels, labels=[1], average='micro')
    f1_B = f1_score(true_labels, predicted_labels, labels=[1], average='micro')
    precision_C = precision_score(true_labels, predicted_labels, labels=[2], average='micro')
    recall_C = recall_score(true_labels, predicted_labels, labels=[2], average='micro')
    f1_C = f1_score(true_labels, predicted_labels, labels=[2], average='micro')
    

    # Print the evaluation metrics
    print(f"Overall Accuracy: {accuracy}")
    print(f"Overall F1 Score: {f1}")
    print(f"MCC: {mcc}")
    print(f"Accuracy between classes CN and MCI: {accuracy_A_B}")
    print(f"Accuracy between classes CN and AD: {accuracy_A_C}")
    print(f"Accuracy between classes MCI and AD: {accuracy_B_C}")
    print(f"Precision for class CN: {precision_A}")
    print(f"Recall for class CN: {recall_A}")
    print(f"F1 Score for class CN: {f1_A}")
    print(f"Precision for class MCI: {precision_B}")
    print(f"Recall for class MCI: {recall_B}")
    print(f"F1 Score for class MCI: {f1_B}")
    print(f"Precision for class AD: {precision_C}")
    print(f"Recall for class AD: {recall_C}")
    print(f"F1 Score for class AD: {f1_C}")
        # Calculate metrics for CN vs MCI
    precision_cn_mci, recall_cn_mci, f1_cn_mci, mcc_cn_mci = calculate_metrics(true_labels, predicted_labels, 0)
    print("Metrics for CN vs MCI:")
    print("Precision:", precision_cn_mci)
    print("Recall:", recall_cn_mci)
    print("F1 Score:", f1_cn_mci)
    print("MCC:", mcc_cn_mci)


    # Calculate metrics for CN vs AD
    precision_cn_ad, recall_cn_ad, f1_cn_ad, mcc_cn_ad = calculate_metrics(true_labels, predicted_labels, 0)
    print("\nMetrics for CN vs AD:")
    print("Precision:", precision_cn_ad)
    print("Recall:", recall_cn_ad)
    print("F1 Score:", f1_cn_ad)
    print("MCC:", mcc_cn_ad)


    # Calculate metrics for AD vs MCI
    precision_ad_mci, recall_ad_mci, f1_ad_mci, mcc_ad_mci = calculate_metrics(true_labels, predicted_labels, 2)
    print("\nMetrics for AD vs MCI:")
    print("Precision:", precision_ad_mci)
    print("Recall:", recall_ad_mci)
    print("F1 Score:", f1_ad_mci)
    print("MCC:", mcc_ad_mci)
    
    return accuracy,f1,mcc,accuracy_A_B,accuracy_A_C,accuracy_B_C,precision_A,recall_A,f1_A,precision_B,recall_B,f1_B,precision_C,recall_C,f1_C


def _get_label_dic():
    from datetime import datetime
    import pandas as pd
    label_metadata = pd.read_csv('./datasets/label_meta.csv')
    def change_time(string_time):
        # 将字符串解析为日期对象
        date_object = datetime.strptime(string_time, "%m/%d/%Y")
        # 将日期对象格式化为所需的字符串格式
        formatted_time = date_object.strftime("%Y-%m-%d")

        # print(formatted_time)
        return formatted_time

    def change_label(string_label):
        if string_label == 'MCI':
            return 1
        elif string_label == 'CN':
            return 0
        elif string_label == 'AD':
            return 2
        else:
            return -1
    label_metadata['date'] =  label_metadata['Acq Date'].apply(change_time)
    label_metadata['label'] =  label_metadata['Group'].apply(change_label)
    label_dict = {f"{a}_{b}": c for a, b, c in zip(label_metadata['Subject'], label_metadata['date'], label_metadata['label'])}
    return label_dict


class elm():
    def __init__(self, hidden_units, activation_function,  x, y, C, elm_type, one_hot=True, random_type='normal'):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = np.array(x)
        self.y = np.array(y)
        self.C = C
        self.class_num = np.unique(self.y).shape[0]     
        self.beta = np.zeros((self.hidden_units, self.class_num))   
        self.elm_type = elm_type
        self.one_hot = one_hot

        # if classification problem and one_hot == True
        if elm_type == 'clf' and self.one_hot:
            self.one_hot_label = np.zeros((self.y.shape[0], self.class_num))
            for i in range(self.y.shape[0]):
                self.one_hot_label[i, int(self.y[i])] = 1

        # Randomly generate the weight matrix and bias vector from input to hidden layer
        # 'uniform': uniform distribution
        # 'normal': normal distribution
        if self.random_type == 'uniform':
            self.W = np.random.uniform(low=0, high=1, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.uniform(low=0, high=1, size=(self.hidden_units, 1))
        if self.random_type == 'normal':
            self.W = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, self.x.shape[1]))
            self.b = np.random.normal(loc=0, scale=0.5, size=(self.hidden_units, 1))

    # compute the output of hidden layer according to different activation function
    def __input2hidden(self, x):
        self.temH = np.dot(self.W, x.T) + self.b

        if self.activation_function == 'sigmoid':
            self.H = 1/(1 + np.exp(- self.temH))

        if self.activation_function == 'relu':
            self.H = self.temH * (self.temH > 0)

        if self.activation_function == 'sin':
            self.H = np.sin(self.temH)

        if self.activation_function == 'tanh':
            self.H = (np.exp(self.temH) - np.exp(-self.temH))/(np.exp(self.temH) + np.exp(-self.temH))

        if self.activation_function == 'leaky_relu':
            self.H = np.maximum(0, self.temH) + 0.1 * np.minimum(0, self.temH)

        return self.H

    # compute the output
    def __hidden2output(self, H):
        self.output = np.dot(H.T, self.beta)
        return self.output


    def fit(self, algorithm):
        # self.time1 = time.time()   # compute running time
        self.H = self.__input2hidden(self.x)
        if self.elm_type == 'clf':
            if self.one_hot:
                self.y_temp = self.one_hot_label
            else:
                self.y_temp = self.y
        if self.elm_type == 'reg':
            self.y_temp = self.y
        # no regularization
        if algorithm == 'no_re':
            self.beta = np.dot(pinv(self.H.T), self.y_temp)
        # faster algorithm 1
        if algorithm == 'solution1':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.tmp1, self.H)
            self.beta = np.dot(self.tmp2, self.y_temp)
        # faster algorithm 2
        if algorithm == 'solution2':
            self.tmp1 = inv(np.eye(self.H.shape[0])/self.C + np.dot(self.H, self.H.T))
            self.tmp2 = np.dot(self.H.T, self.tmp1)
            self.beta = np.dot(self.tmp2.T, self.y_temp)
        # self.time2 = time.time()

        # compute the results
        self.result = self.__hidden2output(self.H)
        # If the problem if classification problem, the output is softmax
        if self.elm_type == 'clf':
            self.result = np.exp(self.result)/np.sum(np.exp(self.result), axis=1).reshape(-1, 1)

        # Evaluate training results
        # If problem is classification, compute the accuracy
        # If problem is regression, compute the RMSE
        if self.elm_type == 'clf':
            self.y_ = np.where(self.result == np.max(self.result, axis=1).reshape(-1, 1))[1]
            self.correct = 0
            for i in range(self.y.shape[0]):
                if self.y_[i] == self.y[i]:
                    self.correct += 1
            self.train_score = self.correct/self.y.shape[0]
        if self.elm_type == 'reg':
            self.train_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y))/self.y.shape[0])
        train_time = ''
        return self.beta, self.train_score, train_time


    def predict(self, x):
        self.H = self.__input2hidden(x)
        self.y_ = self.__hidden2output(self.H)
        if self.elm_type == 'clf':
            self.y_ = np.where(self.y_ == np.max(self.y_, axis=1).reshape(-1, 1))[1]

        return self.y_

    def score(self, x, y):
        self.prediction = self.predict(x)
        if self.elm_type == 'clf':
            self.correct = 0
            for i in range(y.shape[0]):
                if self.prediction[i] == y[i]:
                    self.correct += 1
            self.test_score = self.correct/y.shape[0]
        if self.elm_type == 'reg':
            self.test_score = np.sqrt(np.sum((self.result - self.y) * (self.result - self.y))/self.y.shape[0])

        return self.test_score
    
    
    
def get_polynomialfeatures(input_features):
    feature_vector = np.array(input_features).reshape(1, -1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    expanded_feature_vector = poly.fit_transform(feature_vector)
    return expanded_feature_vector




    










