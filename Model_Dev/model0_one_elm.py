import torch
import torch.nn as nn
from image_poly_features import ImageClassifier,CustomDataset
from sklearn.model_selection import train_test_split
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tools import get_metrics_with_prob,_get_label_dic,get_metrics_without_prob
from non_images import train_20_sub_domain_models,generate_all_non_image
import numpy as np
import math
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



from collections import Counter

def combine_predictions(prediction_lists):
    combined_predictions = []
    for predictions in zip(*prediction_lists):
        # 对每个样本的预测结果进行投票
        counts = Counter(predictions)
        # 获取得票最多的类别
        majority_vote = counts.most_common(1)[0][0]
        combined_predictions.append(majority_vote)
    return combined_predictions




# Define hyperparameters
num_epochs = 3
learning_rate = 0.001
num_classes = 3  # Assuming 3 classes for classification

def main_only_image(train_values,train_labels,test_values,test_labels,ensemble=False):
    # Create custom datasets for train and test
    train_dataset = CustomDataset(train_values,train_labels)
    test_dataset = CustomDataset(test_values,test_labels)
    # Define batch size for training
    batch_size = 8  # You can adjust this batch size as needed
    # Create data loaders for training and test datasets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
 
    # Initialize the model
    model = ImageClassifier(num_classes)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

    # Test the model
    model.eval()
    with torch.no_grad():
        true_labels = []
        predicted_labels = []
        probs = []  # List to store predicted probabilities
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Store true labels, predicted labels, and probabilities
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())

            # Apply softmax to get class probabilities
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(outputs)
            probs.extend(probabilities.tolist())  # Append probabilities to the list
        print('true_labels: ',true_labels)
        print('predicted_labels: ',predicted_labels)
        print('probs: ',probs)
        get_metrics_with_prob(true_labels,predicted_labels,probs)
    
    # Test the train data
    if ensemble:
        model.eval()
        with torch.no_grad():
            true_labels = []
            train_prediction = []
            probs = []  # List to store predicted probabilities
            for images, labels in train_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                # Store true labels, predicted labels, and probabilities
                true_labels.extend(labels.tolist())
                train_prediction.extend(predicted.tolist())
    else:
        train_prediction = []
    return train_prediction,predicted_labels


def main_ad_boost(train_values,train_labels,test_values,test_labels):
    def build_train(train_values,train_labels,test_values,test_labels,name):
        data1_test = []
        labels_test = []
        for data,label in zip(test_values,test_labels):
            labels_test.append(label)
            data1_test.append(data[name])
        data1_test = [d[1:] for d in data1_test]
        data1_test = np.array(data1_test)
        labels_test = np.array(labels_test)
        # sample_num,_,features_num = data1_test.shape
        # data1_test = data1_test.reshape(sample_num, features_num)
        
        data1 = []
        labels = []
        for data,label in zip(train_values,train_labels):
            labels.append(label)
            # data1.append(stdsc.fit_transform(data[name]))
            data1.append(data[name])
        data1 = [d[1:] for d in data1]
        data1 = np.array(data1)
        # sample_num,_,features_num = data1.shape
        # data1 = data1.reshape(sample_num, features_num)
        return data1,labels,data1_test,labels_test
    
    print('*'*40)
    print('\n')
    print('Training samples number: ',len(train_values))
    print('\n')
    print('Testing samples number: ',len(test_values))
    print('\n')
    print('*'*40)
    
    
    
    classifiers = ['model0','dataall']
    classifier_weights = np.ones(len(classifiers))
    errors = np.ones(len(classifiers))
    num_samples = len(test_values)
    sample_weights = np.ones(num_samples) / num_samples
    all_predictions = []
    for i,c in enumerate(classifiers):
        if c == 'model0':
            train_prediction,predictions = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
            all_predictions.append(predictions)
        else:
            data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,'dataall')
            train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,'dataall')
            all_predictions.append(predictions)
        predictions = [int(i) for i in predictions]
        test_labels = [int(i) for i in test_labels]
        incorrect = [p != t for p,t in zip(predictions,test_labels)]
        incorrect = np.array(incorrect)
        error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
        errors[i] = error
        # Boost weight
        # boost = np.log((1 - error) / error) + np.log(len(classifiers) - 1)
        boost = 0.5*np.log((1 - error) / error) + k*math.exp(1 - error)
        classifier_weights[i] = boost
        # Update sample weights
        sample_weights *= np.exp(boost * incorrect * ((sample_weights > 0) | (boost < 0)))
    classifier_weights /= np.sum(classifier_weights)
    print('classifier_weights: ',classifier_weights)
    print('errors: ',errors)
    
    all_predictions = np.array(all_predictions)
    weighted_predictions = np.zeros((len(all_predictions[0]),3))
    for i in range(len(classifier_weights)):
        for j in range(len(all_predictions[0])):
            weighted_predictions[j,all_predictions[i,j]] += classifier_weights[i]
    final_preds = np.argmax(weighted_predictions,axis=1)
    print('final_preds: ',final_preds)
    get_metrics_without_prob(test_labels,final_preds)
    
    
        

if __name__ == '__main__':
    with open('./datasets/top10/model0_ml.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # Extract keys and values from data_dict
    keys = list(data_dict.keys())
    values = list(data_dict.values())


    # Split the data into train and test sets
    train_keys, test_keys, train_values, test_values = train_test_split(keys, values, test_size=0.2, random_state=2024)
    labels_dict = _get_label_dic()
    train_labels = []
    test_labels = []
    for key in train_keys:
        train_labels.append(labels_dict[key])
    for key in test_keys:
        test_labels.append(labels_dict[key])
    main_ad_boost(train_values,train_labels,test_values,test_labels)



