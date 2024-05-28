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
    
    train_prediction,predictions1 = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
    data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,'dataall')
    # 初始化分类器
    svm_classifier = SVC(kernel='linear')
    lg_classifier = LogisticRegression()
    dt_classifier = DecisionTreeClassifier()
    rf_classifier = RandomForestClassifier()
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=5)

    # 训练模型
    svm_classifier.fit(data1, labels)
    lg_classifier.fit(data1, labels)
    dt_classifier.fit(data1, labels)
    rf_classifier.fit(data1, labels)
    mlp_classifier.fit(data1, labels)

    # 测试模型
    svm_pred = svm_classifier.predict(data1_test)
    lg_pred = lg_classifier.predict(data1_test)
    dt_pred = dt_classifier.predict(data1_test)
    rf_pred = rf_classifier.predict(data1_test)
    mlp_pred = mlp_classifier.predict(data1_test)

    
    print('model0 + SVM + LG   final_preds: ',combine_predictions([predictions1, svm_pred, lg_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, svm_pred, lg_pred]))
    print('\n')
    print('*'*40)
    print('model0 + SVM + dt_pred   final_preds: ',combine_predictions([predictions1, svm_pred, dt_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, svm_pred, dt_pred]))
    print('\n')
    print('*'*40)
    print('model0 + SVM + rf_pred   final_preds: ',combine_predictions([predictions1, svm_pred, rf_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, svm_pred, rf_pred]))
    print('\n')
    print('*'*40)
    print('model0 + SVM + mlp_pred   final_preds: ',combine_predictions([predictions1, svm_pred, mlp_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, svm_pred, mlp_pred]))
    print('\n')
    print('*'*40)
    print('model0 + lg_pred + dt_pred   final_preds: ',combine_predictions([predictions1, lg_pred, dt_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, lg_pred, dt_pred]))
    print('\n')
    print('*'*40)
    print('model0 + lg_pred + rf_pred   final_preds: ',combine_predictions([predictions1, lg_pred, rf_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, lg_pred, rf_pred]))
    print('\n')
    print('*'*40)
    print('model0 + lg_pred + mlp_pred   final_preds: ',combine_predictions([predictions1, lg_pred, mlp_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, lg_pred, mlp_pred]))
    print('\n')
    print('*'*40)
    print('model0 + dt_pred + rf_pred   final_preds: ',combine_predictions([predictions1, dt_pred, rf_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, dt_pred, rf_pred]))
    print('\n')
    print('*'*40)
    print('model0 + dt_pred + mlp_pred   final_preds: ',combine_predictions([predictions1, dt_pred, mlp_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, dt_pred, mlp_pred]))
    print('\n')
    print('*'*40)
    print('model0 + rf_pred + mlp_pred   final_preds: ',combine_predictions([predictions1, rf_pred, mlp_pred]))
    get_metrics_without_prob(labels_test,combine_predictions([predictions1, rf_pred, mlp_pred]))
    print('\n')
    print('*'*40)
    
    
        

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



