import torch
import torch.nn as nn
from image_poly_features import ImageClassifier,CustomDataset
from sklearn.model_selection import train_test_split
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tools import get_metrics_with_prob,_get_label_dic,get_metrics_without_prob
from non_images import train_20_sub_domain_models
import numpy as np
import math


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
        # print('true_labels: ',true_labels)
        # print('predicted_labels: ',predicted_labels)
        # print('probs: ',probs)
        # get_metrics_with_prob(true_labels,predicted_labels,probs)
    
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

def build_train(train_values,train_labels,test_values,test_labels,name):
        data1_test = []
        labels_test = []
        for data,label in zip(test_values,test_labels):
            labels_test.append(label)
            data1_test.append(data[name])
        data1_test = [d[1:] for d in data1_test]
        data1_test = np.array(data1_test)
        labels_test = np.array(labels_test)
        
        data1 = []
        labels = []
        for data,label in zip(train_values,train_labels):
            labels.append(label)
            data1.append(data[name])
        data1 = [d[1:] for d in data1]
        data1 = np.array(data1)
        return data1,labels,data1_test,labels_test
    
def main_boost(train_values,train_labels,test_values,test_labels):
    print('*'*50)
    print('\n')
    print('Ensemble: boosting')
    print('\n')
    print('*'*50)
    classifiers = ['model0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    high_performance_classifiers = ['model0']
    for c in classifiers[1:]:
        data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,c)
        # print(data1[0])
        train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,c)
        if test_accuracy < 0.75:
            continue
        else:
            high_performance_classifiers.append(c)
    print('High performance classifiers number: ',len(high_performance_classifiers))
    
    classifier_weights = np.ones(len(high_performance_classifiers))
    errors = np.ones(len(high_performance_classifiers))
    num_samples = len(test_values)
    sample_weights = np.ones(num_samples) / num_samples
    all_predictions = []
    for i, name in enumerate(high_performance_classifiers):
        if i == 0:
            train_prediction,predictions = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
            all_predictions.append(predictions)
        else:# elm models
            data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,name)
            train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,name)
            all_predictions.append(predictions)
        # Error
        predictions = [int(i) for i in predictions]
        test_labels = [int(i) for i in test_labels]
        incorrect = [p != t for p,t in zip(predictions,test_labels)]
        incorrect = np.array(incorrect)
        error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
        errors[i] = error
        boost = 0.5*np.log((1 - error) / error)
        classifier_weights[i] = boost
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
    
    
def main_boost_proposed(train_values,train_labels,test_values,test_labels):
    print('*'*50)
    print('\n')
    print('Ensemble: boosting proposed')
    print('\n')
    print('*'*50)
    classifiers = ['model0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    high_performance_classifiers = ['model0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    # high_performance_classifiers = ['model0']
    # for c in classifiers[1:]:
    #     data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,c)
    #     # print(data1[0])
    #     train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,c)
    #     if test_accuracy < 0.75:
    #         continue
    #     else:
    #         high_performance_classifiers.append(c)
    # print('High performance classifiers number: ',len(high_performance_classifiers))
    
    classifier_weights = np.ones(len(high_performance_classifiers))
    errors = np.ones(len(high_performance_classifiers))
    num_samples = len(test_values)
    sample_weights = np.ones(num_samples) / num_samples
    all_predictions = []
    for i, name in enumerate(high_performance_classifiers):
        if i == 0:
            train_prediction,predictions = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
            all_predictions.append(predictions)
        else:# elm models
            data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,name)
            train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,name)
            all_predictions.append(predictions)
        # Error
        predictions = [int(i) for i in predictions]
        test_labels = [int(i) for i in test_labels]
        incorrect = [p != t for p,t in zip(predictions,test_labels)]
        incorrect = np.array(incorrect)
        error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
        errors[i] = error
        boost = 0.5*np.log((1 - error) / error) + 0.001*math.exp(1 - error)
        classifier_weights[i] = boost
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


def vote_base(train_values,train_labels,test_values,test_labels):
    classifiers = ['model0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    high_performance_classifiers = ['model0']
    for c in classifiers[1:]:
        data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,c)
        # print(data1[0])
        train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,c)
        if test_accuracy < 0.75:
            continue
        else:
            high_performance_classifiers.append(c)
    print('High performance classifiers number: ',len(high_performance_classifiers))
    
    all_predictions = []
    for i, name in enumerate(high_performance_classifiers):
        if i == 0:
            train_prediction,predictions = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
            all_predictions.append(predictions)
        else:# elm models
            data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,name)
            train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,name)
            all_predictions.append(predictions)
    return all_predictions
    
    


def main_vote(train_values,train_labels,test_values,test_labels):
    print('*'*50)
    print('\n')
    print('Ensemble: Voting')
    print('\n')
    print('*'*50)
    all_predictions = vote_base(train_values,train_labels,test_values,test_labels)
    final_preds = combine_predictions(all_predictions)
    print('final_preds: ',final_preds)
    get_metrics_without_prob(test_labels,final_preds)   


def main_bagging(train_values,train_labels,test_values,test_labels):
    print('*'*50)
    print('\n')
    print('Ensemble: Bagging')
    print('\n')
    print('*'*50)
    bagging_res_list = []
    for _ in range(2):
        bagging_res_list.extend(vote_base(train_values,train_labels,test_values,test_labels))
    
    final_preds = combine_predictions(bagging_res_list)
    print('final_preds: ',final_preds)
    get_metrics_without_prob(test_labels,final_preds)   

def main_stacking(train_values,train_labels,test_values,test_labels):
    print('*'*50)
    print('\n')
    print('Ensemble: Stacking')
    print('\n')
    print('*'*50)
    from sklearn.ensemble import RandomForestClassifier
    
    classifiers = ['model0','data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    high_performance_classifiers = ['model0']
    for c in classifiers[1:]:
        data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,c)
        # print(data1[0])
        train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,c)
        if test_accuracy < 0.75:
            continue
        else:
            high_performance_classifiers.append(c)
    print('High performance classifiers number: ',len(high_performance_classifiers))
    all_train_predictions = []
    all_train_labels = []
    all_test_predictions = []
    for i, name in enumerate(high_performance_classifiers):
        if i == 0:
            train_prediction,predictions = main_only_image(train_values,train_labels,test_values,test_labels,ensemble=True)
            all_train_predictions.append(train_prediction)
            all_test_predictions.append(predictions)
            all_train_labels.append(train_labels)
        else:# elm models
            data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,name)
            train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,name)
            all_train_predictions.append(train_prediction)
            all_test_predictions.append(predictions)
            all_train_labels.append(labels)
    all_train_predictions = np.array(all_train_predictions)
    all_test_predictions = np.array(all_test_predictions)
    train_labels = np.array(train_labels)
    all_train_predictions = np.transpose(all_train_predictions)
    all_test_predictions = np.transpose(all_test_predictions)
    print('all_train_predictions shape: ',all_train_predictions.shape)
    print('all_test_predictions shape: ',all_test_predictions.shape)
    print('train_labels shape: ',train_labels.shape)

    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)

    print()
    meta_model.fit(all_train_predictions, train_labels)

    final_preds = meta_model.predict(all_test_predictions)
    
    print('final_preds: ',final_preds)
    get_metrics_without_prob(test_labels,final_preds)

if __name__ == '__main__':
    with open('./datasets/top10/all_data_last_image.pkl', 'rb') as f:
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
    main_boost(train_values,train_labels,test_values,test_labels)
    main_boost_proposed(train_values,train_labels,test_values,test_labels)
    main_bagging(train_values,train_labels,test_values,test_labels)
    main_stacking(train_values,train_labels,test_values,test_labels)
    main_vote(train_values,train_labels,test_values,test_labels)
    
    
        
    



