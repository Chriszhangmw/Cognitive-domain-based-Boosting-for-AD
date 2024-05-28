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
from sklearn.preprocessing import PolynomialFeatures

def get_polynomialfeatures(input_features,d):
    # 将特征列表转换为 m 维度的向量
    feature_vector = np.array(input_features).reshape(1, -1)
    # print(feature_vector)
    # 定义多项式核函数并将特征扩展到 n 维度
    poly = PolynomialFeatures(degree=d, include_bias=False)
    expanded_feature_vector = poly.fit_transform(feature_vector)
    return expanded_feature_vector

def main_ad_boost(train_values,train_labels,test_values,test_labels):
    def build_train(train_values,train_labels,test_values,test_labels,name):
        data1_test = []
        labels_test = []
        for data,label in zip(test_values,test_labels):
            labels_test.append(label)
            data1_test.append(data[name])
        data1_test = [d[1:] for d in data1_test]
        data1_test = np.array(data1_test)
        data1_test_dense = [get_polynomialfeatures(v,5) for v in data1_test]
        data1_test_dense = np.array(data1_test_dense)
        sample_num,_,features_num = data1_test_dense.shape
        data1_test_dense = data1_test_dense.reshape(sample_num, features_num)
        labels_test = np.array(labels_test)
        
        data1 = []
        labels = []
        for data,label in zip(train_values,train_labels):
            labels.append(label)
            # data1.append(stdsc.fit_transform(data[name]))
            data1.append(data[name])
        data1 = [d[1:] for d in data1]
        data1 = np.array(data1)
        data1_dense = [get_polynomialfeatures(v,5) for v in data1]
        data1_dense = np.array(data1_dense)
        sample_num,_,features_num = data1_dense.shape
        data1_dense = data1_dense.reshape(sample_num, features_num)
        return data1_dense,labels,data1_test_dense,labels_test
    
    print('*'*40)
    print('\n')
    print('Training samples number: ',len(train_values))
    print('\n')
    print('Testing samples number: ',len(test_values))
    print('\n')
    print('*'*40)
    
    classifiers = ['data1','data2','data3','data4','data5','data6','data7','data8','data9','data10','data11','data12','data13',
                   'data14','data15','data16','data17','data18','data19','data20','data21','data22','data23','data24','data25','data26',
                   'data27','data28','data29','data30','data31','data32','data33','data34','data35','data36','data37','data38','data39',
                   'data40','data41','data42','data43','data44','data45','data46','data47','data48','data49','data50','data51','data52',
                   'data53','data54','data55','data56','data57','data58','data59','data60','data61','data62','data63']
    
    high_performance_classifiers = []
    for c in classifiers:
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
    # num_samples = len(train_values)
    num_samples = len(test_values)
    sample_weights = np.ones(num_samples) / num_samples
    all_predictions = []
    for i, name in enumerate(high_performance_classifiers):
        data1,labels,data1_test,labels_test = build_train(train_values,train_labels,test_values,test_labels,name)
        # print(data1[0])
        train_accuracy,test_accuracy,train_prediction, predictions = train_20_sub_domain_models(data1,labels,data1_test,labels_test,name)
        all_predictions.append(predictions)
        # Error
        predictions = [int(i) for i in predictions]
        test_labels = [int(i) for i in test_labels]
        incorrect = [p != t for p,t in zip(predictions,test_labels)]
        incorrect = np.array(incorrect)
        error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
        errors[i] = error
        # Boost weight
        # boost = np.log((1 - error) / error) + np.log(len(classifiers) - 1)
        boost = 0.5*np.log((1 - error) / error) + 0.01*math.exp(1 - error)
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
main_ad_boost(train_values,train_labels,test_values,test_labels)