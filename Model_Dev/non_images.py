
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
from tools import elm
import random


"""
data processing
"""


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

def fill_nan_data(values):
    new_values = []
    for v in values:
        if np.isnan(v):
            new_values.append(random.choice([0.0, 1.0]))
        else:
            new_values.append(v)
    return new_values
    
def generate_20_sub_datasets():
    
    import pandas as pd
    label_metadata = pd.read_csv('./datasets/label_meta.csv')
    from datetime import datetime

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
    age_dict = {f"{a}_{b}": c for a, b, c in zip(label_metadata['Subject'], label_metadata['date'], label_metadata['Age'])}
    sex_dict = {f"{a}_{b}": c for a, b, c in zip(label_metadata['Subject'], label_metadata['date'], label_metadata['Sex'])}
    
    import pandas as pd
    mmsedata = pd.read_csv('./datasets/Neuropsychological/MMSE_23Jan2024.csv')
    mmsedata = mmsedata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    mmsedata['key'] = mmsedata.apply(generate_key, axis=1)
    Orientation_index = [11,12,13,14,15,16,17,18,19,20]
    immediateMemory_index = [21,22,23]
    Attention_and_calculation_index = [24,25,26,27,28,29]
    lateMemory_index = [30,31,32]
    naming_index = [33,34]
    Read_and_obey_index = [35,39]
    others_index = [36,37,38,40,41]

    res = {}

    for k,v in label_dict.items():
        filtered_row = mmsedata.loc[mmsedata['key'] == k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            filtered_row = mmsedata.loc[mmsedata['PTID'] == pid].iloc[0]
        else:
            filtered_row = filtered_row.squeeze()
        
        info = {}
        info['label'] = v
        info['age'] = age_dict[k]
        info['sex'] = sex_dict[k]
        
        mmseinfo = {}
        mmseinfo['Orientation'] = filtered_row[Orientation_index].tolist()
        mmseinfo['immediateMemory']= filtered_row[immediateMemory_index].tolist()
        mmseinfo['Attention_and_calculation']= filtered_row[Attention_and_calculation_index].tolist()
        mmseinfo['lateMemory']= filtered_row[lateMemory_index].tolist()
        mmseinfo['naming']= filtered_row[naming_index].tolist()
        mmseinfo['Read_and_obey']= filtered_row[Read_and_obey_index].tolist()
        mmseinfo['others']= filtered_row[others_index].tolist()
        mmseinfo['score'] = filtered_row['MMSCORE']
        info['mmse'] = mmseinfo
        res[k] = info
    
    import pandas as pd
    ccbdata = pd.read_csv('./datasets/Neuropsychological/CBBCOMP_23Jan2024.csv')
    ccbdata = ccbdata.apply(lambda col: col.fillna(col.mode()[0]))
    # mmsedata['VISDATE'].values.tolist()[2]
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    ccbdata['key'] = ccbdata.apply(generate_key, axis=1)
    
    ccb_index = [10,11]

    for k,v in res.items():
        filtered_row = ccbdata.loc[ccbdata['key'] == k]
        ccbinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            # print(pid)
            try:
                filtered_row = ccbdata.loc[ccbdata['PTID'] == pid].iloc[0]
            except:
                ccbinfo['ccb'] = []
                info['ccb'] = ccbinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        
        
        ccbinfo['ccb'] = filtered_row[ccb_index].tolist()
        info['ccb'] = ccbinfo
        res[k] = info
        
    import pandas as pd
    cdrdata = pd.read_csv('./datasets/Neuropsychological/CDR_23Jan2024.csv')
    cdrdata = cdrdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    cdrdata['key'] = cdrdata.apply(generate_key, axis=1)
    
    cdr_index = [14,15,16,17,18,19,20]

    for k,v in res.items():
        info = res[k]
        filtered_row = cdrdata.loc[cdrdata['key'] == k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            # print(pid)
            try:
                filtered_row = cdrdata.loc[cdrdata['PTID'] == pid].iloc[0]
            except:
                cdrinfo['cdr'] = []
                info['cdr'] = cdrinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        cdrinfo = {}
        cdrinfo['cdr'] = filtered_row[cdr_index].tolist()
        info['cdr'] = cdrinfo
        res[k] = info
    
    import pandas as pd
    MODHACHdata = pd.read_csv('./datasets/Neuropsychological/MODHACH_23Jan2024.csv')
    MODHACHdata = MODHACHdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    MODHACHdata['key'] = MODHACHdata.apply(generate_key, axis=1)
    domain1_index = [11,12,13,14,15,16,17,18,19]
    for k,v in res.items():
        filtered_row = MODHACHdata.loc[MODHACHdata['key'] == k]
        MODHACHinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = MODHACHdata.loc[MODHACHdata['PTID'] == pid].iloc[0]
            except:
                MODHACHinfo['MODHACH'] = []
                info['MODHACH'] = MODHACHinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        MODHACHinfo['domain1'] = filtered_row[domain1_index].tolist()
        
        info['MODHACH'] = MODHACHinfo
        res[k] = info
    
    
    import pandas as pd
    NEUROBATdata = pd.read_csv('./datasets/Neuropsychological/NEUROBAT_23Jan2024.csv')
    NEUROBATdata = NEUROBATdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    NEUROBATdata['key'] = NEUROBATdata.apply(generate_key, axis=1)

    domain1_index = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    domain2_index = [41,42,43,44,45,46,47,48,49,50,51]
    for k,v in res.items():
        filtered_row = NEUROBATdata.loc[NEUROBATdata['key'] == k]
        NEUROBATinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = NEUROBATdata.loc[NEUROBATdata['PTID'] == pid].iloc[0]
            except:
                NEUROBATinfo['NEUROBA'] = []
                info['NEUROBA'] = NEUROBATinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        NEUROBATinfo['domain1'] = filtered_row[domain1_index].tolist()
        NEUROBATinfo['domain2'] = filtered_row[domain2_index].tolist()
        info['NEUROBA'] = NEUROBATinfo
        res[k] = info
    
    
    import pandas as pd
    NPIQdata = pd.read_csv('./datasets/Neuropsychological/NPIQ_23Jan2024.csv')
    NPIQdata = NPIQdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    NPIQdata['key'] = NPIQdata.apply(generate_key, axis=1)

    domain1_index = [11,12,13,14,15,16,17,18,19,20,21,22]
    domain2_index = [23,24,26,27,28,29,30,31,32,33,34,35,36]
    for k,v in res.items():
        filtered_row = NPIQdata.loc[NPIQdata['key'] == k]
        NPIQinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = NPIQdata.loc[NPIQdata['PTID'] == pid].iloc[0]
            except:
                NPIQinfo['NPIQ'] = []
                info['NPIQ'] = NPIQinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        NPIQinfo['domain1'] = filtered_row[domain1_index].tolist()
        NPIQinfo['domain2'] = filtered_row[domain2_index].tolist()
        info['NPIQ'] = NPIQinfo
        res[k] = info
    
    
    import pandas as pd
    ADASdata = pd.read_csv('./datasets/Neuropsychological/ADAS_ADNI1_23Jan2024.csv')
    ADASdata = ADASdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['USERDATE']))
    ADASdata['key'] = ADASdata.apply(generate_key, axis=1)


    domain1_index = [30,31,32,33,34,35,36,37,38,39,40,41,42]
    for k,v in res.items():
        filtered_row = ADASdata.loc[ADASdata['key'] == k]
        ADASinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = ADASdata.loc[ADASdata['PTID'] == pid].iloc[0]
            except:
                ADASinfo['ADAS'] = []
                info['ADAS'] = ADASinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        ADASinfo['domain1'] = filtered_row[domain1_index].tolist()
        info['ADAS'] = ADASinfo
        res[k] = info
    data1 = {}
    data2 = {}
    data3 = {}
    data4 = {}
    data5 = {}
    data6 = {}
    data7 = {}
    data8 = {}
    data9 = {}
    data10 = {}
    data11 = {}
    data12 = {}
    data13 = {}
    rdata1 = {}
    rdata2 = {}
    rdata3 = {}
    rdata4 = {}
    rdata5 = {}
    rdata6 = {}
    rdata7 = {}
    rdata8 = {}
    rdata9 = {}
    rdata10 = {}
    rdata11 = {}
    rdata12 = {}
    rdata13 = {}
    rdata14 = {}
    rdata15 = {}
    rdata16 = {}
    rdata17 = {}
    rdata18 = {}
    rdata19 = {}
    rdata20 = {}
    rdata21 = {}
    rdata22 = {}
    rdata23 = {}
    rdata24 = {}
    rdata25 = {}
    rdata26 = {}
    rdata27 = {}
    rdata28 = {}
    rdata29 = {}
    rdata30 = {}
    rdata31 = {}
    rdata32 = {}
    rdata33 = {}
    rdata34 = {}
    rdata35 = {}
    rdata36 = {}
    rdata37 = {}
    rdata38 = {}
    rdata39 = {}
    rdata40 = {}
    rdata41 = {}
    rdata42 = {}
    rdata43 = {}
    rdata44 = {}
    rdata45 = {}
    rdata46 = {}
    rdata47 = {}
    rdata48 = {}
    rdata49 = {}
    rdata50 = {}
    rdatalist = [rdata1, rdata2, rdata3, rdata4, rdata5, rdata6, rdata7, rdata8, rdata9, rdata10, rdata11, rdata12, rdata13, rdata14, rdata15, rdata16, rdata17, rdata18, rdata19, rdata20, rdata21, rdata22, rdata23, rdata24, rdata25, rdata26, rdata27, rdata28, rdata29, rdata30, rdata31, rdata32, rdata33, rdata34, rdata35, rdata36, rdata37, rdata38, rdata39, rdata40, rdata41, rdata42, rdata43, rdata44, rdata45, rdata46, rdata47, rdata48, rdata49, rdata50]

    selected_lists = []
    seed = 202405
    np.random.seed(seed)
    for _ in range(50):
        selected_indices = random.sample(range(17), 4)
        selected_lists.append(selected_indices)
    print(selected_lists)
        
    for index, sample in res.items():
        label = [sample['label']]
        age = [sample['age']]
        sex = sample['sex']
        if sex == 'F':
            sex = [1]
        else:
            sex = [0]
        Orientation = sample['mmse']['Orientation']
        immediateMemory = sample['mmse']['immediateMemory']
        Attention_and_calculation = sample['mmse']['Attention_and_calculation']
        lateMemory = sample['mmse']['lateMemory']
        naming = sample['mmse']['naming']
        Read_and_obey = sample['mmse']['Read_and_obey']
        others = sample['mmse']['others']
        ccb = sample['mmse']['Orientation']
        cdr = sample['cdr']['cdr']
        MODHACH_domain1 = sample['MODHACH']['domain1']
        NEUROBA_domain1 = sample['NEUROBA']['domain1']
        NEUROBA_domain2 = sample['NEUROBA']['domain2']
        NPIQ_domain1 = sample['NPIQ']['domain1']
        NPIQ_domain2 = sample['NPIQ']['domain2']
        ADAS_domain1 = sample['ADAS']['domain1']

        feature1 = age+sex+Orientation+Attention_and_calculation
        feature2 = age+sex+immediateMemory+lateMemory
        feature3 = age+sex+naming+lateMemory
        feature4 = age+sex+Read_and_obey+naming+lateMemory
        feature5 = age+sex+others+lateMemory
        feature6 = age+sex+ccb+Orientation
        feature7 = age+sex+ccb+cdr
        feature8 = age+sex+MODHACH_domain1
        feature9 = age+sex+NEUROBA_domain1
        feature10 = age+sex+NEUROBA_domain2
        feature11 = age+sex+NPIQ_domain1
        feature12 = age+sex+NPIQ_domain2
        feature13 = age+sex+ADAS_domain1
        data1[index]=feature1
        data2[index]=feature2
        data3[index]=feature3
        data4[index]=feature4
        data5[index]=feature5
        data6[index]=feature6
        data7[index]=feature7
        data8[index]=feature8
        data9[index]=feature9
        data10[index]=feature10
        data11[index]=feature11
        data12[index]=feature12
        data13[index]=feature13

        original_list = [age,sex,Orientation,immediateMemory,Attention_and_calculation,lateMemory,
                          naming,Read_and_obey,others,ccb,cdr,MODHACH_domain1,NEUROBA_domain1,NEUROBA_domain2,NPIQ_domain1,NPIQ_domain2,ADAS_domain1]
        
        for i in range(50):
            selected_indexs = selected_lists[i]
            selected_elements = [original_list[i] for i in selected_indexs]
            temp_res = []
            for d in selected_elements:
                temp_res.extend(d)
            rdatalist[i][index] = temp_res
    return data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,rdata1, rdata2, rdata3, rdata4, rdata5, rdata6, rdata7, rdata8, rdata9, rdata10, rdata11, rdata12, rdata13, rdata14, rdata15, rdata16, rdata17, rdata18, rdata19, rdata20, rdata21, rdata22, rdata23, rdata24, rdata25, rdata26, rdata27, rdata28, rdata29, rdata30, rdata31, rdata32, rdata33, rdata34, rdata35, rdata36, rdata37, rdata38, rdata39, rdata40, rdata41, rdata42, rdata43, rdata44, rdata45, rdata46, rdata47, rdata48, rdata49, rdata50


def train_20_sub_domain_models(train_values,train_labels,test_values,test_labels,name):
    # built model and train
    model1 = elm(hidden_units=50, activation_function='leaky_relu', random_type='normal', x=train_values, y=train_labels, C=0.1, elm_type='clf')
    beta, train_accuracy, running_time = model1.fit('solution2')
    # print("classifier beta:\n", beta)
    # print('*'*40)
    # print('Domain name: ',name)
    # print('\n')
    # print("classifier train accuracy:", train_accuracy)
    # print('classifier running time:', running_time)
    # print('\n')
    # test
    prediction = model1.predict(test_values)
    # print("classifier test prediction:", prediction)
    test_accuracy = model1.score(test_values, test_labels)
    # print('classifier test accuracy:', test_accuracy)
    # print('\n')
    # print('*'*40)
    train_prediction = model1.predict(train_values)
    return train_accuracy,test_accuracy,train_prediction, prediction

def generate_all_non_image():
    
    import pandas as pd
    label_metadata = pd.read_csv('./datasets/label_meta.csv')
    from datetime import datetime

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
    age_dict = {f"{a}_{b}": c for a, b, c in zip(label_metadata['Subject'], label_metadata['date'], label_metadata['Age'])}
    sex_dict = {f"{a}_{b}": c for a, b, c in zip(label_metadata['Subject'], label_metadata['date'], label_metadata['Sex'])}
    
    import pandas as pd
    mmsedata = pd.read_csv('./datasets/Neuropsychological/MMSE_23Jan2024.csv')
    mmsedata = mmsedata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    mmsedata['key'] = mmsedata.apply(generate_key, axis=1)
    Orientation_index = [11,12,13,14,15,16,17,18,19,20]
    immediateMemory_index = [21,22,23]
    Attention_and_calculation_index = [24,25,26,27,28,29]
    lateMemory_index = [30,31,32]
    naming_index = [33,34]
    Read_and_obey_index = [35,39]
    others_index = [36,37,38,40,41]

    res = {}

    for k,v in label_dict.items():
        filtered_row = mmsedata.loc[mmsedata['key'] == k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            filtered_row = mmsedata.loc[mmsedata['PTID'] == pid].iloc[0]
        else:
            filtered_row = filtered_row.squeeze()
        
        info = {}
        info['label'] = v
        info['age'] = age_dict[k]
        info['sex'] = sex_dict[k]
        
        mmseinfo = {}
        mmseinfo['Orientation'] = filtered_row[Orientation_index].tolist()
        mmseinfo['immediateMemory']= filtered_row[immediateMemory_index].tolist()
        mmseinfo['Attention_and_calculation']= filtered_row[Attention_and_calculation_index].tolist()
        mmseinfo['lateMemory']= filtered_row[lateMemory_index].tolist()
        mmseinfo['naming']= filtered_row[naming_index].tolist()
        mmseinfo['Read_and_obey']= filtered_row[Read_and_obey_index].tolist()
        mmseinfo['others']= filtered_row[others_index].tolist()
        mmseinfo['score'] = filtered_row['MMSCORE']
        info['mmse'] = mmseinfo
        res[k] = info
    
    import pandas as pd
    ccbdata = pd.read_csv('./datasets/Neuropsychological/CBBCOMP_23Jan2024.csv')
    ccbdata = ccbdata.apply(lambda col: col.fillna(col.mode()[0]))
    # mmsedata['VISDATE'].values.tolist()[2]
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    ccbdata['key'] = ccbdata.apply(generate_key, axis=1)
    
    ccb_index = [10,11]

    for k,v in res.items():
        filtered_row = ccbdata.loc[ccbdata['key'] == k]
        ccbinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            # print(pid)
            try:
                filtered_row = ccbdata.loc[ccbdata['PTID'] == pid].iloc[0]
            except:
                ccbinfo['ccb'] = []
                info['ccb'] = ccbinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        
        
        ccbinfo['ccb'] = filtered_row[ccb_index].tolist()
        info['ccb'] = ccbinfo
        res[k] = info
        
    import pandas as pd
    cdrdata = pd.read_csv('./datasets/Neuropsychological/CDR_23Jan2024.csv')
    cdrdata = cdrdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))

    cdrdata['key'] = cdrdata.apply(generate_key, axis=1)
    
    cdr_index = [14,15,16,17,18,19,20]

    for k,v in res.items():
        info = res[k]
        filtered_row = cdrdata.loc[cdrdata['key'] == k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            # print(pid)
            try:
                filtered_row = cdrdata.loc[cdrdata['PTID'] == pid].iloc[0]
            except:
                cdrinfo['cdr'] = []
                info['cdr'] = cdrinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        cdrinfo = {}
        cdrinfo['cdr'] = filtered_row[cdr_index].tolist()
        info['cdr'] = cdrinfo
        res[k] = info
    
    import pandas as pd
    MODHACHdata = pd.read_csv('./datasets/Neuropsychological/MODHACH_23Jan2024.csv')
    MODHACHdata = MODHACHdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    MODHACHdata['key'] = MODHACHdata.apply(generate_key, axis=1)
    domain1_index = [11,12,13,14,15,16,17,18,19]
    for k,v in res.items():
        filtered_row = MODHACHdata.loc[MODHACHdata['key'] == k]
        MODHACHinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = MODHACHdata.loc[MODHACHdata['PTID'] == pid].iloc[0]
            except:
                MODHACHinfo['MODHACH'] = []
                info['MODHACH'] = MODHACHinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        MODHACHinfo['domain1'] = filtered_row[domain1_index].tolist()
        
        info['MODHACH'] = MODHACHinfo
        res[k] = info
    
    
    import pandas as pd
    NEUROBATdata = pd.read_csv('./datasets/Neuropsychological/NEUROBAT_23Jan2024.csv')
    NEUROBATdata = NEUROBATdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    NEUROBATdata['key'] = NEUROBATdata.apply(generate_key, axis=1)

    domain1_index = [11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
    domain2_index = [41,42,43,44,45,46,47,48,49,50,51]
    for k,v in res.items():
        filtered_row = NEUROBATdata.loc[NEUROBATdata['key'] == k]
        NEUROBATinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = NEUROBATdata.loc[NEUROBATdata['PTID'] == pid].iloc[0]
            except:
                NEUROBATinfo['NEUROBA'] = []
                info['NEUROBA'] = NEUROBATinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        NEUROBATinfo['domain1'] = filtered_row[domain1_index].tolist()
        NEUROBATinfo['domain2'] = filtered_row[domain2_index].tolist()
        info['NEUROBA'] = NEUROBATinfo
        res[k] = info
    
    
    import pandas as pd
    NPIQdata = pd.read_csv('./datasets/Neuropsychological/NPIQ_23Jan2024.csv')
    NPIQdata = NPIQdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['VISDATE']))
    NPIQdata['key'] = NPIQdata.apply(generate_key, axis=1)

    domain1_index = [11,12,13,14,15,16,17,18,19,20,21,22]
    domain2_index = [23,24,26,27,28,29,30,31,32,33,34,35,36]
    for k,v in res.items():
        filtered_row = NPIQdata.loc[NPIQdata['key'] == k]
        NPIQinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = NPIQdata.loc[NPIQdata['PTID'] == pid].iloc[0]
            except:
                NPIQinfo['NPIQ'] = []
                info['NPIQ'] = NPIQinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        NPIQinfo['domain1'] = filtered_row[domain1_index].tolist()
        NPIQinfo['domain2'] = filtered_row[domain2_index].tolist()
        info['NPIQ'] = NPIQinfo
        res[k] = info
    
    
    import pandas as pd
    ADASdata = pd.read_csv('./datasets/Neuropsychological/ADAS_ADNI1_23Jan2024.csv')
    ADASdata = ADASdata.apply(lambda col: col.fillna(col.mode()[0]))
    def generate_key(row):
        return str(row['PTID'] + '_' +  str(row['USERDATE']))
    ADASdata['key'] = ADASdata.apply(generate_key, axis=1)


    domain1_index = [30,31,32,33,34,35,36,37,38,39,40,41,42]
    for k,v in res.items():
        filtered_row = ADASdata.loc[ADASdata['key'] == k]
        ADASinfo = {}
        info = res[k]
        if filtered_row.empty:
            pid = k.split('_')[:3]
            pid = '_'.join(pid)
            try:
                filtered_row = ADASdata.loc[ADASdata['PTID'] == pid].iloc[0]
            except:
                ADASinfo['ADAS'] = []
                info['ADAS'] = ADASinfo
                res[k] = info
                continue
        else:
            filtered_row = filtered_row.squeeze()
        ADASinfo['domain1'] = filtered_row[domain1_index].tolist()
        info['ADAS'] = ADASinfo
        res[k] = info
    dataall = {}
        
    for index, sample in res.items():
        label = [sample['label']]
        age = [sample['age']]
        sex = sample['sex']
        if sex == 'F':
            sex = [1]
        else:
            sex = [0]
        Orientation = sample['mmse']['Orientation']
        immediateMemory = sample['mmse']['immediateMemory']
        Attention_and_calculation = sample['mmse']['Attention_and_calculation']
        lateMemory = sample['mmse']['lateMemory']
        naming = sample['mmse']['naming']
        Read_and_obey = sample['mmse']['Read_and_obey']
        others = sample['mmse']['others']
        ccb = sample['mmse']['Orientation']
        cdr = sample['cdr']['cdr']
        MODHACH_domain1 = sample['MODHACH']['domain1']
        NEUROBA_domain1 = sample['NEUROBA']['domain1']
        NEUROBA_domain2 = sample['NEUROBA']['domain2']
        NPIQ_domain1 = sample['NPIQ']['domain1']
        NPIQ_domain2 = sample['NPIQ']['domain2']
        ADAS_domain1 = sample['ADAS']['domain1']

        
        original_list = [age,sex,Orientation,immediateMemory,Attention_and_calculation,lateMemory,
                          naming,Read_and_obey,others,ccb,cdr,MODHACH_domain1,NEUROBA_domain1,NEUROBA_domain2,NPIQ_domain1,NPIQ_domain2,ADAS_domain1]
        alldatalist = []
        for d in original_list:
            alldatalist.extend(d)
        dataall[index] = alldatalist
    return dataall



























