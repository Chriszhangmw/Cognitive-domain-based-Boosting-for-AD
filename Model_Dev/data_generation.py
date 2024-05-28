
from non_images import generate_20_sub_datasets,generate_all_non_image
import pickle
import numpy as np
from tools import get_polynomialfeatures

all_data = {}
# dataall = generate_all_non_image()
data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,rdata1, rdata2, rdata3, rdata4, rdata5, rdata6, rdata7, rdata8, rdata9, rdata10, rdata11, rdata12, rdata13, rdata14, rdata15, rdata16, rdata17, rdata18, rdata19, rdata20, rdata21, rdata22, rdata23, rdata24, rdata25, rdata26, rdata27, rdata28, rdata29, rdata30, rdata31, rdata32, rdata33, rdata34, rdata35, rdata36, rdata37, rdata38, rdata39, rdata40, rdata41, rdata42, rdata43, rdata44, rdata45, rdata46, rdata47, rdata48, rdata49, rdata50 = generate_20_sub_datasets()

# 从.pkl文件加载数据
with open('./datasets/top10/res_last_hidden_states1.pkl', 'rb') as f:
    iamge1 = pickle.load(f)

with open('./datasets/top10/res_last_hidden_states2.pkl', 'rb') as f:
    image2 = pickle.load(f)

with open('./datasets/top10/res_last_hidden_states3.pkl', 'rb') as f:
    image3 = pickle.load(f)

print('key number : ',len(list(set(iamge1.keys()))))


# for k,v in iamge1.items():
#     onedata = {}
#     onedata['image1'] = v[-1]
#     onedata['image2'] = image2[k][-1]
#     onedata['image3'] = image3[k][-1]
#     onedata['dataall'] = dataall[k]
#     all_data[k] = onedata
# import pickle

# # 将字典保存为 Pickle 文件
# with open('./datasets/top10/model0_ml.pkl', 'wb') as pickle_file:
#     pickle.dump(all_data, pickle_file)
    
    
for k,v in iamge1.items():
    onedata = {}
    onedata['image1'] = v[-1]
    onedata['image2'] = image2[k][-1]
    onedata['image3'] = image3[k][-1]
    onedata['data1'] = data1[k]
    onedata['data2'] = data2[k]
    onedata['data3'] = data3[k]
    onedata['data4'] = data4[k]
    onedata['data5'] = data5[k]
    onedata['data6'] = data6[k]
    onedata['data7'] = data7[k]
    onedata['data8'] = data8[k]
    onedata['data9'] = data9[k]
    onedata['data10'] = data10[k]
    onedata['data11'] = data11[k]
    onedata['data12'] = data12[k]
    onedata['data13'] = data13[k]
    onedata['data14'] = rdata1[k]
    onedata['data15'] = rdata2[k]
    onedata['data16'] = rdata3[k]
    onedata['data17'] = rdata4[k]
    onedata['data18'] = rdata5[k]
    onedata['data19'] = rdata6[k]
    onedata['data20'] = rdata7[k]
    onedata['data21'] = rdata8[k]
    onedata['data22'] = rdata9[k]
    onedata['data23'] = rdata10[k]
    onedata['data24'] = rdata11[k]
    onedata['data25'] = rdata12[k]
    onedata['data26'] = rdata13[k]
    onedata['data27'] = rdata14[k]
    onedata['data28'] = rdata15[k]
    onedata['data29'] = rdata16[k]
    onedata['data30'] = rdata17[k]
    onedata['data31'] = rdata18[k]
    onedata['data32'] = rdata19[k]
    onedata['data33'] = rdata20[k]
    onedata['data34'] = rdata21[k]
    onedata['data35'] = rdata22[k]
    onedata['data36'] = rdata23[k]
    onedata['data37'] = rdata24[k]
    onedata['data38'] = rdata25[k]
    onedata['data39'] = rdata26[k]
    onedata['data40'] = rdata27[k]
    onedata['data41'] = rdata28[k]
    onedata['data42'] = rdata29[k]
    onedata['data43'] = rdata30[k]
    onedata['data44'] = rdata31[k]
    onedata['data45'] = rdata32[k]
    onedata['data46'] = rdata33[k]
    onedata['data47'] = rdata34[k]
    onedata['data48'] = rdata35[k]
    onedata['data49'] = rdata36[k]
    onedata['data50'] = rdata37[k]
    onedata['data51'] = rdata38[k]
    onedata['data52'] = rdata39[k]
    onedata['data53'] = rdata40[k]
    onedata['data54'] = rdata41[k]
    onedata['data55'] = rdata42[k]
    onedata['data56'] = rdata43[k]
    onedata['data57'] = rdata44[k]
    onedata['data58'] = rdata45[k]
    onedata['data59'] = rdata46[k]
    onedata['data60'] = rdata47[k]
    onedata['data61'] = rdata48[k]
    onedata['data62'] = rdata49[k]
    onedata['data63'] = rdata50[k]


    onedata['data1_dense'] = get_polynomialfeatures(data1[k])
    onedata['data2_dense'] = get_polynomialfeatures(data2[k])
    onedata['data3_dense'] = get_polynomialfeatures(data3[k])
    onedata['data4_dense'] = get_polynomialfeatures(data4[k])
    onedata['data5_dense'] = get_polynomialfeatures(data5[k])
    onedata['data6_dense'] = get_polynomialfeatures(data6[k])
    onedata['data7_dense'] = get_polynomialfeatures(data7[k])
    onedata['data8_dense'] = get_polynomialfeatures(data8[k])
    onedata['data9_dense'] = get_polynomialfeatures(data9[k])
    onedata['data10_dense'] = get_polynomialfeatures(data10[k])
    onedata['data11_dense'] = get_polynomialfeatures(data11[k])
    onedata['data12_dense'] = get_polynomialfeatures(data12[k])
    onedata['data13_dense'] = get_polynomialfeatures(data13[k])
    onedata['data14_dense'] = get_polynomialfeatures(rdata1[k])
    onedata['data15_dense'] = get_polynomialfeatures(rdata2[k])
    onedata['data16_dense'] = get_polynomialfeatures(rdata3[k])
    onedata['data17_dense'] = get_polynomialfeatures(rdata4[k])
    onedata['data18_dense'] = get_polynomialfeatures(rdata5[k])
    onedata['data19_dense'] = get_polynomialfeatures(rdata6[k])
    onedata['data20_dense'] = get_polynomialfeatures(rdata7[k])
    onedata['data21_dense'] = get_polynomialfeatures(rdata8[k])
    onedata['data22_dense'] = get_polynomialfeatures(rdata9[k])
    onedata['data23_dense'] = get_polynomialfeatures(rdata10[k])
    onedata['data24_dense'] = get_polynomialfeatures(rdata11[k])
    onedata['data25_dense'] = get_polynomialfeatures(rdata12[k])
    onedata['data26_dense'] = get_polynomialfeatures(rdata13[k])
    onedata['data27_dense'] = get_polynomialfeatures(rdata14[k])
    onedata['data28_dense'] = get_polynomialfeatures(rdata15[k])
    onedata['data29_dense'] = get_polynomialfeatures(rdata16[k])
    onedata['data30_dense'] = get_polynomialfeatures(rdata17[k])
    onedata['data31_dense'] = get_polynomialfeatures(rdata18[k])
    onedata['data32_dense'] = get_polynomialfeatures(rdata19[k])
    onedata['data33_dense'] = get_polynomialfeatures(rdata20[k])
    onedata['data34_dense'] = get_polynomialfeatures(rdata21[k])
    onedata['data35_dense'] = get_polynomialfeatures(rdata22[k])
    onedata['data36_dense'] = get_polynomialfeatures(rdata23[k])
    onedata['data37_dense'] = get_polynomialfeatures(rdata24[k])
    onedata['data38_dense'] = get_polynomialfeatures(rdata25[k])
    onedata['data39_dense'] = get_polynomialfeatures(rdata26[k])
    onedata['data40_dense'] = get_polynomialfeatures(rdata27[k])
    onedata['data41_dense'] = get_polynomialfeatures(rdata28[k])
    onedata['data42_dense'] = get_polynomialfeatures(rdata29[k])
    onedata['data43_dense'] = get_polynomialfeatures(rdata30[k])
    onedata['data44_dense'] = get_polynomialfeatures(rdata31[k])
    onedata['data45_dense'] = get_polynomialfeatures(rdata32[k])
    onedata['data46_dense'] = get_polynomialfeatures(rdata33[k])
    onedata['data47_dense'] = get_polynomialfeatures(rdata34[k])
    onedata['data48_dense'] = get_polynomialfeatures(rdata35[k])
    onedata['data49_dense'] = get_polynomialfeatures(rdata36[k])
    onedata['data50_dense'] = get_polynomialfeatures(rdata37[k])
    onedata['data51_dense'] = get_polynomialfeatures(rdata38[k])
    onedata['data52_dense'] = get_polynomialfeatures(rdata39[k])
    onedata['data53_dense'] = get_polynomialfeatures(rdata40[k])
    onedata['data54_dense'] = get_polynomialfeatures(rdata41[k])
    onedata['data55_dense'] = get_polynomialfeatures(rdata42[k])
    onedata['data56_dense'] = get_polynomialfeatures(rdata43[k])
    onedata['data57_dense'] = get_polynomialfeatures(rdata44[k])
    onedata['data58_dense'] = get_polynomialfeatures(rdata45[k])
    onedata['data59_dense'] = get_polynomialfeatures(rdata46[k])
    onedata['data60_dense'] = get_polynomialfeatures(rdata47[k])
    onedata['data61_dense'] = get_polynomialfeatures(rdata48[k])
    onedata['data62_dense'] = get_polynomialfeatures(rdata49[k])
    onedata['data63_dense'] = get_polynomialfeatures(rdata50[k])
    
    all_data[k] = onedata
    
    
import pickle

# 将字典保存为 Pickle 文件
with open('./datasets/top10/all_data_last_image.pkl', 'wb') as pickle_file:
    pickle.dump(all_data, pickle_file)







