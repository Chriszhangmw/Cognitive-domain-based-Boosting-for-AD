from PIL import Image
import requests
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import ViTFeatureExtractor, ViTModel
import torch
repo_name = "./checkpoint-840/"
feature_extractor = ViTFeatureExtractor.from_pretrained(repo_name)
model = ViTModel.from_pretrained(repo_name)

def generate_embeddings_from_finetuned_model(image):
  inputs = feature_extractor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  last_hidden_states = outputs.last_hidden_state.tolist()
  pooler_output = outputs.pooler_output.tolist()
  return last_hidden_states,pooler_output


import pickle
from tqdm import tqdm
with open('./datasets/top70/data_info_images_path_7top_to7top_2.pkl', 'rb') as f:
    data_info = pickle.load(f)

res_last_hidden_states1 = {}
res_pooler_output1 = {}

res_last_hidden_states2 = {}
res_pooler_output2 = {}

res_last_hidden_states3 = {}
res_pooler_output3 = {}


for k,v in tqdm(data_info.items()):
  top_image_paths_list1 = v['top_image_paths_list1']
  top_image_paths_list2 = v['top_image_paths_list2']
  top_image_paths_list3 = v['top_image_paths_list3']

  last_hidden_states1_list = []
  last_hidden_states2_list = []
  last_hidden_states3_list = []
  pooler_output1_list = []
  pooler_output2_list = []
  pooler_output3_list = []
  for p in top_image_paths_list1:
    p = str(p)
    p = p.replace('\\', '/')
    # p = p.replace('./datasets','../datasets')
    # print(p)
    image = Image.open(p).convert('RGB')
    last_hidden_states1,pooler_output1 = generate_embeddings_from_finetuned_model(image)
    last_hidden_states1_list.append(last_hidden_states1)
    pooler_output1_list.append(pooler_output1)
  res_last_hidden_states1[k] = last_hidden_states1_list
  res_pooler_output1[k] = pooler_output1_list
  for p in top_image_paths_list2:
    p = str(p)
    p = p.replace('\\', '/')
    # p = p.replace('./datasets','../datasets')
    image = Image.open(p).convert('RGB')
    last_hidden_states2,pooler_output2 = generate_embeddings_from_finetuned_model(image)
    last_hidden_states2_list.append(last_hidden_states2)
    pooler_output2_list.append(pooler_output2)
  res_last_hidden_states2[k] = last_hidden_states2_list
  res_pooler_output2[k] = pooler_output2_list
  for p in top_image_paths_list3:
    p = str(p)
    p = p.replace('\\', '/')
    # p = p.replace('./datasets','../datasets')
    image = Image.open(p).convert('RGB')
    last_hidden_states3,pooler_output3 = generate_embeddings_from_finetuned_model(image)
    last_hidden_states3_list.append(last_hidden_states3)
    pooler_output3_list.append(pooler_output3)
  res_last_hidden_states3[k] = last_hidden_states3_list
  res_pooler_output3[k] = pooler_output3_list

import pickle

# 将字典保存为 Pickle 文件
with open('./datasets/top70/res_last_hidden_states1.pkl', 'wb') as pickle_file:
    pickle.dump(res_last_hidden_states1, pickle_file)
with open('./datasets/top70/res_last_hidden_states2.pkl', 'wb') as pickle_file:
    pickle.dump(res_last_hidden_states2, pickle_file)
with open('./datasets/top70/res_last_hidden_states3.pkl', 'wb') as pickle_file:
    pickle.dump(res_last_hidden_states3, pickle_file)

with open('./datasets/top70/res_pooler_output1.pkl', 'wb') as pickle_file:
    pickle.dump(res_pooler_output1, pickle_file)
with open('./datasets/top70/res_pooler_output2.pkl', 'wb') as pickle_file:
    pickle.dump(res_pooler_output2, pickle_file)
with open('./datasets/top70/res_pooler_output3.pkl', 'wb') as pickle_file:
    pickle.dump(res_pooler_output3, pickle_file)

