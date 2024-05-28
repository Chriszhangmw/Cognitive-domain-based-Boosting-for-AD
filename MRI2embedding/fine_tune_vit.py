from torchvision.transforms import (CenterCrop,
                    Compose,
                    Normalize,
                    RandomHorizontalFlip,
                    RandomResizedCrop,
                    Resize,
                    ToTensor)
from transformers import ViTImageProcessor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import AutoImageProcessor
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_metric
import numpy as np
import torch
import pickle
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"



def collate_fn(examples):
    pixel_values = torch.stack([example["img"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

metric = load_metric("accuracy")
id2label = {0:'CN',1:'MCI',2:'AD'}
label2id = {'CN':0,'MCI':1,'AD':2}

model_checkpoint = "./pretrain_model" # pre-trained model from which to fine-tune
batch_size = 8 # batch size for training and evaluation
image_processor  = AutoImageProcessor.from_pretrained(model_checkpoint)

class ADDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_path = self.data_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {'img': image, 'label': label}
    def set_transform(self, transform):
        self.transform = transform


def get_data_from_kpl(kpl_path,label_path):
    # 读取.pkl文件
    with open(kpl_path, 'rb') as f:
        data_info = pickle.load(f)
    
    label_metadata = pd.read_csv(label_path)
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
    
    datas = []
    labels = []
    for k,v in data_info.items():
        label = label_dict[k]
        top_image_paths_list1 = v['top_image_paths_list1']
        top_image_paths_list2 = v['top_image_paths_list2']
        top_image_paths_list3 = v['top_image_paths_list3']
        for p in top_image_paths_list1:
            p = str(p)
            p = p.replace('\\', '/')
            datas.append(p)
            labels.append(label)
        for p in top_image_paths_list2:
            p = str(p)
            p = p.replace('\\', '/')
            datas.append(p)
            labels.append(label)
        for p in top_image_paths_list3:
            p = str(p)
            p = p.replace('\\', '/')
            datas.append(p)
            labels.append(label)
    print('The total fine tune images number is : ',len(datas),len(labels))
    print('\n')
    print('---------------------------------------------------------------------')
    print('\n')
    print('Start train-test splitting processing')
    X_train, X_val, y_train, y_val = train_test_split(datas, labels, test_size=0.1, random_state=2024)
    print('\n')
    print('---------------------------------------------------------------------')
    print('\n')
    print('The total training number is : ',len(X_train),len(y_train))
    print('The total validating number is : ',len(X_val),len(y_val))
    return X_train, X_val, y_train, y_val
        
    
def generate_dataset(kpl_path,label_path):
    X_train, X_val, y_train, y_val = get_data_from_kpl(kpl_path,label_path)
    # 创建Dataset对象
    dataset_train = ADDataset(data_paths=X_train, labels=y_train)
    dataset_val = ADDataset(data_paths=X_val, labels=y_val)
    
    processor = ViTImageProcessor.from_pretrained(model_checkpoint)
    image_mean = processor.image_mean
    image_std = processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )
    dataset_train.set_transform(_train_transforms)
    dataset_val.set_transform(_val_transforms)
    
    return dataset_train,dataset_val



def build_model(dataset_train,dataset_val,num_train_epochs=20,learning_rate=5e-5):
    model = AutoModelForImageClassification.from_pretrained(
        model_checkpoint, 
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"./model/{model_name}-finetuned-adani",
        remove_unused_columns=False,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    return trainer
    


def main():
    #device = torch.device('gpu')
    #print('device info: ',device)
    kpl_path = './datasets/data_info_images_path.pkl'
    label_path = './datasets/label_meta.csv'
    dataset_train,dataset_val = generate_dataset(kpl_path,label_path)
    trainer = build_model(dataset_train,dataset_val)
    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    # some nice to haves:
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()





