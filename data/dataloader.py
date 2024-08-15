import re
import os
import random
import logging
import jieba
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder


__all__ = ['MMDataLoader','TextDataLoader','ImageDataLoader']
logger = logging.getLogger('')


def preprocess_text(sen):
    # Removing html tags
    sentence = re.sub(r'<[^>]+>', '', sen)
    # Remove punctuations and numbers
    sentence = sentence.strip()
    # Single character removal
    sentence = re.sub(r'[^\u4e00-\u9fa5]', ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

# Data Aug
def get_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ]
    )

class MMDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 256
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').fillna("")
        else:
            logger.info('数据集无效')
            return
        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Weibo17','Weibo21']:
            tweet_id, image_name, text, label = self.df.iloc[index].values
            img_path = self.args.data_dir +'/'+ self.args.dataset +'/new_images/' + image_name
            text = preprocess_text(text)
        text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
        original_shape = (1, 3, 224, 224)
        zero_img_inputs = torch.zeros(original_shape)
        try:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                image = Image.open(os.path.join(img_path)).convert("RGB")
                image = self.transforms(image)
                img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
            else:
                img_inputs = zero_img_inputs
        except OSError as e:
            print(f"Error loading image {img_path}: {e}")
            img_inputs = zero_img_inputs  
        return img_inputs, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
        
        
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:  
        return [] 
    return torch.utils.data.dataloader.default_collate(batch)

def MMDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21']:
        train_set = MMDataset(args, mode='train')
        valid_set = MMDataset(args, mode='val')
        test_set = MMDataset(args, mode='test')
    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    return train_loader, valid_loader, test_loader




class TextDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 256
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').fillna("")
        else:
            logger.info('数据集无效')
            return
        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Weibo17','Weibo21']:
            tweet_id, image_name, text, label = self.df.iloc[index].values
            text = preprocess_text(text)
        text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
        return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
        
def TextDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21']:
        train_set = TextDataset(args, mode='train')
        valid_set = TextDataset(args, mode='val')
        test_set = TextDataset(args, mode='test')
    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    return train_loader, valid_loader, test_loader


class TextDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 256
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').fillna("")
        else:
            logger.info('数据集无效')
            return
        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Weibo17','Weibo21']:
            tweet_id, image_name, text, label = self.df.iloc[index].values
            text = preprocess_text(text)
        text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
        return text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], label
        
def TextDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21']:
        train_set = TextDataset(args, mode='train')
        valid_set = TextDataset(args, mode='val')
        test_set = TextDataset(args, mode='test')
    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    return train_loader, valid_loader, test_loader



class ImageDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.transforms = get_transforms()
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').fillna("")
        else:
            logger.info('数据集无效')
            return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Weibo17','Weibo21']:
            tweet_id, image_name, text, label = self.df.iloc[index].values
            img_path = self.args.data_dir +'/'+ self.args.dataset +'/new_images/' + image_name
        
        original_shape = (1, 3, 224, 224)
        zero_img_inputs = torch.zeros(original_shape)
        try:
            if os.path.exists(img_path) and os.path.isfile(img_path):
                image = Image.open(os.path.join(img_path)).convert("RGB")
                image = self.transforms(image)
                img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
            else:
                img_inputs = zero_img_inputs
        except OSError as e:
            print(f"Error loading image {img_path}: {e}")
            img_inputs = zero_img_inputs  
        return img_inputs, label
    

        
def ImageDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21']:
        train_set = ImageDataset(args, mode='train')
        valid_set = ImageDataset(args, mode='val')
        test_set = ImageDataset(args, mode='test')
    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
    return train_loader, valid_loader, test_loader