import os
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from pkg.NormName import norm_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_img_feature(dataset_name, batch_size, encode_model):
  
    def embed_img_batch(batch_imgs, processor, model):
        
        inputs = processor(batch_imgs, return_tensors='pt')
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    print(f"Using device: {device}")
    
    dataset = load_dataset(dataset_name)
    
    processor = AutoImageProcessor.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model).to(device)
    
    embeddings = []
    labels = []

    for split in ['train', 'test']:
        imgs = [example['img'] for example in dataset[split]]
        if dataset_name == 'cifar100':
            batch_labels = [example['fine_label'] for example in dataset[split]]
        else:
            batch_labels = [example['label'] for example in dataset[split]]
        for i in tqdm(range(0, len(imgs), batch_size)):
            batch_imgs = imgs[i:i + batch_size]
            batch_embeddings = embed_img_batch(batch_imgs, processor, model)
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels[i:i + batch_size])

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    encode_model_filename = norm_name(encode_model)
    dataset_dirname = norm_name(dataset_name)
    os.makedirs(f'Data/{dataset_dirname}', exist_ok=True)
    np.save(f'Data/{dataset_dirname}/{encode_model_filename}.npy', embeddings)
    np.save(f'Data/{dataset_dirname}/labels.npy', labels)
    

    print(f"Embeddings saved! Shape: {embeddings.shape}")
    print(f"Labels saved! Shape: {labels.shape}")
    return encode_model_filename, dataset_dirname
    

def extract_text_feature(dataset_name, batch_size, encode_model):

    def embed_text_batch(batch_texts, tokenizer, model):
        
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
      
    print(f"Using device: {device}")
    
    dataset = load_dataset(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model).to(device)
    
    embeddings = []
    labels = []

    for split in ['train', 'test']:
        texts = [example['text'] for example in dataset[split]]
        batch_labels = [example['label'] for example in dataset[split]]
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embed_text_batch(batch_texts, tokenizer, model)
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels[i:i + batch_size])

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    encode_model_filename = norm_name(encode_model)
    dataset_dirname = norm_name(dataset_name)
    os.makedirs(f'Data/{dataset_dirname}', exist_ok=True)
    np.save(f'Data/{dataset_dirname}/{encode_model_filename}.npy', embeddings)
    np.save(f'Data/{dataset_dirname}/labels.npy', labels)

    print(f"Embeddings saved! Shape: {embeddings.shape}")
    print(f"Labels saved! Shape: {labels.shape}")
    return encode_model_filename, dataset_dirname


def extract_clothing_img(data_loader, encode_model):
  
    def embed_img_batch(batch_imgs, processor, model):
        
        inputs = processor(batch_imgs, return_tensors='pt')
        
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    print(f"Using device: {device}")
    
    processor = AutoImageProcessor.from_pretrained(encode_model)
    processor.do_rescale = False
    model = AutoModel.from_pretrained(encode_model).to(device)
    
    embeddings = []
    clean_labels = []
    noise_labels = []

    for batch_imgs, batch_clean_labels, batch_noise_labels in tqdm(data_loader):
        batch_imgs = [img.to(device) for img in batch_imgs]
        
        batch_embeddings = embed_img_batch(batch_imgs, processor, model)
        embeddings.append(batch_embeddings)
        clean_labels.extend(batch_clean_labels)
        noise_labels.extend(batch_noise_labels)
        # break

    embeddings = np.vstack(embeddings)
    clean_labels = np.array(clean_labels)
    noise_labels = np.array(noise_labels)
    
    print('Noise:', np.sum(clean_labels != noise_labels))

    os.makedirs(f'Data/clothing', exist_ok=True)
    np.save(f'Data/clothing/embeddings.npy', embeddings)
    np.save(f'Data/clothing/clean_labels.npy', clean_labels)
    np.save(f'Data/clothing/noise_labels.npy', noise_labels)
    

    print(f"Embeddings saved! Shape: {embeddings.shape}")
    print(f"Clean labels saved! Shape: {clean_labels.shape}")
    print(f"Noise labels saved! Shape: {noise_labels.shape}")