import os
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoImageProcessor, AutoModel
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_img_embedding(dataset_name, batch_size, encode_model):
  
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

    encode_model_filename = encode_model.replace('/', '_').replace('-', '_')
    dataset_dirname = dataset_name.replace('/', '_').replace('-', '_')
    os.makedirs(f'Data/{dataset_dirname}', exist_ok=True)
    np.save(f'Data/{dataset_dirname}/{encode_model_filename}.npy', embeddings)
    np.save(f'Data/{dataset_dirname}/labels.npy', labels)

    print(f"Embeddings saved! Shape: {embeddings.shape}")
    print(f"Labels saved! Shape: {labels.shape}")
    return encode_model_filename, dataset_dirname
    

def extract_text_embedding(dataset_name, batch_size, encode_model):

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

    encode_model_filename = encode_model.replace('/', '_').replace('-', '_')
    dataset_dirname = dataset_name.replace('/', '_').replace('-', '_')
    os.makedirs(f'Data/{dataset_dirname}', exist_ok=True)
    np.save(f'Data/{dataset_dirname}/{encode_model_filename}.npy', embeddings)
    np.save(f'Data/{dataset_dirname}/labels.npy', labels)

    print(f"Embeddings saved! Shape: {embeddings.shape}")
    print(f"Labels saved! Shape: {labels.shape}")
    return encode_model_filename, dataset_dirname

def extract_audio_embedding(dataset_name, batch_size, encode_model):

    def embed_audio_batch(batch_audio, processor, model):

        inputs = processor(batch_audio, return_tensors = 'pt', padding=True)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
    
    print(f"Using device: {device}")

    if dataset == "google/speech_commands":
        dataset = load_dataset(dataset_name, "v0.02", trust_remote_code=True)
    else:
        dataset = load_dataset(dataset_name)


    processor = AutoProcessor.from_pretrained(encode_model)
    model = AutoModel.from_pretrained(encode_model).to(device)

    if dataset == "Zahra99/IEMOCAP_Audio":
        data_splits = ['session1', 'session2', 'session3', 'session4', 'session5']
    else:   
        data_splits = ['train', 'test']

    for split in data_splits:
        audios = [example['audio']["array"] for example in dataset[split]]
    
       
        batch_labels = [example['label'] for example in dataset[split]]
        for i in tqdm(range(0, len(audios), batch_size)):
            embeddings = []
            labels = []
            batch_imgs = audios[i:i + batch_size]
            batch_embeddings = embed_audio_batch(batch_imgs, processor, model)
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels[i:i + batch_size])
            embeddings = np.vstack(embeddings)
            labels = np.array(labels)
        
            encode_model_filename = encode_model.replace('/', '_').replace('-', '_') + str(i)
            dataset_dirname = dataset_name.replace('/', '_').replace('-', '_')
            os.makedirs(f'Data/{dataset_dirname}', exist_ok=True)
            np.save(f'Data/{dataset_dirname}/{encode_model_filename}.npy', embeddings)
            label_name = "labels" + str(i)
            np.save(f'Data/{dataset_dirname}/{label_name}.npy', labels)
            del embeddings
            del labels
            del batch_embeddings
       

            # print(f"Embeddings saved! Shape: {embeddings.shape}")
            # print(f"Labels saved! Shape: {labels.shape}")
    return encode_model_filename, dataset_dirname