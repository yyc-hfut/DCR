import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, AutoModelForSpeechSeq2Seq, AutoProcessor                              

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Dataset_meld_audio(Dataset):
    def __init__(self, csv_path, audio_directory, max_length=96000, cache_dir=None, model_type=None):
        df = pd.read_csv(csv_path)
        self.df = df
        self.targets_emotion = df['Emotion']
        self.audio_file_paths = []
        self.sample_names = []         
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        for i in range(len(df)):
            dialogue_id = df['Dialogue_ID'][i]
            utterance_id = df['Utterance_ID'][i]
            file_name = f"dia{dialogue_id}_utt{utterance_id}.wav"
            file_path = os.path.join(audio_directory, file_name)
            if os.path.exists(file_path):
                self.audio_file_paths.append(file_path)
                self.sample_names.append(f"dia{dialogue_id}_utt{utterance_id}")                      
            else:
                print(f"Warning: Audio file {file_path} not found.")
                self.audio_file_paths.append(None)
                self.sample_names.append(f"dia{dialogue_id}_utt{utterance_id}")                 
        
        self.sampling_rate = 16000
        self.max_length = max_length
        if model_type == "WavLM":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
        elif model_type == "Data2Vec":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/data2vec-audio-large-960h")
        elif model_type == "Wav2Vec2":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
        elif model_type == "Hubert":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        elif model_type == 'Whisper':
            self.processor = AutoProcessor.from_pretrained("openai/whisper-small")

        self.model_type = model_type

        self.emotion_map = {
            "neutral": 2, "anger": 3, "joy": 0, "sadness": 1,
            "fear": 6, "disgust": 5, "surprise": 4
        }
                     
        self.cached_data = self._load_data()
        for i in range(len(self.cached_data)):
            if self.cached_data[i]['input_values'].shape[0] != 128:
                self.cached_data[i]['input_values'] = self.cached_data[i-1]['input_values']
                self.cached_data[i]['attention_mask'] = self.cached_data[i-1]['attention_mask']
                

    
    def _is_cache_complete(self):
        if not self.cache_dir:
            return False
        expected_files = len(self.audio_file_paths)
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pt')]
        return len(cache_files) == expected_files and all(
            os.path.exists(os.path.join(self.cache_dir, f"sample_{i}.pt")) for i in range(expected_files)
        )
    
    def _load_cached_data(self):
        cached_data = []
        for i in tqdm(range(len(self.audio_file_paths)), desc="Loading cached data"):
            cache_file = os.path.join(self.cache_dir, f"sample_{i}.pt")
            data = torch.load(cache_file, weights_only=False)
            cached_data.append(data)
        return cached_data
    
    def _load_or_cache_data(self):
        cached_data = []
        for i, file_path in enumerate(tqdm(self.audio_file_paths, desc="Caching data")):
            cache_file = os.path.join(self.cache_dir, f"sample_{i}.pt") if self.cache_dir else None
            
            if cache_file and os.path.exists(cache_file):
                data = torch.load(cache_file, weights_only=False)
            else:
                if file_path is None:
                    input_values = torch.zeros(self.max_length)
                    attention_mask = torch.zeros(self.max_length, dtype=torch.long)
                else:
                    sound, sr = torchaudio.load(file_path)
                    sound_data = torch.mean(sound, dim=0, keepdim=False)
                    if sr != self.sampling_rate:
                        sound_data = torchaudio.transforms.Resample(sr, self.sampling_rate)(sound_data)
                    processed = self.processor(
                        sound_data.numpy(),
                        sampling_rate=self.sampling_rate,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True
                    )
                    input_values = processed['input_values'].squeeze(0)
                    attention_mask = processed['attention_mask'].squeeze(0) if 'attention_mask' in processed else torch.ones_like(input_values, dtype=torch.long)
                
                data = {"input_values": input_values, "attention_mask": attention_mask}
                if cache_file:
                    torch.save(data, cache_file)
            cached_data.append(data)
        return cached_data
    
    def _load_data(self):
        data_list = []
        for file_path in tqdm(self.audio_file_paths, desc="Generating data"):
            if file_path is None:
                input_values = torch.zeros(self.max_length)
                attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            else:
                sound, sr = torchaudio.load(file_path)
                sound_data = torch.mean(sound, dim=0, keepdim=False)
                if sr != self.sampling_rate:
                    sound_data = torchaudio.transforms.Resample(sr, self.sampling_rate)(sound_data)
                processed = self.processor(
                    sound_data.numpy(),
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True
                )
                if self.model_type == "Whisper":
                    input_values = processed['input_features'].squeeze(0)
                    attention_mask = processed['attention_mask'].squeeze(0) if 'attention_mask' in processed else torch.ones_like(input_values, dtype=torch.long)
                else:
                    input_values = processed['input_values'].squeeze(0)
                    attention_mask = processed['attention_mask'].squeeze(0) if 'attention_mask' in processed else torch.ones_like(input_values, dtype=torch.long)
            
            data = {"input_values": input_values, "attention_mask": attention_mask}
            data_list.append(data)
        return data_list
    
    def __getitem__(self, index):
        emotion_label = self.emotion_map.get(self.targets_emotion[index], 2)
        data = self.cached_data[index]
        return {
            "input_values": data["input_values"],
            "attention_mask": data["attention_mask"],
            "targets": torch.tensor(emotion_label, dtype=torch.long),
            "sample_name": self.sample_names[index]         
        }
    
    def __len__(self):
        return len(self.targets_emotion)

def data_loader_meld_audio(batch_size, max_seq_length, model_type, num_workers=4, cache_dir="/path/to/data/MELD/cache", ):
    base_path = '/path/to/data/MELD'
    audio_base_path = '/path/to/data/MELD'
    
    splits = {
        'train': ('train_sent_emo.csv', 'train_A/train'),
        'test': ('test_sent_emo.csv', 'test_A/test')
    }
    
    train_data = Dataset_meld_audio(
        csv_path=os.path.join(base_path, splits['train'][0]),
        audio_directory=os.path.join(audio_base_path, splits['train'][1]),
        max_length=max_seq_length,
        cache_dir=os.path.join(cache_dir, "train"),
        model_type=model_type
    )
    test_data = Dataset_meld_audio(
        csv_path=os.path.join(base_path, splits['test'][0]),
        audio_directory=os.path.join(audio_base_path, splits['test'][1]),
        max_length=max_seq_length,
        cache_dir=os.path.join(cache_dir, "test"),
        model_type= model_type
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader

if __name__ == "__main__":
          
    batch_size = 32
    train_loader, test_loader = data_loader_meld_audio(batch_size, )
    for batch in train_loader:
        print(batch['input_values'].shape, batch['attention_mask'].shape, batch['targets'].shape)
        print(batch['targets'])          
        break