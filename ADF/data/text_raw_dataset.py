import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from .data_loader_word import wordMELDDataset


class T_raw_MELDDataset(Dataset):
    """
    PyTorch Dataset for MELD multimodal emotion recognition.
    Loads and aligns text, audio, and video features based on sample_names.
    """
    def __init__(
        self,
        text_npz: str,
        audio_npz: str,
        video_npz: str,
        modalities: List[str],
        split: str = 'train',
        feature_type: str = 'pooled_features',
        text_path: str = 'data/MELD.Raw/train_raw.csv'
    ):
        """
        Initialize the dataset.
        
        Args:
            text_npz (str): Path to text features .npz file.
            audio_npz (str): Path to audio features .npz file.
            video_npz (str): Path to video features .npz file.
            modalities (List[str]): List of modalities to include (e.g., ['T', 'A', 'V']).
            split (str): Dataset split ('train' or 'test').
            feature_type (str): Type of feature to load (e.g., 'pooled_features', 'sequence_features').
        """
        self.modalities = modalities
        self.split = split
        self.feature_type = feature_type
        self.text_path = text_path
        
        self.text_data = self.load_text_features()
        self.audio_data = self.load_features(audio_npz, 'audio') if 'A' in modalities else None
        self.video_data = self.load_features(video_npz, 'video') if 'V' in modalities else None
        
        sample_names_sets = []
        if 'T' in modalities:
            sample_names_sets.append(set(self.text_data['sample_names']))
        if 'A' in modalities:
            sample_names_sets.append(set(self.audio_data['sample_names']))
        if 'V' in modalities:
            sample_names_sets.append(set(self.video_data['sample_names']))
        
        self.sample_names = sorted(list(set.intersection(*sample_names_sets)))
        print(f"{split} split: {len(self.sample_names)} samples after alignment")
        
                  
        if 'T' in modalities:
            indices = [self.text_data['sample_names'].index(name) for name in self.sample_names]
            self.text_input_ids = self.text_data['input_ids'][indices]  # (num_samples, max_seq_length)
            self.text_attention_mask = self.text_data['attention_mask'][indices]  # (num_samples, max_seq_length)
            self.text_target_start_pos = self.text_data['target_start_pos'][indices]  # (num_samples,)
            self.text_target_end_pos = self.text_data['target_end_pos'][indices]  # (num_samples,)
            self.labels = self.text_data['labels'][indices]  # (num_samples,)
            print(f"Aligned text input_ids shape: {self.text_input_ids.shape}")
            print(f"Aligned text attention_mask shape: {self.text_attention_mask.shape}")

        if 'A' in modalities:
                              
            indices = [np.where(self.audio_data['sample_names'] == name)[0][0] for name in self.sample_names]
            self.audio_features = self.audio_data['sequence_features'][indices]
            print(f"Aligned audio shape: {self.audio_features.shape}")

        if 'V' in modalities:
                              
            indices = [np.where(self.video_data['sample_names'] == name)[0][0] for name in self.sample_names]
            self.video_features = self.video_data['sequence_features'][indices]
            print(f"Aligned video shape: {self.video_features.shape}")

        print(f"Aligned labels shape: {self.labels.shape}")


    
    def load_features(self, npz_path: str, modality: str) -> Dict[str, Any]:
        """
        Load features from .npz file and validate.
        
        Args:
            npz_path (str): Path to .npz file.
            modality (str): Modality name for logging (e.g., 'text', 'audio', 'video').
        
        Returns:
            Dict containing loaded features.
        """
        try:
            data = np.load(npz_path, allow_pickle=True)
            features = {key: data[key] for key in data}
            
            if self.feature_type not in features:
                raise KeyError(f"Feature type '{self.feature_type}' not found in {npz_path}. Available keys: {list(features.keys())}")
            
            if 'sample_names' not in features:
                raise KeyError(f"'sample_names' not found in {npz_path}")
            print(f"Loaded {modality} features from {npz_path}: {len(features['sample_names'])} samples, "
                  f"{self.feature_type} shape: {features[self.feature_type].shape}")
            
            return features
        except Exception as e:
            print(f"Error loading {modality} features from {npz_path}: {e}")
            raise

    

    def load_text_features(self, context_len=6, max_seq_length=196, use_all_context=False, add_speaker=True):

        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

                             
        dataset = wordMELDDataset(
            data_path=self.text_path,
            tokenizer=tokenizer,
            context_len=context_len,
            max_seq_length=max_seq_length,
            use_all_context=use_all_context,
            add_speaker=add_speaker
        )

              
        sample_names = []
        input_ids_list = []
        attention_mask_list = []
        target_start_pos_list = []
        target_end_pos_list = []
        labels_list = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_names.append(sample["sample_name"])
            input_ids_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            target_start_pos_list.append(sample["target_start_pos"])
            target_end_pos_list.append(sample["target_end_pos"])
            labels_list.append(sample["label"])

               
        return {
            "sample_names": sample_names,
            "input_ids": torch.stack(input_ids_list),  # (num_samples, max_seq_length)
            "attention_mask": torch.stack(attention_mask_list),  # (num_samples, max_seq_length)
            "target_start_pos": torch.tensor(target_start_pos_list, dtype=torch.long),  # (num_samples,)
            "target_end_pos": torch.tensor(target_end_pos_list, dtype=torch.long),  # (num_samples,)
            "labels": torch.stack(labels_list)  # (num_samples,)
        }

    
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample = {}
        if 'T' in self.modalities:
            sample['text'] = {
                'input_ids': self.text_input_ids[idx],  # (max_seq_length,)
                'attention_mask': self.text_attention_mask[idx],  # (max_seq_length,)
                'target_start_pos': self.text_target_start_pos[idx],  # scalar
                'target_end_pos': self.text_target_end_pos[idx]  # scalar
            }
        if 'A' in self.modalities:
            sample['audio'] = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        if 'V' in self.modalities:
            sample['video'] = torch.tensor(self.video_features[idx], dtype=torch.float32)
        sample['label'] = self.labels[idx]
        sample['sample_name'] = self.sample_names[idx]
        return sample

