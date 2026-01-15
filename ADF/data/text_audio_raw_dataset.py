import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from .audio_dataset import Dataset_meld_audio
from .data_loader_word import wordMELDDataset


class TA_raw_MELDDataset(Dataset):
    def __init__(
        self,
        text_npz: str,
        audio_npz: str,
        video_npz: str,
        modalities,
        split: str = "train",
        feature_type: str = "pooled_features",
        text_path: str = "data/MELD.Raw/train_raw.csv",
        audio_csv_path: str = None,
        audio_data_path: str = None,
        audio_model_type: str = "Whisper",
    ):
        self.modalities = modalities
        self.split = split
        self.feature_type = feature_type
        self.text_path = text_path

        self.text_data = self._load_text_features()
        self.audio_data = None
        if "A" in modalities:
            if audio_csv_path is None or audio_data_path is None:
                raise ValueError("audio_csv_path and audio_data_path are required for raw audio input.")
            self.audio_data = Dataset_meld_audio(
                csv_path=audio_csv_path,
                audio_directory=audio_data_path,
                max_length=96000,
                cache_dir=None,
                model_type=audio_model_type,
            )
        self.video_data = self._load_features(video_npz) if "V" in modalities else None

        sample_names_sets = []
        if "T" in modalities:
            sample_names_sets.append(set(self.text_data["sample_names"]))
        if "A" in modalities:
            sample_names_sets.append(set(self.audio_data.sample_names))
        if "V" in modalities:
            sample_names_sets.append(set(self.video_data["sample_names"]))

        self.sample_names = sorted(list(set.intersection(*sample_names_sets)))

        if "T" in modalities:
            indices = [self.text_data["sample_names"].index(name) for name in self.sample_names]
            self.text_input_ids = self.text_data["input_ids"][indices]
            self.text_attention_mask = self.text_data["attention_mask"][indices]
            self.text_target_start_pos = self.text_data["target_start_pos"][indices]
            self.text_target_end_pos = self.text_data["target_end_pos"][indices]
            self.labels = self.text_data["labels"][indices]

        if "A" in modalities:
            audio_lookup = {name: idx for idx, name in enumerate(self.audio_data.sample_names)}
            self.audio_indices = [audio_lookup[name] for name in self.sample_names]

        if "V" in modalities:
            indices = [np.where(self.video_data["sample_names"] == name)[0][0] for name in self.sample_names]
            self.video_features = self.video_data["sequence_features"][indices]

    def _load_features(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        features = {key: data[key] for key in data}
        if self.feature_type not in features:
            raise KeyError(
                f"Feature type '{self.feature_type}' not found in {npz_path}. "
                f"Available keys: {list(features.keys())}"
            )
        if "sample_names" not in features:
            raise KeyError(f"'sample_names' not found in {npz_path}")
        return features

    def _load_text_features(self, context_len=6, max_seq_length=196, use_all_context=False, add_speaker=True):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        dataset = wordMELDDataset(
            data_path=self.text_path,
            tokenizer=tokenizer,
            context_len=context_len,
            max_seq_length=max_seq_length,
            use_all_context=use_all_context,
            add_speaker=add_speaker,
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
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "target_start_pos": torch.tensor(target_start_pos_list, dtype=torch.long),
            "target_end_pos": torch.tensor(target_end_pos_list, dtype=torch.long),
            "labels": torch.stack(labels_list),
        }

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample = {}
        if "T" in self.modalities:
            sample["text"] = {
                "input_ids": self.text_input_ids[idx],
                "attention_mask": self.text_attention_mask[idx],
                "target_start_pos": self.text_target_start_pos[idx],
                "target_end_pos": self.text_target_end_pos[idx],
            }
        if "A" in self.modalities:
            audio_item = self.audio_data[self.audio_indices[idx]]
            sample["audio"] = {
                "input_values": audio_item["input_values"],
                "attention_mask": audio_item["attention_mask"],
            }
        if "V" in self.modalities:
            sample["video"] = torch.tensor(self.video_features[idx], dtype=torch.float32)
        sample["label"] = self.labels[idx]
        sample["sample_name"] = self.sample_names[idx]
        return sample
