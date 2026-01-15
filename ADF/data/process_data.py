import os
import numpy as np
from typing import List, Dict, Any

def clean_audio_sample_names(sample_names: List[str]) -> List[str]:
    """
    Remove .wav suffix from audio sample names.
    """
    return [name.replace('.wav', '') for name in sample_names]

def load_features(npz_path: str) -> Dict[str, Any]:
    """
    Load features from .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data}

def get_feature_dim(features: np.ndarray) -> tuple:
    """
    Determine the feature dimension from the first valid feature.
    Returns the shape of the feature (e.g., (768,) for pooled_features, (40, 768) for sequence_features).
    """
    for feat in features:
        if feat is not None and feat.size > 0:
            return feat.shape
    raise ValueError("No valid features found to determine dimension")

def process_video_features(
    video_npz: str,
    audio_sample_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Process video features: align both pooled_features and sequence_features with audio sample names,
    fill missing with next utterance or zeros. Align labels accordingly.
    """
    video_data = load_features(video_npz)
    video_sample_names = video_data['sample_names'].tolist()
    video_pooled_features = video_data.get('pooled_features', np.array([]))
    video_sequence_features = video_data.get('sequence_features', np.array([]))
    
    pooled_feature_dim = get_feature_dim(video_pooled_features) if video_pooled_features.size > 0 else (0,)
    sequence_feature_shape = get_feature_dim(video_sequence_features) if video_sequence_features.size > 0 else (0, 0)
    print(f"Pooled feature dimension: {pooled_feature_dim}")
    print(f"Sequence feature shape: {sequence_feature_shape}")
    
    aligned_pooled_features = []
    aligned_sequence_features = []
    aligned_labels = []
    missing_samples = []
    
    for sample in audio_sample_names:
        if sample in video_sample_names:
            idx = video_sample_names.index(sample)
            if video_pooled_features.size > 0:
                aligned_pooled_features.append(video_pooled_features[idx].flatten())
            else:
                aligned_pooled_features.append(np.zeros(pooled_feature_dim, dtype=np.float32))
            if video_sequence_features.size > 0:
                aligned_sequence_features.append(video_sequence_features[idx])
            else:
                aligned_sequence_features.append(np.zeros(sequence_feature_shape, dtype=np.float32))
            aligned_labels.append(video_data['labels'][idx] if 'labels' in video_data else 0)
        else:
            missing_samples.append(sample)
            try:
                base_name, utt_num = sample.rsplit('_', 1)
                next_utt = f"{base_name}_{int(utt_num)+1}"
            except ValueError:
                print(f"Unexpected sample name format: {sample}, filling with zeros")
                aligned_pooled_features.append(np.zeros(pooled_feature_dim, dtype=np.float32))
                aligned_sequence_features.append(np.zeros(sequence_feature_shape, dtype=np.float32))
                aligned_labels.append(0)
                continue
            
            if next_utt in video_sample_names:
                idx = video_sample_names.index(next_utt)
                if video_pooled_features.size > 0:
                    aligned_pooled_features.append(video_pooled_features[idx].flatten())
                else:
                    aligned_pooled_features.append(np.zeros(pooled_feature_dim, dtype=np.float32))
                if video_sequence_features.size > 0:
                    aligned_sequence_features.append(video_sequence_features[idx])
                else:
                    aligned_sequence_features.append(np.zeros(sequence_feature_shape, dtype=np.float32))
                aligned_labels.append(video_data['labels'][idx] if 'labels' in video_data else 0)
            else:
                print(f"Missing video feature for {sample}, filling with zeros")
                aligned_pooled_features.append(np.zeros(pooled_feature_dim, dtype=np.float32))
                aligned_sequence_features.append(np.zeros(sequence_feature_shape, dtype=np.float32))
                aligned_labels.append(0)
    
    if missing_samples:
        print(f"Missing {len(missing_samples)} video samples: {missing_samples[:5]}...")
    
    aligned_pooled_features = np.array(aligned_pooled_features, dtype=np.float32) if aligned_pooled_features else np.array([])
    aligned_sequence_features = np.array(aligned_sequence_features, dtype=np.float32) if aligned_sequence_features else np.array([])
    aligned_labels = np.array(aligned_labels, dtype=np.int64)
    
    print(f"Aligned pooled features shape: {aligned_pooled_features.shape}")
    print(f"Aligned sequence features shape: {aligned_sequence_features.shape}")
    print(f"Aligned labels shape: {aligned_labels.shape}")
    
    return {
        'sample_names': np.array(audio_sample_names),
        'pooled_features': aligned_pooled_features,
        'sequence_features': aligned_sequence_features,
        'labels': aligned_labels
    }

def save_processed_features(
    audio_data: Dict[str, Any],
    video_data: Dict[str, Any],
    audio_output_dir: str,
    video_output_dir: str,
    split: str
):
    """
    Save processed audio and video features to .npz files in the same directories as input.
    """
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)
    
    audio_output = os.path.join(audio_output_dir, f"{split}_audio_features_processed.npz")
    np.savez(
        audio_output,
        pooled_features=audio_data['pooled_features'],
        labels=audio_data.get('labels', np.array([])),
        sample_names=audio_data['sample_names'],
        sequence_features=audio_data.get('sequence_features', np.array([]))
    )
    
    video_output = os.path.join(video_output_dir, f"{split}_video_features_processed.npz")
    np.savez(
        video_output,
        pooled_features=video_data['pooled_features'],
        labels=video_data['labels'],
        sample_names=video_data['sample_names'],
        sequence_features=video_data['sequence_features']
    )
    print(f"Saved processed features to {audio_output} and {video_output}")

def main():
    base_dir = "/path/to/data/MELD"
    audio_feature_dir = os.path.join(base_dir, "audio_features")
    video_feature_dir = os.path.join(base_dir, "face_features")
    audio_npz_train = os.path.join(audio_feature_dir, "train_audio_features.npz")
    audio_npz_test = os.path.join(audio_feature_dir, "test_audio_features.npz")
    video_npz_train = os.path.join(video_feature_dir, "train_face_features.npz")
    video_npz_test = os.path.join(video_feature_dir, "test_face_features.npz")
    
    print("Processing train split...")
    train_audio_data = load_features(audio_npz_train)
    train_audio_data['sample_names'] = clean_audio_sample_names(train_audio_data['sample_names'].tolist())
    
    train_video_data = process_video_features(
        video_npz_train, 
        train_audio_data['sample_names']
    )
    
    save_processed_features(
        train_audio_data, 
        train_video_data, 
        audio_feature_dir, 
        video_feature_dir, 
        "train"
    )
    
    print("Processing test split...")
    test_audio_data = load_features(audio_npz_test)
    test_audio_data['sample_names'] = clean_audio_sample_names(test_audio_data['sample_names'].tolist())
    
    test_video_data = process_video_features(
        video_npz_test, 
        test_audio_data['sample_names']
    )
    
    save_processed_features(
        test_audio_data, 
        test_video_data, 
        audio_feature_dir, 
        video_feature_dir, 
        "test"
    )

if __name__ == "__main__":
    main()