import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import os
from tqdm import tqdm
from data.feature_dataset import MELDDataset
from baseline_model import MultimodalFusionModel
from training.engine import EMA, train, evaluate, set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR             
import gc
from data.text_raw_dataset import T_raw_MELDDataset
from data.text_audio_raw_dataset import TA_raw_MELDDataset


def prepare_inputs(batch, modalities, device):
    modality_map = {'T': 'text', 'A': 'audio', 'V': 'video'}
    inputs = {}
    for mod in modalities:
        key = modality_map[mod]
        data = batch[key]
        if isinstance(data, dict):
            inputs[mod] = {k: v.to(device) for k, v in data.items()}
        else:
            inputs[mod] = data.to(device)
    return inputs


def collect_predictions(model, data_loader, device, modalities, use_cam_loss, cam_type):
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting predictions"):
            inputs = prepare_inputs(batch, modalities, device)
            labels = batch['label']
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device)
            else:
                labels = torch.tensor(labels, device=device)
            labels = labels.view(-1)

            outputs = model(inputs, labels=None)
            if use_cam_loss:
                outputs = outputs[0]

            logits = outputs.detach()
            preds = torch.argmax(logits, dim=1)

            names = batch.get('sample_name')
            if isinstance(names, torch.Tensor):
                names = names.tolist()
            if isinstance(names, np.ndarray):
                names = names.tolist()
            if isinstance(names, (str, bytes)):
                names = [names]

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_names.extend([str(n) for n in names])

    logits_np = torch.cat(all_logits).numpy() if all_logits else np.array([])
    preds_np = torch.cat(all_preds).numpy() if all_preds else np.array([])
    labels_np = torch.cat(all_labels).numpy() if all_labels else np.array([])
    names_np = np.array(all_names)

    return {
        'sample_names': names_np,
        'logits': logits_np,
        'preds': preds_np,
        'labels': labels_np,
    }

                    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    seeds = [ 42, 43, 44, 45]
    num_labels = 7
    num_epochs = 64
    batch_size = 32
    lr = 1e-6
    eta_min = 2e-7
    patience = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modalities = ['T', "A", 'V']  # Can be modified to ['T', 'A'], etc.
    hidden_dim = 512
    feature_tpye = "sequence_features" # "pooled_features" or "sequence_features"
    use_cross_modal = True
    use_raw_text = True
    use_cam_loss = False
    use_raw_audio = False
    whisper_use_adapters = True
    cam_type = "AVcam_to_CAM" # "T_to_CAM" or "AV_to_CAM" "Tcam_to_CAM"  "AVcam_to_CAM "


    if use_raw_audio:
        batch_size = 16


    if feature_tpye=="pooled_features":
        use_cross_modal = True

    data_paths = {
        'train': {
            'text': '/path/to/data/MELD/text_features/train_text_features.npz',
            'text_raw_path': '/path/to/data/MELD/processed_train_T_emo.csv',
            'audio': '/path/to/data/MELD/audio_features/train_audio_features_processed.npz',
            'video': '/path/to/data/MELD/face_features/train_video_features_processed.npz',
            'audio_csv_path': '/path/to/data/MELD/train_sent_emo.csv',
            'audio_data_path':  '/path/to/data/MELD/train_A/train'
        },
        'test': {
            'text': '/path/to/data/MELD/text_features/test_text_features.npz',
            'text_raw_path': '/path/to/data/MELD/processed_test_T_emo.csv',
            'audio': '/path/to/data/MELD/audio_features/test_audio_features_processed.npz',
            'video': '/path/to/data/MELD/face_features/test_video_features_processed.npz',
            'audio_csv_path': '/path/to/data/MELD/test_sent_emo.csv',
            'audio_data_path': '/path/to/data/MELD/test_A/test'
        }
    }

    def build_dataset(split: str):
        cfg = data_paths[split]
        if use_raw_text and use_raw_audio:
            return TA_raw_MELDDataset(
                cfg['text'],
                cfg['audio'],
                cfg['video'],
                modalities,
                split=split,
                feature_type=feature_tpye,
                text_path=cfg['text_raw_path'],
                audio_csv_path=cfg['audio_csv_path'],
                audio_data_path=cfg['audio_data_path']
            )
        elif use_raw_text:
            return T_raw_MELDDataset(
                cfg['text'],
                cfg['audio'],
                cfg['video'],
                modalities,
                split=split,
                feature_type=feature_tpye,
                text_path=cfg['text_raw_path']
            )
        else:
            return MELDDataset(
                cfg['text'],
                cfg['audio'],
                cfg['video'],
                modalities,
                split=split,
                feature_type=feature_tpye
            )

    def create_model():
        return MultimodalFusionModel(
            text_dim=1024,
            audio_dim=768,
            video_dim=768,
            hidden_dim=hidden_dim,
            num_classes=num_labels,
            modalities=modalities,
            feature_type=feature_tpye,
            use_cross_modal=use_cross_modal,
            use_raw_text=use_raw_text,
            use_cam_loss=use_cam_loss,
            use_raw_audio=use_raw_audio,
            whisper_use_adapters=whisper_use_adapters,
            cam_type=cam_type
        ).to(device)

    emotion_labels = {
        3: "anger",
        5: "disgust",
        6: "fear",
        0: "joy",
        2: "neutral",
        1: "sadness",
        4: "surprise"
    }

    best_accuracies = []
    best_weighted_f1s = []

    global_best_acc = 0.0
    global_best_f1 = 0.0
    global_best_seed = None
    global_best_model_state = None

    for seed in seeds:
        print(f"\n=== Training with Seed {seed} ===\n")
        set_seed(seed)

        train_dataset = build_dataset('train')
        test_dataset = build_dataset('test')

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = create_model()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=8, eta_min=eta_min)
        ema = EMA(model, decay=0.999)

        best_acc = 0.0
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, optimizer, scheduler, ema, device, use_cam_loss=use_cam_loss, use_raw_audio=use_raw_audio, cam_type=cam_type)
            test_loss, test_accuracy, test_weighted_f1, class_accuracies = evaluate(
                model, test_loader, ema, device, num_labels, use_cam_loss, use_raw_audio, cam_type = cam_type
            )

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Weighted-F1: {test_weighted_f1:.4f}")
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            print("Class-wise Accuracies:")
            class_acc_str = "  "
            for label_id, acc in class_accuracies.items():
                emotion_name = emotion_labels.get(label_id, f"Label {label_id}")
                class_acc_str += f"{emotion_name}: {acc:.4f}  "
            print(class_acc_str)

            if test_weighted_f1 > global_best_f1:
                global_best_acc = test_accuracy
                global_best_f1 = test_weighted_f1
                global_best_seed = seed
                ema.apply_shadow()
                global_best_model_state = model.state_dict()
                ema.restore()
                print(f"New global best found: Test Accuracy={global_best_acc:.4f},  Weighted-F1={global_best_f1:.4f},  Seed={seed}, Epoch={epoch+1}")

            if test_weighted_f1 > best_f1:
                best_acc = test_accuracy
                best_f1 = test_weighted_f1
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Patience Counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        best_accuracies.append(best_acc)
        best_weighted_f1s.append(best_f1)
        print(f"\nSeed {seed} - Best Test Accuracy: {best_acc:.4f}, Corresponding Weighted-F1: {best_f1:.4f}")

        print(f"\nClearing GPU memory for Seed {seed}...")
        del model
        del optimizer
        del scheduler
        del ema
        del train_loader
        del test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared. Current allocated memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    if global_best_acc > 0:
        save_path = f"/path/to/checkpoints/multimodal_fusion_best_acc_{global_best_acc:.4f}_seed_{global_best_seed}.pth"
        torch.save(global_best_model_state, save_path)
        print(f"\nSaved global best model parameters to: {save_path}")
        print(f"Global Best Test Accuracy: {global_best_acc:.4f}, Weighted-F1: {global_best_f1:.4f}, Seed: {global_best_seed}")

        set_seed(global_best_seed)
        best_train_dataset = build_dataset('train')
        best_test_dataset = build_dataset('test')

        train_loader_best = DataLoader(
            best_train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader_best = DataLoader(
            best_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        best_model = create_model()
        best_model.load_state_dict(global_best_model_state)

        train_preds = collect_predictions(best_model, train_loader_best, device, modalities, use_cam_loss, cam_type)
        test_preds = collect_predictions(best_model, test_loader_best, device, modalities, use_cam_loss, cam_type)

        prefix = os.path.splitext(save_path)[0]
        train_pred_path = f"{prefix}_train_predictions.npz"
        test_pred_path = f"{prefix}_test_predictions.npz"
        np.savez(train_pred_path, **train_preds)
        np.savez(test_pred_path, **test_preds)
        print(f"Saved train predictions to: {train_pred_path}")
        print(f"Saved test predictions to: {test_pred_path}")

    acc_mean = np.mean(best_accuracies)
    acc_std = np.std(best_accuracies)
    f1_mean = np.mean(best_weighted_f1s)
    f1_std = np.std(best_weighted_f1s)

    print("\n=== Final Results Across Seeds ===")
    print(f"Best Test Accuracies: {best_accuracies}")
    print(f"Mean Test Accuracy: {acc_mean:.4f}, Std Dev: {acc_std:.4f}")
    print(f"Corresponding Weighted-F1s: {best_weighted_f1s}")
    print(f"Mean Weighted-F1: {f1_mean:.4f}, Std Dev: {f1_std:.4f}")

if __name__ == "__main__":
    main()
