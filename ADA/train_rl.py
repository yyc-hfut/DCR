import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.feature_dataset import MELDDataset
from rl.seed import set_seed
from models.unimodal import TextClassifier, AudioClassifier, VideoClassifier
from rl.agent import RLAgent, ValueNet
from rl.pipeline import augment_agent_inputs, build_agent_inputs, evaluate_agent, load_general_feature_dict, load_multimodal_predictions, normalize_sample_name



os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def main():
    seeds = [41, 42, 43, 44, 45]
    num_labels = 7
    num_epochs = 300
    batch_size = 64
    agent_lr = 5e-6
    value_lr = 1e-5
    hidden_dim = 512
    agent_embed_dim = 768
    value_hidden_dim = 256
    value_coef = 0.5
    entropy_coef = 0.01
    num_actions = 2
    general_pooling_mode = 'mean'
    general_conv_kernel = 3
    include_memory_tokens = False
    modality_drop_prob_single = 0.2
    modality_drop_prob_double = 0.05
    agent_noise_std = 0.01
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    expert_model_paths = {
        'T': '/path/to/checkpoints/unimodal_experts',
        'A': '/path/to/checkpoints/unimodal_experts',                       
        'V': '/path/to/checkpoints/unimodal_experts'
    }

    data_paths = {
        'train': {
            'text': '/path/to/data/MELD/text_features',
            'audio': '/path/to/data/MELD/audio_features',
            'video': '/path/to/data/MELD/face_features',
        },
        'test': {
            'text': '/path/to/data/MELD/text_features',
            'audio': '/path/to/data/MELD/audio_features',
            'video': '/path/to/data/MELD/face_features',
        }
    }

    print("Loading datasets and general features...")
    train_dataset = MELDDataset(
        data_paths['train']['text'],
        data_paths['train']['audio'],
        data_paths['train']['video'],
        modalities=['T', 'A', 'V'],
        feature_type='sequence_features'
    )
    test_dataset = MELDDataset(
        data_paths['test']['text'],
        data_paths['test']['audio'],
        data_paths['test']['video'],
        modalities=['T', 'A', 'V'],
        split='test',
        feature_type='sequence_features'
    )

    text_feat_dim = train_dataset.aligned_data['text'].shape[-1]
    text_seq_len = train_dataset.aligned_data['text'].shape[1] if train_dataset.aligned_data['text'].ndim == 3 else 1

    agent_feature_paths = {
        'train': {
            'T': '/path/to/data/MELD/text_features/train_text_features_general.npz',
            'A': '/path/to/data/MELD/audio_features/train_audio_features_general.npz',
            'V': '/path/to/data/MELD/face_features/train_face_features_general_clip.npz',
        },
        'test': {
            'T': '/path/to/data/MELD/text_features/test_text_features_general.npz',
            'A': '/path/to/data/MELD/audio_features/test_audio_features_general.npz',
            'V': '/path/to/data/MELD/face_features/test_face_features_general_clip.npz',
        }
    }

    agent_feature_store = {split: {} for split in ['train', 'test']}
    agent_feature_defaults = {}
    for split in ['train', 'test']:
        for modality in ['T', 'A', 'V']:
            mapping, feat_shape = load_general_feature_dict(agent_feature_paths[split][modality])
            agent_feature_store[split][modality] = mapping
            agent_feature_defaults.setdefault(modality, np.zeros(feat_shape, dtype=np.float32))

    for modality in ['T', 'A', 'V']:
        missing_train = [name for name in train_dataset.sample_names if normalize_sample_name(name) not in agent_feature_store['train'][modality]]
        if missing_train:
            print(f"Warning: {len(missing_train)} missing agent features for modality {modality} in training split. Example: {missing_train[:5]}")
        missing_test = [name for name in test_dataset.sample_names if normalize_sample_name(name) not in agent_feature_store['test'][modality]]
        if missing_test:
            print(f"Warning: {len(missing_test)} missing agent features for modality {modality} in test split. Example: {missing_test[:5]}")

    modality_dims = {}
    sample_name_example = normalize_sample_name(train_dataset.sample_names[0])
    for modality in ['T', 'A', 'V']:
        sample_feat = agent_feature_store['train'][modality].get(sample_name_example)
        if sample_feat is None:
            sample_feat = agent_feature_defaults[modality]
        modality_dims[modality] = sample_feat.shape[-1] if sample_feat.ndim >= 1 else 1

    multimodal_train_npz = '/path/to/checkpoints/multimodal_fusion_train_predictions.npz'
    multimodal_test_npz = '/path/to/checkpoints/multimodal_fusion_test_predictions.npz'
    multimodal_available = os.path.exists(multimodal_train_npz) and os.path.exists(multimodal_test_npz)
    multimodal_predictions = load_multimodal_predictions(multimodal_train_npz, multimodal_test_npz) if multimodal_available else None

    base_save_dir = "/path/to/checkpoints/rl_agent"
    os.makedirs(base_save_dir, exist_ok=True)

    best_records = []

    for seed in seeds:
        print(f"\n===== Training with seed {seed} =====")
        set_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        text_expert = TextClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_labels,
            use_precomputed=True,
            input_dim=text_feat_dim,
            target_seq_len=text_seq_len,
        ).to(device)
        text_expert.load_state_dict(torch.load(expert_model_paths['T'], map_location=device), strict=False)

        audio_expert = AudioClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
        audio_expert.load_state_dict(torch.load(expert_model_paths['A'], map_location=device))

        video_expert = VideoClassifier(hidden_dim=hidden_dim, num_classes=num_labels).to(device)
        video_expert.load_state_dict(torch.load(expert_model_paths['V'], map_location=device))

        experts = {'T': text_expert, 'A': audio_expert, 'V': video_expert}
        for expert in experts.values():
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            sample_text_feat = experts['T'](sample_batch['text'].to(device))[1]
            sample_audio_feat = experts['A'](sample_batch['audio'].to(device))[1]
            sample_video_feat = experts['V'](sample_batch['video'].to(device))[1]
        query_dims = {
            'T': sample_text_feat.shape[-1],
            'A': sample_audio_feat.shape[-1],
            'V': sample_video_feat.shape[-1],
        }
        del sample_batch

        multimodal_train = multimodal_predictions['train'] if multimodal_predictions else None
        multimodal_test = multimodal_predictions['test'] if multimodal_predictions else None

        agent = RLAgent(
            modality_dims=modality_dims,
            num_actions=num_actions,
            query_dims=query_dims,
            embed_dim=agent_embed_dim,
            general_pooling=general_pooling_mode,
            conv_kernel_size=general_conv_kernel,
            include_memory_tokens=include_memory_tokens,
            num_classes=num_labels,
        ).to(device)
        value_net = ValueNet(input_dim=agent_embed_dim, hidden_dim=value_hidden_dim).to(device)
        agent_optimizer = torch.optim.Adam(agent.parameters(), lr=agent_lr)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_lr)

        save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
        os.makedirs(save_dir_seed, exist_ok=True)
        metrics_dir = os.path.join(save_dir_seed, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_log = []
        best_wf1 = -1.0
        best_record = {
            'seed': seed,
            'best_wf1': -1.0,
            'best_acc': 0.0,
            'best_action_dist': '',
            'best_epoch': 0,
        }

        for epoch in range(num_epochs):
            total_rewards = 0
            total_loss = 0
            total_value_loss = 0
            total_entropy = 0
            epoch_preds = []
            epoch_labels = []
            epoch_actions = []

            for batch in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}/{num_epochs}"):
                text_input = batch['text'].to(device)
                audio_input = batch['audio'].to(device)
                video_input = batch['video'].to(device)
                labels = batch['label'].squeeze(-1).to(device)
                sample_names = batch['sample_name']
                if isinstance(sample_names, torch.Tensor):
                    sample_names = sample_names.tolist()
                if isinstance(sample_names, tuple):
                    sample_names = list(sample_names)
                if isinstance(sample_names, (str, bytes)):
                    sample_names = [sample_names]

                with torch.no_grad():
                    text_logits, text_feat = experts['T'](text_input)
                    audio_logits, audio_feat = experts['A'](audio_input)
                    video_logits, video_feat = experts['V'](video_input)

                    expert_queries = {
                        'T': text_feat,
                        'A': audio_feat,
                        'V': video_feat,
                    }
                    agent_inputs, logit_inputs = build_agent_inputs(
                        sample_names,
                        agent_feature_store['train'],
                        agent_feature_defaults,
                        device,
                        expert_queries,
                        text_logits=text_logits,
                        audio_logits=audio_logits,
                        video_logits=video_logits,
                        multimodal_store=multimodal_train,
                    )
                    agent_inputs, logit_inputs = augment_agent_inputs(
                        agent_inputs,
                        logit_inputs,
                        drop_prob_single=modality_drop_prob_single,
                        drop_prob_double=modality_drop_prob_double,
                        noise_std=agent_noise_std,
                    )

                logit_payload = logit_inputs if logit_inputs else None
                action_logits, agent_repr = agent(agent_inputs, logits=logit_payload, return_repr=True)
                action_dist = Categorical(logits=action_logits)
                actions = action_dist.sample()

                with torch.no_grad():
                    preds_t = torch.argmax(text_logits, dim=1)

                    if multimodal_train is not None:
                        mm_vals = []
                        for i, raw_name in enumerate(sample_names):
                            key = normalize_sample_name(raw_name)
                            mm_pred = multimodal_train['preds'].get(key)
                            if mm_pred is None:
                                mm_pred = preds_t[i].item()
                            mm_vals.append(mm_pred)
                        preds_m = torch.tensor(mm_vals, device=device, dtype=torch.long)
                    else:
                        preds_m = preds_t.clone()

                    probs_t = F.softmax(text_logits, dim=1)
                    modality_probs = [probs_t]

                    if multimodal_train is not None and 'logits' in multimodal_train:
                        mm_logits = []
                        for i, raw_name in enumerate(sample_names):
                            key = normalize_sample_name(raw_name)
                            mm_logit = multimodal_train['logits'].get(key)
                            if mm_logit is None:
                                mm_logit = text_logits[i].detach().cpu().numpy()
                            mm_logits.append(mm_logit)
                        mm_tensor = torch.tensor(mm_logits, device=device, dtype=text_logits.dtype)
                        probs_m = F.softmax(mm_tensor, dim=1)
                    else:
                        probs_m = F.one_hot(preds_m, num_classes=num_labels).float()

                    modality_probs.append(probs_m)

                    all_preds = torch.stack([preds_t, preds_m], dim=1)
                    batch_indices = torch.arange(actions.size(0), device=actions.device)
                    chosen_preds = all_preds[batch_indices, actions]

                    stacked_probs = torch.stack([
                        modality_probs[0][batch_indices, actions],
                        modality_probs[1][batch_indices, actions],
                    ], dim=1)
                    chosen_probs = stacked_probs[torch.arange(actions.size(0), device=actions.device), actions]

                    correct_mask = chosen_preds.eq(labels)
                    rewards = torch.where(correct_mask, chosen_probs, -chosen_probs)

                    epoch_preds.append(chosen_preds.detach().cpu())
                    epoch_labels.append(labels.detach().cpu())
                    epoch_actions.append(actions.detach().cpu())

                values = value_net(agent_repr)
                log_probs = action_dist.log_prob(actions)
                with torch.no_grad():
                    advantages = rewards - values.detach()
                    adv_mean = advantages.mean()
                    adv_std = advantages.std()
                    adv_std_val = adv_std.item()
                    if math.isnan(adv_std_val) or adv_std_val < 1e-6:
                        advantages = advantages - adv_mean
                    else:
                        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                policy_loss = -(log_probs * advantages).mean()
                value_loss = F.mse_loss(values, rewards)
                entropy = action_dist.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                agent_optimizer.zero_grad()
                value_optimizer.zero_grad()
                loss.backward()
                agent_optimizer.step()
                value_optimizer.step()

                total_rewards += rewards.mean().item()
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

            avg_reward = total_rewards / len(train_loader)
            avg_loss = total_loss / len(train_loader)
            avg_value_loss = total_value_loss / len(train_loader)
            avg_entropy = total_entropy / len(train_loader)

            epoch_preds_tensor = torch.cat(epoch_preds)
            epoch_labels_tensor = torch.cat(epoch_labels)
            epoch_actions_tensor = torch.cat(epoch_actions)
            preds_np = epoch_preds_tensor.numpy()
            labels_np = epoch_labels_tensor.numpy()
            epoch_accuracy = float((preds_np == labels_np).mean()) if preds_np.size > 0 else 0.0
            try:
                epoch_weighted_f1 = f1_score(labels_np, preds_np, average='weighted') if preds_np.size > 0 else 0.0
            except ValueError:
                epoch_weighted_f1 = 0.0
            if preds_np.size > 0:
                conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
                class_totals = conf_mat.sum(axis=1)
                class_correct = conf_mat.diagonal()
                class_acc_dict = {
                    f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
                    for i in range(num_labels)
                }
            else:
                class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}
            action_counts = torch.bincount(epoch_actions_tensor, minlength=num_actions)
            total_actions = max(epoch_actions_tensor.numel(), 1)
            action_distribution = ", ".join(
                [f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)]
            )

            test_metrics = evaluate_agent(
                agent,
                experts,
                test_loader,
                device,
                num_labels,
                num_actions,
                agent_feature_store['test'],
                agent_feature_defaults,
                multimodal_test,
            )

            metrics_entry = {
                "seed": seed,
                "epoch": epoch + 1,
                "train_avg_reward": avg_reward,
                "train_avg_loss": avg_loss,
                "train_avg_value_loss": avg_value_loss,
                "train_avg_entropy": avg_entropy,
                "train_accuracy": epoch_accuracy,
                "train_weighted_f1": epoch_weighted_f1,
                "train_action_dist": action_distribution,
                "test_accuracy": test_metrics['accuracy'],
                "test_weighted_f1": test_metrics['weighted_f1'],
                "test_action_dist": test_metrics['action_dist']
            }
            metrics_entry.update({f"train_{k}": v for k, v in class_acc_dict.items()})
            metrics_entry.update({f"test_{k}": v for k, v in test_metrics['class_acc'].items()})
            metrics_log.append(metrics_entry)

            print(
                f"Seed {seed} Epoch {epoch+1}/{num_epochs} | Train Reward: {avg_reward:.4f} | Train Loss: {avg_loss:.4f} "
                f"| Train ValLoss: {avg_value_loss:.4f} | Train Entropy: {avg_entropy:.4f} "
                f"| Train Acc: {epoch_accuracy * 100:.2f}% | Train W-F1: {epoch_weighted_f1:.4f} "
                f"| Train Action Dist: [{action_distribution}]"
            )
            print("  Train Class Accuracies: " + ", ".join([f"{k}:{v*100:.2f}%" for k, v in class_acc_dict.items()]))
            print(
                f"  Test  | Acc: {test_metrics['accuracy'] * 100:.2f}% | W-F1: {test_metrics['weighted_f1']:.4f} "
                f"| Action Dist: [{test_metrics['action_dist']}]"
            )
            print("  Test Class Accuracies: " + ", ".join([f"{k}:{v*100:.2f}%" for k, v in test_metrics['class_acc'].items()]))

            if test_metrics['weighted_f1'] > best_wf1:
                best_wf1 = test_metrics['weighted_f1']
                best_record.update({
                    'best_wf1': test_metrics['weighted_f1'],
                    'best_acc': test_metrics['accuracy'],
                    'best_action_dist': test_metrics['action_dist'],
                    'best_epoch': epoch + 1,
                })

        save_dir_seed = os.path.join(base_save_dir, f"seed_{seed}")
        save_path = os.path.join(save_dir_seed, "agent_final.pth")
        torch.save(agent.state_dict(), save_path)
        print(f"Saved trained RL agent to: {save_path}")

        metrics_dir = os.path.join(save_dir_seed, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_df = pd.DataFrame(metrics_log)
        metrics_csv_path = os.path.join(metrics_dir, "train_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved training metrics to: {metrics_csv_path}")

        best_records.append(best_record)
        print(
            f"Seed {seed} best W-F1: {best_record['best_wf1']:.4f} | Best Acc: {best_record['best_acc'] * 100:.2f}% "
            f"| Action Dist: [{best_record['best_action_dist']}] at epoch {best_record['best_epoch']}"
        )

        del agent, value_net, text_expert, audio_expert, video_expert
        torch.cuda.empty_cache()

    summary_df = pd.DataFrame(best_records)
    summary_path = os.path.join(base_save_dir, "best_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print("\n===== Best results across seeds =====")
    print(summary_df)
    print(f"Saved best summary to: {summary_path}")


if __name__ == "__main__":
    main()

