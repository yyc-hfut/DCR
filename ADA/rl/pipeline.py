import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


def normalize_sample_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    name = str(name).replace("\\", "/")
    parts = name.split("/")
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"dia{parts[-3]}_utt{parts[-2]}"
    base = parts[-1]
    return os.path.splitext(base)[0]


def load_general_feature_dict(npz_path, prefer_key="sequence_features"):
    data = np.load(npz_path, allow_pickle=True)
    available_keys = data.files
    if prefer_key in available_keys:
        key = prefer_key
    elif "sequence_features" in available_keys:
        key = "sequence_features"
    elif "pooled_features" in available_keys:
        key = "pooled_features"
    else:
        raise KeyError(
            f"No suitable feature key found in {npz_path}. Available keys: {available_keys}"
        )

    names = [normalize_sample_name(n) for n in data["sample_names"]]
    feats = data[key]
    mapping = {name: feats[i].astype(np.float32) for i, name in enumerate(names)}
    feature_shape = feats.shape[1:]
    return mapping, feature_shape


def build_agent_inputs(
    sample_names,
    feature_store,
    default_features,
    device,
    expert_features,
    text_logits=None,
    audio_logits=None,
    video_logits=None,
    multimodal_store=None,
):
    required_modalities = ["T", "A", "V"]
    for key in required_modalities:
        if key not in expert_features:
            raise KeyError(f"Missing expert feature for modality '{key}' in expert_features")

    feature_inputs = {}
    logits_inputs = {}

    for modality in required_modalities:
        feats = []
        masks = []
        mapping = feature_store[modality]
        default = default_features[modality]
        for raw_name in sample_names:
            name = normalize_sample_name(raw_name)
            feature = mapping.get(name)
            if feature is None:
                feats.append(default)
                masks.append(0.0)
            else:
                feats.append(feature)
                masks.append(1.0)
        arr = np.stack(feats, axis=0).astype(np.float32)
        mask_arr = np.array(masks, dtype=np.float32).reshape(-1, 1)
        feature_inputs[modality] = {
            "memory": torch.from_numpy(arr).to(device),
            "mask": torch.from_numpy(mask_arr).to(device),
            "query": expert_features[modality].detach().to(device).float(),
        }

    if video_logits is not None:
        logits_inputs["V"] = video_logits.detach().to(device)
    if text_logits is not None:
        logits_inputs["T"] = text_logits.detach().to(device)
    if audio_logits is not None:
        logits_inputs["A"] = audio_logits.detach().to(device)

    if multimodal_store is not None and text_logits is not None:
        mm_vals = []
        for i, raw_name in enumerate(sample_names):
            key = normalize_sample_name(raw_name)
            mm_logit = multimodal_store["logits"].get(key)
            if mm_logit is None:
                mm_logit = text_logits[i].detach().cpu().numpy()
            mm_vals.append(mm_logit)
        mm_tensor = torch.tensor(mm_vals, device=device, dtype=text_logits.dtype)
        logits_inputs["M"] = mm_tensor
    elif text_logits is not None:
        logits_inputs["M"] = text_logits.detach().to(device)

    return feature_inputs, logits_inputs


def augment_agent_inputs(
    agent_inputs,
    logit_inputs,
    drop_prob_single=0.0,
    drop_prob_double=0.0,
    noise_std=0.0,
):
    if logit_inputs is None:
        logit_inputs = {}

    modalities = list(agent_inputs.keys())
    if not modalities:
        return agent_inputs, logit_inputs

    batch_size = agent_inputs[modalities[0]]["memory"].size(0)
    device = agent_inputs[modalities[0]]["memory"].device

    if drop_prob_single > 0.0 or drop_prob_double > 0.0:
        rand_vals = torch.rand(batch_size, device=device)
        drop_mask = torch.zeros(batch_size, len(modalities), dtype=torch.bool, device=device)
        cumulative_double = drop_prob_double
        cumulative_single = drop_prob_double + drop_prob_single
        for i in range(batch_size):
            drop_count = 0
            if rand_vals[i] < cumulative_double:
                drop_count = min(2, len(modalities))
            elif rand_vals[i] < cumulative_single:
                drop_count = min(1, len(modalities))
            if drop_count > 0:
                selected = random.sample(modalities, k=drop_count)
                for mod in selected:
                    drop_mask[i, modalities.index(mod)] = True
    else:
        drop_mask = None

    for mod_idx, modality in enumerate(modalities):
        entry = agent_inputs[modality]
        mem = entry["memory"]
        mask = entry["mask"]
        query = entry["query"]

        if noise_std > 0.0:
            mask_expand = mask
            for _ in range(mem.dim() - mask.dim()):
                mask_expand = mask_expand.unsqueeze(-1)
            mem += torch.randn_like(mem) * noise_std * mask_expand
            query += torch.randn_like(query) * noise_std * mask

        if drop_mask is not None:
            to_drop = drop_mask[:, mod_idx]
            if to_drop.any():
                mem[to_drop] = 0.0
                mask[to_drop] = 0.0
                query[to_drop] = 0.0
                if modality in logit_inputs:
                    logit_inputs[modality][to_drop] = 0.0

    return agent_inputs, logit_inputs


def load_multimodal_predictions(train_npz, test_npz):
    multimodal = {}
    for split, path in [("train", train_npz), ("test", test_npz)]:
        data = np.load(path, allow_pickle=True)
        names = [normalize_sample_name(n) for n in data["sample_names"]]
        preds = data["preds"]
        logits = data["logits"]
        multimodal[split] = {
            "preds": {names[i]: preds[i] for i in range(len(names))},
            "logits": {names[i]: logits[i] for i in range(len(names))},
        }
    return multimodal


def evaluate_agent(
    agent,
    experts,
    data_loader,
    device,
    num_labels,
    num_actions,
    agent_feature_store,
    default_features,
    multimodal_store=None,
):
    was_training = agent.training
    agent.eval()

    all_preds = []
    all_labels = []
    action_counts = torch.zeros(num_actions, dtype=torch.long)

    with torch.no_grad():
        for batch in data_loader:
            text_input = batch["text"].to(device)
            audio_input = batch["audio"].to(device)
            video_input = batch["video"].to(device)
            labels = batch["label"].squeeze(-1).to(device)
            sample_names = batch["sample_name"]
            if isinstance(sample_names, torch.Tensor):
                sample_names = sample_names.tolist()
            if isinstance(sample_names, tuple):
                sample_names = list(sample_names)
            if isinstance(sample_names, (str, bytes)):
                sample_names = [sample_names]

            text_logits, text_features = experts["T"](text_input)
            audio_logits, audio_features = experts["A"](audio_input)
            video_logits, video_features = experts["V"](video_input)

            expert_queries = {
                "T": text_features,
                "A": audio_features,
                "V": video_features,
            }
            agent_inputs, logit_inputs = build_agent_inputs(
                sample_names,
                agent_feature_store,
                default_features,
                device,
                expert_queries,
                text_logits=text_logits,
                audio_logits=audio_logits,
                video_logits=video_logits,
                multimodal_store=multimodal_store,
            )

            logit_payload = logit_inputs if logit_inputs else None
            action_logits = agent(agent_inputs, logits=logit_payload)
            actions = torch.argmax(action_logits, dim=1)

            action_counts += torch.bincount(actions.cpu(), minlength=num_actions)

            preds_t = torch.argmax(text_logits, dim=1)

            if multimodal_store is not None:
                mm_preds = []
                for i, name in enumerate(sample_names):
                    key = normalize_sample_name(name)
                    mm_pred = multimodal_store["preds"].get(key)
                    if mm_pred is None:
                        mm_pred = preds_t[i].item()
                    mm_preds.append(mm_pred)
                preds_m = torch.tensor(mm_preds, device=device, dtype=torch.long)
            else:
                preds_m = preds_t.clone()

            expert_preds = torch.stack(
                [
                    preds_t,
                    preds_m,
                ],
                dim=1,
            )
            batch_indices = torch.arange(actions.size(0), device=actions.device)
            chosen_preds = expert_preds[batch_indices, actions]

            all_preds.append(chosen_preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if was_training:
        agent.train()

    if all_preds:
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        preds_np = preds_tensor.numpy()
        labels_np = labels_tensor.numpy()
        accuracy = float((preds_np == labels_np).mean())
        try:
            weighted_f1 = f1_score(labels_np, preds_np, average="weighted")
        except ValueError:
            weighted_f1 = 0.0
        conf_mat = confusion_matrix(labels_np, preds_np, labels=list(range(num_labels)))
        class_totals = conf_mat.sum(axis=1)
        class_correct = conf_mat.diagonal()
        class_acc_dict = {
            f"class_{i}_acc": (class_correct[i] / class_totals[i] if class_totals[i] > 0 else 0.0)
            for i in range(num_labels)
        }
    else:
        accuracy = 0.0
        weighted_f1 = 0.0
        class_acc_dict = {f"class_{i}_acc": 0.0 for i in range(num_labels)}

    total_actions = max(action_counts.sum().item(), 1)
    action_distribution = ", ".join(
        [f"{idx}:{(count.item() / total_actions) * 100:.1f}%" for idx, count in enumerate(action_counts)]
    )

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "class_acc": class_acc_dict,
        "action_dist": action_distribution,
    }
