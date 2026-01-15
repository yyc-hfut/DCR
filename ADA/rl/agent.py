import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralMemoryEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, pooling="mean", conv_kernel_size=3):
        super().__init__()
        valid_pooling = {"mean", "conv1d", "none"}
        if pooling not in valid_pooling:
            raise ValueError(f"Unsupported pooling mode '{pooling}'. Expected one of {valid_pooling}.")
        self.pooling = pooling
        if pooling == "conv1d":
            self.encoder = nn.Sequential(
                nn.Conv1d(
                    input_dim,
                    embed_dim,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.encoder = nn.Conv1d(input_dim, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory):
        if memory.dim() == 1:
            memory = memory.unsqueeze(0).unsqueeze(0)
        elif memory.dim() == 2:
            memory = memory.unsqueeze(1)
        memory = memory.float()
        memory = memory.transpose(1, 2)
        encoded = self.encoder(memory)
        if self.pooling == "mean":
            encoded = encoded.mean(dim=-1, keepdim=True)
        elif self.pooling == "conv1d":
            encoded = F.adaptive_avg_pool1d(encoded, 1)
        encoded = encoded.transpose(1, 2)
        encoded = self.norm(encoded)
        return encoded


class RLAgent(nn.Module):
    def __init__(
        self,
        modality_dims,
        num_actions,
        query_dims=None,
        embed_dim=256,
        num_heads=4,
        ff_multiplier=2,
        dropout=0.1,
        general_pooling="mean",
        conv_kernel_size=3,
        include_memory_tokens=False,
        num_classes=None,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        if not self.modalities:
            raise ValueError("modality_dims must not be empty.")

        if query_dims is None:
            query_dims = {m: modality_dims[m] for m in self.modalities}

        for modality in self.modalities:
            if modality not in query_dims:
                raise KeyError(f"Missing query dimension for modality '{modality}' in query_dims")
        self.modality_to_idx = {m: i for i, m in enumerate(self.modalities)}
        self.embed_dim = embed_dim
        self.include_memory_tokens = include_memory_tokens

        if isinstance(general_pooling, str):
            pooling_map = {m: general_pooling for m in self.modalities}
        else:
            pooling_map = {m: general_pooling.get(m, "mean") for m in self.modalities}

        if isinstance(conv_kernel_size, int):
            kernel_map = {m: conv_kernel_size for m in self.modalities}
        else:
            kernel_map = {m: conv_kernel_size.get(m, 3) for m in self.modalities}

        self.memory_encoders = nn.ModuleDict()
        self.query_proj = nn.ModuleDict()
        self.query_norm = nn.ModuleDict()
        self.modality_attn = nn.ModuleDict()
        self.mod_ln = nn.ModuleDict()

        self.mod_embeddings = nn.Parameter(torch.randn(len(self.modalities), embed_dim))
        self.missing_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(embed_dim)) for m in self.modalities}
        )

        for modality in self.modalities:
            self.memory_encoders[modality] = GeneralMemoryEncoder(
                input_dim=modality_dims[modality],
                embed_dim=embed_dim,
                pooling=pooling_map[modality],
                conv_kernel_size=kernel_map[modality],
            )
            self.query_proj[modality] = nn.Linear(query_dims[modality], embed_dim)
            self.query_norm[modality] = nn.LayerNorm(embed_dim)
            self.modality_attn[modality] = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True, dropout=dropout
            )
            self.mod_ln[modality] = nn.LayerNorm(embed_dim)

        if num_classes is not None:
            self.logit_proj = nn.ModuleDict(
                {key: nn.Linear(num_classes, embed_dim) for key in ["V", "T", "M", "A"]}
            )
            self.logit_embeddings = nn.ParameterDict(
                {key: nn.Parameter(torch.randn(embed_dim)) for key in ["V", "T", "M", "A"]}
            )
        else:
            self.logit_proj = None
            self.logit_embeddings = None

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.self_ln = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_multiplier * embed_dim),
            nn.ReLU(),
            nn.Linear(ff_multiplier * embed_dim, embed_dim),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.query_token = nn.Parameter(torch.randn(embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

        self.out_proj = nn.Linear(embed_dim, num_actions)

    def forward(self, features, logits=None, return_repr=False):
        if not isinstance(features, dict):
            raise TypeError("features must be a dict containing modality inputs.")

        modality_tokens = []
        batch_size = None
        for modality in self.modalities:
            if modality not in features:
                raise KeyError(f"Missing modality '{modality}' in features")
            entry = features[modality]
            mask = None
            if isinstance(entry, torch.Tensor):
                memory = entry
                query = None
            elif isinstance(entry, dict):
                mask = entry.get("mask")
                memory = entry.get("memory")
                if memory is None and "sequence" in entry:
                    memory = entry["sequence"]
                query = entry.get("query")
                if query is None and "pooled" in entry:
                    query = entry["pooled"]
            else:
                raise TypeError(f"Unsupported feature format for modality '{modality}'")

            if memory is None and isinstance(query, torch.Tensor):
                if query.dim() == 1:
                    memory = query.unsqueeze(0).unsqueeze(1)
                elif query.dim() == 2:
                    memory = query.unsqueeze(1)
                else:
                    raise ValueError(f"Cannot infer memory from query with shape {query.shape}")

            if query is None:
                if not isinstance(memory, torch.Tensor):
                    raise KeyError(
                        f"Modality '{modality}' requires at least tensor inputs to infer query"
                    )
                if memory.dim() == 3:
                    query = memory.mean(dim=1)
                elif memory.dim() == 2:
                    query = memory
                elif memory.dim() == 1:
                    query = memory.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported memory shape {memory.shape} for modality '{modality}'")

            if not isinstance(memory, torch.Tensor) or not isinstance(query, torch.Tensor):
                raise TypeError(f"Memory and query must be tensors for modality '{modality}'")

            if query.dim() == 3 and query.size(1) == 1:
                query = query.squeeze(1)
            if query.dim() == 1:
                query = query.unsqueeze(0)

            if batch_size is None:
                batch_size = query.size(0)
            elif query.size(0) != batch_size:
                raise ValueError("All modalities must share the same batch size")

            memory_tokens = self.memory_encoders[modality](memory)
            query_vec = self.query_proj[modality](query.float())
            query_vec = self.query_norm[modality](query_vec).unsqueeze(1)
            attn_out, _ = self.modality_attn[modality](query_vec, memory_tokens, memory_tokens)
            context = self.mod_ln[modality](attn_out + query_vec)
            context = self.dropout(context)
            context = context + self.mod_embeddings[self.modality_to_idx[modality]].view(1, 1, -1)

            if mask is not None:
                mask = mask.to(context.device).float()
                if mask.dim() == 1:
                    mask = mask.view(-1, 1, 1)
                elif mask.dim() == 2:
                    mask = mask.unsqueeze(-1)
                elif mask.dim() != 3:
                    mask = mask.view(mask.size(0), 1, 1)
                missing_embed = self.missing_embeddings[modality].view(1, 1, -1)
                context = context * mask + (1 - mask) * missing_embed

            modality_tokens.append(context)

            if self.include_memory_tokens:
                mem_tokens = memory_tokens + self.mod_embeddings[self.modality_to_idx[modality]].view(
                    1, 1, -1
                )
                if mask is not None:
                    mem_mask = mask.expand(-1, mem_tokens.size(1), -1)
                    missing_embed = self.missing_embeddings[modality].view(1, 1, -1)
                    mem_tokens = mem_tokens * mem_mask + (1 - mem_mask) * missing_embed
                modality_tokens.append(self.dropout(mem_tokens))

        if batch_size is None:
            raise ValueError("No modality data provided to the agent.")

        tokens = torch.cat(modality_tokens, dim=1)

        if self.logit_proj is not None and logits is not None:
            logit_tokens = []
            for key, proj in self.logit_proj.items():
                if key not in logits or logits[key] is None:
                    continue
                log_tensor = logits[key]
                if not isinstance(log_tensor, torch.Tensor):
                    log_tensor = torch.tensor(log_tensor, dtype=torch.float32, device=tokens.device)
                log_tensor = log_tensor.to(tokens.device).float()
                if log_tensor.dim() == 1:
                    log_tensor = log_tensor.unsqueeze(0)
                log_proj = proj(log_tensor)
                log_proj = log_proj + self.logit_embeddings[key]
                logit_tokens.append(log_proj.unsqueeze(1))
            if logit_tokens:
                tokens = torch.cat([tokens, torch.cat(logit_tokens, dim=1)], dim=1)

        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.self_ln(tokens + self.dropout(attn_out))

        ffn_out = self.ffn(tokens)
        tokens = self.ffn_ln(tokens + self.dropout(ffn_out))

        query = self.query_token.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, self.embed_dim)
        query_out, _ = self.cross_attn(query, tokens, tokens)
        agent_repr = query_out.squeeze(1)

        action_logits = self.out_proj(agent_repr)
        if return_repr:
            return action_logits, agent_repr
        return action_logits


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, representation):
        x = F.relu(self.fc1(representation))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)
