class Config:
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        layer_norm_eps,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
