import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperConfig
from .modules import CustomWhisperEncoderLayer, BaseModel
from transformers.models.whisper import modeling_whisper as whisper_model
import numpy as np

whisper_model.WhisperEncoderLayer = CustomWhisperEncoderLayer

               
try:
    from timm.models.registry import register_model
except:
    from timm.models import register_model

 

def create_custom_processor(model_name="openai/whisper-small", chunk_length=3):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_name,
        chunk_length=chunk_length,      
    )

    tokenizer = WhisperTokenizer.from_pretrained(
        model_name,
    )

    processor = WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )

    return processor


class CustomWhisperModel(BaseModel):
    def __init__(self, num_classes=0, model_name="openai/whisper-small.en", chunk_length=5, adapter_dim=192, adapter_scale=0.1):
        super(CustomWhisperModel, self).__init__()
        
                        
        
                 
        config = WhisperConfig.from_pretrained(model_name)
        config.chunk_length = chunk_length
        config.adapter_dim = adapter_dim
        config.adapter_scale = adapter_scale
        self.audio_encoder = WhisperModel.from_pretrained(model_name, config=config).encoder
        
                     
        original_weights = self.audio_encoder.embed_positions.weight.data
        
                               
        max_positions = int(chunk_length * 100 / 2)               
        self.audio_encoder.config.max_source_positions = max_positions
        self.audio_encoder.embed_positions = nn.Embedding(
            max_positions,
            self.audio_encoder.config.d_model
        )
                                              
        self.audio_encoder.embed_positions.weight.data[:] = original_weights[:max_positions]
        self.audio_encoder.embed_positions.requires_grad_(False)
        self.last_hiddim_dim = self.audio_encoder.config.d_model


        if num_classes > 0:
            self.cls_head = nn.Linear(
                self.last_hiddim_dim, num_classes)
        else:
            self.cls_head = None
        self.init_extra_weights()
    @torch.no_grad()
    def preprocess(self, audio, sampling_rate=16000):
        features = self.processor(
            audio, 
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features
        return features


    def forward_features(self, x, sampling_rate=16000):
        x = x.to(self.audio_encoder.device)
        x = self.audio_encoder(x).last_hidden_state
        return x
    def forward(self, v, audio, sampling_rate=16000):
        x = self.forward_features(audio, sampling_rate)
        x = x.mean(dim=1)        
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x


@register_model
def whisper_audio_encoder_small_3s(num_classes=7, pretrained=None, pretrained_cfg=None, **kwargs):
    model = CustomWhisperModel(num_classes=num_classes, model_name="openai/whisper-small", chunk_length=3)
            
    for param in model.parameters():
        param.requires_grad = False

                    
    for layer in model.audio_encoder.layers:
        if isinstance(layer, CustomWhisperEncoderLayer):
            for param in layer.S_Adapter.parameters():
                param.requires_grad = True
            for param in layer.MLP_Adapter.parameters():
                param.requires_grad = True
    for param in model.cls_head.parameters():
        param.requires_grad = True
    num_param = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)/1e6
    num_total_param = sum(p.numel() for p in model.parameters())/1e6
    print(f"Trainable parameters: {num_param:.2f}M")
    print(f"Total parameters: {num_total_param:.2f}M")
    return model

@register_model
def whisper_audio_encoder_small_5s(num_classes=7, pretrained=None, pretrained_cfg=None, **kwargs):
    model = CustomWhisperModel(num_classes=num_classes, model_name="openai/whisper-small", chunk_length=5)
            
    for param in model.parameters():
        param.requires_grad = False

                    
    for layer in model.audio_encoder.layers:
        if isinstance(layer, CustomWhisperEncoderLayer):
            for param in layer.S_Adapter.parameters():
                param.requires_grad = True
            for param in layer.MLP_Adapter.parameters():
                param.requires_grad = True
    for param in model.cls_head.parameters():
        param.requires_grad = True
    num_param = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)/1e6
    num_total_param = sum(p.numel() for p in model.parameters())/1e6
    print(f"Trainable parameters: {num_param:.2f}M")
    print(f"Total parameters: {num_total_param:.2f}M")
    return model
@register_model
def whisper_audio_encoder_small_6s(num_classes=7, pretrained=None, pretrained_cfg=None, **kwargs):
    model = CustomWhisperModel(num_classes=num_classes, model_name="openai/whisper-small", chunk_length=5)
            
    for param in model.parameters():
        param.requires_grad = False

                    
    for layer in model.audio_encoder.layers:
        if isinstance(layer, CustomWhisperEncoderLayer):
            for param in layer.S_Adapter.parameters():
                param.requires_grad = True
            for param in layer.MLP_Adapter.parameters():
                param.requires_grad = True
    for param in model.cls_head.parameters():
        param.requires_grad = True
    num_param = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)/1e6
    num_total_param = sum(p.numel() for p in model.parameters())/1e6
    print(f"Trainable parameters: {num_param:.2f}M")
    print(f"Total parameters: {num_total_param:.2f}M")
    return model

@register_model
def whisper_audio_encoder_base_5s(num_classes=7, pretrained=None, pretrained_cfg=None, **kwargs):
    model = CustomWhisperModel(
        num_classes=num_classes,
        model_name="openai/whisper-base",
        chunk_length=5)
            
    for param in model.parameters():
        param.requires_grad = False

                    
    for layer in model.audio_encoder.layers:
        if isinstance(layer, CustomWhisperEncoderLayer):
            for param in layer.S_Adapter.parameters():
                param.requires_grad = True
            for param in layer.MLP_Adapter.parameters():
                param.requires_grad = True
    for param in model.cls_head.parameters():
        param.requires_grad = True
    num_param = sum(p.numel()
                    for p in model.parameters() if p.requires_grad)/1e6
    num_total_param = sum(p.numel() for p in model.parameters())/1e6
    print(f"Trainable parameters: {num_param:.2f}M")
    print(f"Total parameters: {num_total_param:.2f}M")
    return model

      
if __name__ == "__main__":
          
    model = whisper_audio_encoder_small_3s(num_classes=7)

            
    audio_signal = np.random.randn(57000)          

    predictions = model(None, audio_signal)
    print(f"Model predictions shape: {predictions.shape}")  # (1, 7)
