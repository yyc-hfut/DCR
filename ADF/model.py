import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import os
import torch.nn.functional as F
from transformer_FacialMMT import MELDTransEncoder, AdditiveAttention
from modules.CrossmodalTransformer import CrossModalTransformerEncoder
from modules.transformer_block import TransformerEncoder, InteractionAttention, SelfAttention
from transformers import RobertaTokenizer, RobertaModel, HubertModel, WhisperModel, WhisperConfig
from whisper import CustomWhisperEncoderLayer, BaseModel
from transformers.models.whisper import modeling_whisper as whisper_model
import logging
import transformers
                          
logging.getLogger("transformers").setLevel(logging.ERROR)

whisper_model.WhisperEncoderLayer = CustomWhisperEncoderLayer

    
        
                  
        
                
        
    

                                        
                                        
                                      
    
              
                                                                 
                                           
                                                                       
                                                            
                         
                                                             
        
                     
        
                  
                                                                 
        
                    
        
                                  
                             
            
                          
        
                  
        
                          
        
                
        

class CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(CAMModule, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, features, labels=None):
        batch_size, seq_len, hidden_dim = features.size()
        
                   
        pooled = features.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        
                    
        weights = self.classifier.weight  # (num_classes, hidden_dim)
        cam = torch.einsum('btd,cd->btc', features, weights)  # (batch_size, seq_len, num_classes)
        cam = cam + self.classifier.bias.view(1, 1, -1)        
        
                  
        time_weights = torch.ones(batch_size, seq_len, device=features.device) / seq_len
        
                                
        if labels is not None and self.training:
                           
            label_indices = labels.view(-1, 1, 1).expand(-1, seq_len, 1)  # (batch_size, seq_len, 1)
            true_cam = cam.gather(dim=2, index=label_indices).squeeze(-1)  # (batch_size, seq_len)
            
                        
            time_weights = F.softmax(true_cam, dim=1)  # (batch_size, seq_len)
        
                
        weighted_cam = cam * time_weights.unsqueeze(-1)  # (batch_size, seq_len, num_classes)
        
                        
        soft_labels = F.softmax(weighted_cam, dim=2)  # (batch_size, seq_len, num_classes)
        
              
        loss = 0.0
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return soft_labels, logits, loss, time_weights
    
class RAW_CAMModule(nn.Module):
    def __init__(self, hidden_dim, num_classes, proj_layer=None, transformer=None, conv_layer=None):
        super(RAW_CAMModule, self).__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.proj_layer = proj_layer                          
        self.transformer = transformer                                         
        self.conv_layer = conv_layer                          
    
    def forward(self, features, raw_input=None, labels=None):
                   
        pooled = features.mean(dim=1)  # (batch_size, hidden_dim)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        weights = self.classifier.weight  # (num_classes, hidden_dim)
        cam = torch.einsum('btd,cd->btc', features, weights)  # (batch_size, seq_len, num_classes)
        soft_labels = F.softmax(cam, dim=2)  # (batch_size, seq_len, num_classes)
        
                         
        raw_soft_labels = soft_labels        
        raw_cam = None
        if raw_input is not None and labels is not None and all([self.proj_layer, self.transformer, self.conv_layer]):
                                 
            
                                         
            x = self.proj_layer(raw_input)  # (batch_size, seq_len, hidden_dim)
            x = self.transformer(x, attention_mask=None)  # (batch_size, seq_len, hidden_dim)
            x = x.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
            x = self.conv_layer(x)  # (batch_size, hidden_dim, seq_len)
            x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
            pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
            logits = self.classifier(pooled)  # (batch_size, num_classes)
            
                              
            raw_cam = torch.zeros(raw_input.shape[0], raw_input.shape[1], self.classifier.out_features, 
                                 device=raw_input.device)
            for c in range(self.classifier.out_features):
                      
                if raw_input.grad is not None:
                    raw_input.grad.zero_()
                               
                grad_mask = torch.zeros_like(logits)
                grad_mask[:, c] = 1.0
                      
                logits.backward(gradient=grad_mask, retain_graph=True)
                gradients = raw_input.grad  # (batch_size, seq_len, input_dim)
                            
                alpha = gradients.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
                        
                raw_cam[:, :, c:c+1] = (alpha * raw_input).sum(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
                raw_cam[:, :, c:c+1] = F.relu(raw_cam[:, :, c:c+1])           
            
                 
            raw_soft_labels = F.softmax(raw_cam, dim=2)  # (batch_size, seq_len, num_classes)
            
                
            raw_input.requires_grad_(False)
        
        loss = 0.0
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return soft_labels, raw_soft_labels, logits, loss


class MultimodalFusionModel(nn.Module):
    def __init__(self, text_dim: int, audio_dim: int, video_dim: int, 
                 hidden_dim: int, num_classes: int, modalities: List[str], feature_type: str, use_cross_modal: bool, use_raw_text: bool, 
                 use_cam_loss: bool, use_raw_audio: bool, whisper_use_adapters: bool, cam_type: str):
        super(MultimodalFusionModel, self).__init__()
        self.modalities = modalities
        self.feature_type = feature_type
        self.use_cross_modal = use_cross_modal
        self.use_raw_text = use_raw_text
        self.target_seq_len = 60
        self.use_cam_loss = use_cam_loss
        self.use_raw_audio = use_raw_audio
        self.whisper_use_adapters = whisper_use_adapters
        self.cam_type = cam_type

        if self.use_raw_audio:
            audio_dim = 768
        else:
            audio_dim = 1280
        
        if self.feature_type == 'pooled_features':
            print("Using pooled features")
            input_dim = 0
            if 'T' in modalities:
                input_dim += text_dim
            if 'A' in modalities:
                input_dim += audio_dim
            if 'V' in modalities:
                input_dim += video_dim

            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, num_classes)
            )

        elif self.feature_type == 'sequence_features':
            print("Using sequence features")

            if 'T' in modalities and use_raw_text:
                self.roberta_model = RobertaModel.from_pretrained("roberta-large")
            
            if 'A' in modalities and use_raw_audio:
                config = WhisperConfig.from_pretrained('openai/whisper-small')                               
                config.chunk_length = 12
                config.adapter_dim = 96
                config.adapter_scale = 0.1
                self.whisper_model = WhisperModel.from_pretrained("openai/whisper-small", config = config)
                self.whisper_feature_dim = 768
                self.whisper_model = self.whisper_model.encoder

                             
                original_weights = self.whisper_model.embed_positions.weight.data
                
                                       
                max_positions = int(12 * 100 / 2)               
                self.whisper_model.config.max_source_positions = max_positions
                self.whisper_model.embed_positions = nn.Embedding(
                    max_positions,
                    self.whisper_model.config.d_model
                )
                                                      
                self.whisper_model.embed_positions.weight.data[:] = original_weights[:max_positions]
                self.whisper_model.embed_positions.requires_grad_(False)

                if self.whisper_use_adapters:
                    for param in self.whisper_model.parameters():
                        param.requires_grad = False

                                    
                    for layer in self.whisper_model.layers:
                        if isinstance(layer, CustomWhisperEncoderLayer):
                            for param in layer.S_Adapter.parameters():
                                param.requires_grad = True
                            for param in layer.MLP_Adapter.parameters():
                                param.requires_grad = True
                    num_param = sum(p.numel()
                                    for p in self.whisper_model.parameters() if p.requires_grad)/1e6
                    num_total_param = sum(p.numel() for p in self.whisper_model.parameters())/1e6
                    print(f"Whisper small Trainable parameters: {num_param:.2f}M")
                    print(f"Whisper small Total parameters: {num_total_param:.2f}M")
                
                                    
                self.whisper_conv1d = nn.Conv1d(
                    in_channels=audio_dim,
                    out_channels=audio_dim,
                    kernel_size=10,
                    stride=10,
                    padding=0
                )

            self.text_proj = nn.Linear(text_dim, hidden_dim) if 'T' in modalities else None
            self.audio_proj = nn.Linear(audio_dim, hidden_dim) if 'A' in modalities else None
            self.video_proj = nn.Linear(video_dim, hidden_dim) if 'V' in modalities else None


                        
            self.text_transpose_conv = nn.ConvTranspose1d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=2,padding=1) if 'T' in modalities else None
            self.audio_conv = nn.Conv1d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0) if 'A' in modalities else None
            self.video_transpose_conv = nn.ConvTranspose1d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,stride=2,padding=1) if 'V' in modalities else None

           
            
        
            config = Config(hidden_size=hidden_dim,num_attention_heads=8,  intermediate_size=4 * hidden_dim,   attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,layer_norm_eps=1e-6)
            self.text_transformer = MELDTransEncoder(config=config,layer_num=1,get_max_lens=38,hidden_size=hidden_dim) if 'T' in modalities else None
            self.audio_transformer = MELDTransEncoder(config=config,layer_num=1,get_max_lens=60,hidden_size=hidden_dim) if 'A' in modalities else None
            self.video_transformer = MELDTransEncoder(config=config,layer_num=1,get_max_lens=40,hidden_size=hidden_dim) if 'V' in modalities else None

                        
                    
            if self.cam_type == 'T_to_CAM':
                self.text_cam = CAMModule(hidden_dim, num_classes) if 'T' in modalities else None
                self.audio_seq_classifier = nn.Linear(hidden_dim, num_classes) if 'A' in modalities else None
                self.video_seq_classifier = nn.Linear(hidden_dim, num_classes) if 'V' in modalities else None

            elif self.cam_type == 'AV_to_CAM':
                self.audio_cam = CAMModule(hidden_dim, num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, num_classes) if 'V' in modalities else None
                self.text_seq_classifier = nn.Linear(hidden_dim, num_classes) if 'T' in modalities else None
            
            elif self.cam_type == "Tcam_to_CAM":
                self.text_cam = CAMModule(hidden_dim, num_classes) if 'T' in modalities else None
                self.audio_cam = CAMModule(hidden_dim, num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, num_classes) if 'V' in modalities else None
            
            elif self.cam_type == "AVcam_to_CAM":
                self.text_cam = CAMModule(hidden_dim, num_classes) if 'T' in modalities else None
                self.audio_cam = CAMModule(hidden_dim, num_classes) if 'A' in modalities else None
                self.video_cam = CAMModule(hidden_dim, num_classes) if 'V' in modalities else None
            
            


            if self.use_cross_modal:
                print("Using cross-modal transformers")
                self.crossmodal_ta = CrossModalTransformerEncoder(embed_dim=hidden_dim,num_heads=8,layers=1,attn_dropout=0.1,gelu_dropout=0.1,res_dropout=0.1,embed_dropout=0.1,) if 'T' in modalities and 'A' in modalities else None
                self.crossmodal_ta_v = CrossModalTransformerEncoder(embed_dim=hidden_dim,num_heads=8,layers=1,attn_dropout=0.1,gelu_dropout=0.1,res_dropout=0.1,embed_dropout=0.1,) if 'T' in modalities and 'V' in modalities else None
                self.crossmodal_av = CrossModalTransformerEncoder(embed_dim=hidden_dim,num_heads=8,layers=1,attn_dropout=0.1,gelu_dropout=0.1,res_dropout=0.1,embed_dropout=0.1,) if 'A' in modalities and 'V' in modalities else None

            self.attention = AdditiveAttention(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_dim, num_classes)
            
            self.seq_len = self.target_seq_len * len(modalities)

    
    def forward(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor=None):
        if self.feature_type == 'pooled_features':
            features = []

            if 'T' in self.modalities:
                features.append(inputs['T'])
            if 'A' in self.modalities:
                features.append(inputs['A'])
            if 'V' in self.modalities:
                features.append(inputs['V'])
            
            fused = torch.cat(features, dim=-1)
            
            fused = F.layer_norm(fused, fused.size()[1:])
            output = self.classifier(fused)
            return output
        
        elif self.feature_type == 'sequence_features':
            text_feat = audio_feat = video_feat = None

            if 'T' in self.modalities and self.use_raw_text:
                text_input_ids = inputs['T']['input_ids']
                text_attention_mask = inputs['T']['attention_mask']
                text_output = self.roberta_model(text_input_ids, text_attention_mask)
                text_feat = text_output.last_hidden_state  # (batch_size, 38, 1024)

                batch_size = text_input_ids.shape[0]
                hidden_size = text_feat.shape[-1]
                word_level_features = torch.zeros(batch_size, 38, hidden_size).to(text_input_ids.device)
                word_level_masks = torch.zeros(batch_size, 38).to(text_input_ids.device)

                target_start_pos = inputs['T']['target_start_pos']
                target_end_pos = inputs['T']['target_end_pos']
                
                if target_start_pos is not None and target_end_pos is not None:
                    for i in range(batch_size):
                        start = target_start_pos[i].item()
                        end = target_end_pos[i].item()
                        curr_utt_len = end - start
                        if curr_utt_len > 38:
                            curr_utt_len = 38
                        if curr_utt_len > 0:
                            word_level_features[i, :curr_utt_len] = text_feat[i, start:start + curr_utt_len]
                            word_level_masks[i, :curr_utt_len] = 1
                else:
                    raise ValueError("target_start_pos and target_end_pos must be provided for word-level feature extraction")

                text_feat = word_level_features
                text_attention_mask = word_level_masks

                text_seq = self.text_proj(text_feat)  # (batch_size, 38, hidden_dim)
                text_mask = (text_seq.sum(dim=-1) != 0).float()  # (batch_size, 38)
                text_extended_mask = (1.0 - text_mask.unsqueeze(1).unsqueeze(2)) * -10000.0  # (batch_size, 1, 1, 38)
                                                                                      
                text_feat = self.text_transformer(text_seq, attention_mask=text_extended_mask)  # (batch_size, 38, hidden_dim)

                            
                text_feat = text_feat.transpose(1, 2)  # (batch_size, hidden_dim, 38)
                text_feat = self.text_transpose_conv(text_feat)  # (batch_size, hidden_dim, ~75)
                text_feat = F.adaptive_avg_pool1d(text_feat, self.target_seq_len)  # (batch_size, hidden_dim, 60)
                text_feat = text_feat.transpose(1, 2)  # (batch_size, 60, hidden_dim)
                text_mask = F.interpolate(text_mask.unsqueeze(1), size=self.target_seq_len, mode='linear').squeeze(1)

                if self.use_cam_loss and self.cam_type == 'T_to_CAM':
                    text_cam, text_logits, text_loss, text_time_weights = self.text_cam(text_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'AV_to_CAM':
                            
                    text_logits = self.text_seq_classifier(text_feat)  # (batch_size, 60, num_classes)
                    text_logits = F.log_softmax(text_logits, dim=2)
                elif self.use_cam_loss and self.cam_type == 'Tcam_to_CAM':
                    text_cam, text_logits, text_loss, text_time_weights = self.text_cam(text_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'AVcam_to_CAM':
                    text_cam, text_logits, text_loss, text_time_weights = self.text_cam(text_feat, labels=labels)

            
                     
            elif 'T' in self.modalities and self.text_proj is not None:
                text_seq = self.text_proj(inputs['T'])  # (batch_size, 38, hidden_dim)
                text_feat = self.text_transformer(text_seq, attention_mask=None)  # (batch_size, 38, hidden_dim)
            
            if 'A' in self.modalities and self.audio_proj is not None and self.use_raw_audio:
                audio_input = inputs['A']['input_values']  # (batch_size, 60, 768)
                if self.whisper_use_adapters:
                    audio_outputs = self.whisper_model(audio_input, attention_mask=None)
                
                                
                hidden_states = audio_outputs.last_hidden_state  # (batch_size, 60, 768)
                hidden_states = hidden_states.transpose(1, 2)  # (batch_size, 768, 60)
                hidden_states = self.whisper_conv1d(hidden_states)  # (batch_size, 768, 6)
                hidden_states = hidden_states.transpose(1, 2)  # (batch_size, 6, 768)

                audio_seq = self.audio_proj(hidden_states)
                audio_feat = self.audio_transformer(audio_seq, attention_mask=None)  # (batch_size, 60, hidden_dim)

                                     
                audio_feat = audio_feat.transpose(1, 2)
                audio_feat = self.audio_conv(audio_feat)  # (batch_size, hidden_dim, 60)
                audio_feat = audio_feat.transpose(1, 2)  # (batch_size, 60, hidden_dim)

                if self.use_cam_loss and self.cam_type == 'AV_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'T_to_CAM':
                    audio_logits = self.audio_seq_classifier(audio_feat)  # (batch_size, 60, num_classes)
                    audio_logits = F.log_softmax(audio_logits, dim=2)
                elif self.use_cam_loss and self.cam_type == 'Tcam_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'AVcam_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)
            
            elif 'A' in self.modalities and self.audio_proj is not None:
                audio_seq = self.audio_proj(inputs['A'])  # (batch_size, 60, hidden_dim)
                audio_feat = self.audio_transformer(audio_seq, attention_mask=None)  # (batch_size, 60, hidden_dim)
                audio_feat = audio_feat.transpose(1, 2)
                audio_feat = self.audio_conv(audio_feat)  # (batch_size, hidden_dim, 60)
                audio_feat = audio_feat.transpose(1, 2)  # (batch_size, 60, hidden_dim)

                       
                if self.use_cam_loss and self.cam_type == 'AV_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'T_to_CAM':
                    audio_logits = self.audio_seq_classifier(audio_feat)  # (batch_size, 60, num_classes)
                    audio_logits = F.log_softmax(audio_logits, dim=2)
                elif self.use_cam_loss and self.cam_type == 'Tcam_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'AVcam_to_CAM':
                    audio_cam, audio_logits, audio_loss, audio_time_weights = self.audio_cam(audio_feat, labels=labels)

            
            if 'V' in self.modalities and self.video_proj is not None:
                video_seq = self.video_proj(inputs['V'])  # (batch_size, 40, hidden_dim)
                video_feat = self.video_transformer(video_seq, attention_mask=None)  # (batch_size, 40, hidden_dim)
                video_feat = video_feat.transpose(1, 2)
                video_feat = self.video_transpose_conv(video_feat)  # (batch_size, hidden_dim, ~75)
                video_feat = F.adaptive_avg_pool1d(video_feat, self.target_seq_len)  # (batch_size, hidden_dim, 60)
                video_feat = video_feat.transpose(1, 2)  # (batch_size, 60, hidden_dim)

                if self.use_cam_loss and self.cam_type == 'AV_to_CAM':
                    video_cam, video_logits, video_loss, video_time_weights = self.video_cam(video_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'T_to_CAM':
                    video_logits = self.video_seq_classifier(video_feat)  # (batch_size, 60, num_classes)
                    video_logits = F.log_softmax(video_logits, dim=2)
                elif self.use_cam_loss and self.cam_type == 'AVcam_to_CAM':
                    video_cam, video_logits, video_loss, video_time_weights = self.video_cam(video_feat, labels=labels)
                elif self.use_cam_loss and self.cam_type == 'Tcam_to_CAM':
                    video_cam, video_logits, video_loss, video_time_weights = self.video_cam(video_feat, labels=labels)

                                                    
                                   
                    
                      
            cam_loss = 0
            if self.use_cam_loss:
                kl_loss = nn.KLDivLoss(reduction='none')
                if self.cam_type == 'T_to_CAM':
                    text_cam = text_cam.detach()
                    if 'A' in self.modalities:
                        kl_div_a = kl_loss(audio_logits, text_cam)
                        weighted_kl_a = (kl_div_a * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_a
                    if 'V' in self.modalities:
                        kl_div_v = kl_loss(video_logits, text_cam)
                        weighted_kl_v = (kl_div_v * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_v
                elif self.cam_type == 'AV_to_CAM':
                    audio_cam = audio_cam.detach()
                    video_cam = video_cam.detach()
                    if 'A' in self.modalities:
                        kl_div_ta = kl_loss(text_logits, audio_cam)
                        weighted_kl_ta = (kl_div_ta * audio_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_ta
                    if 'V' in self.modalities:
                        kl_div_tv = kl_loss(text_logits, video_cam)
                        weighted_kl_tv = (kl_div_tv * video_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_tv
                elif self.cam_type == 'Tcam_to_CAM':
                    text_cam = text_cam.detach()
                    if 'A' in self.modalities:
                        kl_div_a = kl_loss(audio_cam, text_cam)
                        weighted_kl_a = (kl_div_a * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_a
                    if 'V' in self.modalities:
                        kl_div_v = kl_loss(video_cam, text_cam)
                        weighted_kl_v = (kl_div_v * text_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_v
                elif self.cam_type == 'AVcam_to_CAM':
                    audio_cam = audio_cam.detach()
                    video_cam = video_cam.detach()
                    if 'A' in self.modalities:
                        kl_div_ta = kl_loss(text_cam, audio_cam)
                        weighted_kl_ta = (kl_div_ta * audio_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_ta
                    if 'V' in self.modalities:
                        kl_div_tv = kl_loss(text_cam, video_cam)
                        weighted_kl_tv = (kl_div_tv * video_time_weights.unsqueeze(-1)).mean()
                        cam_loss += 0.5*weighted_kl_tv
            

            if self.use_cross_modal:
                if 'T' in self.modalities and 'A' in self.modalities and self.crossmodal_ta is not None:
                    text_feat = text_feat.transpose(0, 1)  # (38, batch_size, hidden_dim)
                    audio_feat = audio_feat.transpose(0, 1)  # (60, batch_size, hidden_dim)
                    text_cross_audio = self.crossmodal_ta(text_feat, audio_feat, audio_feat)  # (38, batch_size, hidden_dim)
                    audio_cross_text = self.crossmodal_ta(audio_feat, text_feat, text_feat)  # (60, batch_size, hidden_dim)
                    ta_cross_feat = text_cross_audio # (38, batch_size, hidden_dim)
                else:
                    if 'T' in self.modalities:
                        ta_cross_feat = text_feat.transpose(0, 1)  # (38, batch_size, hidden_dim)
                    if 'A' in self.modalities:
                        ta_cross_feat = audio_feat.transpose(0, 1)  # (60, batch_size, hidden_dim)
                
                if 'V' in self.modalities and self.crossmodal_ta_v is not None:
                    video_feat = video_feat.transpose(0, 1)  # (40, batch_size, hidden_dim)
                    video_cross_ta = self.crossmodal_ta_v(video_feat, ta_cross_feat, ta_cross_feat)  # (40, batch_size, hidden_dim)
                    ta_cross_video = self.crossmodal_ta_v(ta_cross_feat, video_feat, video_feat)  # (38+60, batch_size, hidden_dim)
                    final_feat = ta_cross_video # (38+60, batch_size, hidden_dim)
                elif 'V' in self.modalities and self.crossmodal_av is None:
                    video_feat = video_feat.transpose(0, 1)  # (40, batch_size, hidden_dim)
                    video_cross_ta = self.crossmodal_av(video_feat, ta_cross_feat, ta_cross_feat)  # (40, batch_size, hidden_dim)
                    ta_cross_video = self.crossmodal_av(ta_cross_feat, video_feat, video_feat)  # (38+60, batch_size, hidden_dim)
                    final_feat = ta_cross_video # (38+60, batch_size, hidden_dim)
                else:
                    final_feat = ta_cross_feat

            else:
                final_feat = []
                if 'T' in self.modalities:
                    final_feat.append(text_feat.transpose(0, 1))
                if 'A' in self.modalities:
                    final_feat.append(audio_feat.transpose(0, 1))
                if 'V' in self.modalities:
                    final_feat.append(video_feat.transpose(0, 1))
                
                final_feat = torch.cat(final_feat, dim=0)
                
            final_feat = final_feat.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
            multimodal_out, _ = self.attention(final_feat)  # (batch_size, hidden_dim)
            multimodal_out = self.dropout(multimodal_out)
            logits = self.classifier(multimodal_out)

            if self.use_cam_loss and self.cam_type == 'T_to_CAM':
                return logits, cam_loss, text_loss
            elif self.use_cam_loss and self.cam_type == 'AV_to_CAM':
                return logits, cam_loss, audio_loss, video_loss
            elif self.use_cam_loss and self.cam_type == 'Tcam_to_CAM':
                return logits, cam_loss, text_loss
            elif self.use_cam_loss and self.cam_type == 'AVcam_to_CAM':
                return logits, cam_loss, audio_loss, video_loss
            else:
                return logits
        

     
class Config:
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, attention_probs_dropout_prob,
                 hidden_dropout_prob, layer_norm_eps):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps