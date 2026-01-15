import torch
from torch import nn
import transformers
from transformers import RobertaTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
import string



class wordMELDDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_len=5, max_seq_length=160, use_all_context=False, add_speaker = False):
              
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.use_all_context = use_all_context
        self.add_speaker = add_speaker
        if context_len == None or use_all_context == True:
            self.max_seq_length = 512
        else:
            self.max_seq_length = max_seq_length
        
                          
        self.dialogues = self.data.groupby("Dialogue_ID")
        self.dialogue_ids = list(self.dialogues.groups.keys())
        
        if self.add_speaker and "Speaker" not in self.data.columns:
            raise ValueError("Data must contain a 'Speaker' column when add_speaker=True")
        
                   
        print(f"Loaded dataset from {data_path}")
        print(f"Total utterances: {len(self.data)}")
        print(f"Total dialogues: {len(self.dialogue_ids)}")
        print(f"Use full context: {use_all_context}")
        if not use_all_context:
            print(f"Context length: {context_len}")
        if self.add_speaker:
            print("Adding speaker prefix to utterances")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
                
        row = self.data.iloc[idx]
        dialogue_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        target_utterance = row["Utterance"]
        emotion_id = row["emotion_id"]


               
        sample_name = f"dia{dialogue_id}_utt{utterance_id}"


                       
        if self.add_speaker:
            speaker = row["Speaker"]
            target_utterance = f"{speaker}: {target_utterance}"
        
              
        dialogue = self.dialogues.get_group(dialogue_id)
        
                                    

               
        if self.use_all_context:
            all_utterances = dialogue.sort_values("Utterance_ID")[["Utterance", "Speaker"]].values
            target_idx = dialogue[dialogue["Utterance_ID"] == utterance_id].index[0]
            target_pos_in_list = dialogue.index.get_loc(target_idx)
                                         
            if self.add_speaker:
                all_utterances = [f"{speaker}: {utt}" for utt, speaker in all_utterances]
            else:
                all_utterances = [utt for utt, _ in all_utterances]
        else:
            context_utterances = dialogue[dialogue["Utterance_ID"] < utterance_id][["Utterance", "Speaker"]].values
            if self.context_len is not None and len(context_utterances) > self.context_len:
                context_utterances = context_utterances[-self.context_len:]
                                             
            if self.add_speaker:
                all_utterances = [f"{speaker}: {utt}" for utt, speaker in context_utterances] + [target_utterance]
            else:
                all_utterances = [utt for utt, _ in context_utterances] + [target_utterance]
            target_pos_in_list = len(context_utterances)

                              
        
                                         
        
                    
                                                      
        input_text = "</s></s>".join(all_utterances)                                                       
        
              
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,                     
            return_tensors="pt"                      
        )

                        
        input_ids = encoding["input_ids"][0]
        sep_positions = [i for i, token_id in enumerate(input_ids) if token_id == self.tokenizer.sep_token_id]                                         


                        
        if target_pos_in_list == 0:            
            target_start_pos = 1          
        else:
                               
            if (target_pos_in_list*2 -1) >= len(sep_positions):                                 
                target_start_pos = sep_positions[-2]
            
            else:
                target_start_pos = sep_positions[target_pos_in_list * 2 - 1] + 1
        
                                   
        if target_pos_in_list < len(all_utterances) - 1 and (target_pos_in_list + 1) * 2 <= len(sep_positions):
            target_end_pos = sep_positions[target_pos_in_list * 2]
        else:
            if (target_pos_in_list*2 -1) >= len(sep_positions):
                target_end_pos = sep_positions[-1]
            else: 
                target_end_pos = sep_positions[-1] if input_ids[-1] in [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id] else len(input_ids)

                                                 
                                                                                             

        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(emotion_id, dtype=torch.long),
            "target_start_pos": target_start_pos,
            "target_end_pos": target_end_pos,
            "sample_name" :sample_name
        }
    
      
def test_dataset():
                  
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
           
    train_dataset = wordMELDDataset(
        data_path="/path/to/data/MELD/processed_test_T_emo.csv",
        tokenizer=tokenizer,
        context_len=None,
        max_seq_length=128,
        use_all_context=True,
        add_speaker = True
    )
    
                        
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    
             
    for i, batch in enumerate(train_loader):
        if i >= 5:              
            break
        
        input_ids = batch["input_ids"][0]
        attention_mask = batch["attention_mask"][0]
        label = batch["label"][0]
        target_start_pos = batch["target_start_pos"][0].item()
        target_end_pos = batch["target_end_pos"][0].item()
        
                            
        target_tokens = input_ids[target_start_pos:target_end_pos]
        decoded_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
        
                    
        full_decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        print(f"\nSample {i}:")
        print(f"Full decoded input text:\n{full_decoded_text}")
        print(f"Target start position: {target_start_pos}")
        print(f"Target end position: {target_end_pos}")
        print(f"Target utterance token IDs: {target_tokens.tolist()}")
        print(f"Target utterance decoded: {decoded_text}")
        print(f"Label: {label.item()}")

if __name__ == "__main__":
    test_dataset()