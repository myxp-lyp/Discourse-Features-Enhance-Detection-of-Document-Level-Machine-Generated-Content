import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn import Parameter
import math
from tqdm import tqdm
import torch.nn.init as init
import os
from pytorch_transformers import BertTokenizer

from custom_transformer import TransformerEncoderLayer

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad=True):
        super(LearnedPositionalEmbedding, self).__init__()
        self.left_pad = left_pad
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        """Forward pass for generating positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor of input ids.

        Returns:
            torch.Tensor: Tensor containing learned positional embeddings.
        """
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        if self.left_pad:
            positions = positions.flip(dims=[0])
        
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        #positions[input_ids == self.padding_idx] = self.padding_idx
        # Clone the positions tensor before modifying it
        positions_clone = positions.clone()
        positions_clone[input_ids == self.padding_idx] = self.padding_idx

        
        embeddings = self.embedding(positions_clone)

        return embeddings

def generate_positional_ids(input_ids):
    """
    Generate positional IDs for a batch of input sequences.

    Args:
        input_ids (torch.Tensor): Tensor of input IDs with shape (batch_size, seq_length).
    
    Returns:
        torch.Tensor: Tensor of positional IDs with the same shape as input_ids.
    """
    batch_size, seq_length, _ = input_ids.shape
    positional_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
    
    return positional_ids


class SentenceEncoder(nn.Module):
    def __init__(self, tokenizer,args):
        super(SentenceEncoder, self).__init__()
        
        # Load the pre-trained transformer model
        self.padding_idx = tokenizer.pad_token_id

        self.dropout = args.dropout
        embed_tokens = nn.Embedding(tokenizer.vocab_size, args.encoder_embed_dim, self.padding_idx)
        embed_dim = args.encoder_embed_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = LearnedPositionalEmbedding(
            args.max_seq_length, embed_dim, self.padding_idx,
            left_pad=False,
        )
        # Modify some layers in the encoder
        
        self.encoder = nn.ModuleList([])
        self.encoder.extend([
            TransformerEncoderLayer(d_model=embed_dim, nhead=args.encoder_attention_heads, dim_feedforward=args.encoder_ffn_embed_dim)
            for i in range(args.encoder_layers)
        ])
        
        self.encoder1 = nn.ModuleList([])
        self.encoder1.extend([
            TransformerEncoderLayer(d_model=embed_dim, nhead=args.encoder_attention_heads, dim_feedforward=args.encoder_ffn_embed_dim)
            for i in range(args.encoder_layers)
        ])
        # This is used for checking whether the score gain comes from hibert or discourse
        '''
        self.doc_encoder = nn.ModuleList([])
        self.doc_encoder.extend([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=args.encoder_attention_heads, dim_feedforward=args.encoder_ffn_embed_dim)
            for i in range(args.encoder_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
            nn.Sigmoid()  # Sigmoid for binary classification
        )        
        '''
    
                    
    def forward(self, input_ids, mask):#, bsz, n_sent):
        bsz, n_sent, seqlen = input_ids.size()
        input_ids = input_ids.view(bsz * n_sent, seqlen)
        mask = mask.view(bsz * n_sent, seqlen)
        
        x = self.embed_scale * self.embed_tokens(input_ids)
        x += self.embed_positions(input_ids)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # x: [bsz*n_sent, seqlen, hid_dim] -> [s, b*n, h]
        x = x.transpose(0, 1)
        #for mask, True is for value 0 that needs to be masked
        mask = mask == 0
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask = mask)
        
        # we select the first token to be representation sent_repr: [bsz, n_sent, hid_dim]
        sent_repr = x[0].view(bsz, n_sent, x.shape[2]) #still 3D as bsz * n_sent is the first dimension before this step
        return sent_repr
        
        #following is for ablation study
        for layer in self.doc_encoder:
            sent_repr = layer(sent_repr)
            
        sent_repr = sent_repr.transpose(0, 1)
        last_hidden_state = sent_repr[0] # first sentence to represent the paragraph embedding
        
    
        # Classification layer
        logits = self.classifier(last_hidden_state)
        return logits



class SentenceDecoder(nn.Module):
    def __init__(self, args):
        super(SentenceDecoder, self).__init__()
        #This embedding is for PDTB codes of each sentence
        
        self.padding_idx = args.Discourse_code_num -1 #PDTB padding as in PDTB_predict is : num_embedding -1

        self.embedding = nn.Embedding(args.Discourse_code_num, args.decoder_embed_dim, self.padding_idx)
        self.positional_encoding = LearnedPositionalEmbedding(args.max_para_length, args.decoder_embed_dim, self.padding_idx, left_pad=False)
        
        self.decoder_layers = nn.TransformerDecoderLayer(args.decoder_embed_dim, nhead=args.decoder_attention_heads, dim_feedforward=args.decoder_ffn_embed_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, args.decoder_layers)
        
        #Final classification
        self.fc_out = nn.Linear(args.decoder_embed_dim, 2)
        self.activation = nn.Sigmoid()

    def forward(self, target, memory, target_mask, memory_mask, target_key_padding_mask, memory_key_padding_mask):
        #use key padding mask instead of target mask here
        #target shape:[bsz, para_len]
        x = self.embedding(target) 
        x += self.positional_encoding(target)
        
        # shape: [bsz, para_len, hid_dim] -> [p, b, h]
        x = x.transpose(0, 1)
        #shape: [bsz, para_len]
        target_key_padding_mask = target_key_padding_mask == 0
        out = self.transformer_decoder(
            x,
            memory,
            tgt_mask=target_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=target_key_padding_mask #mem_padding = tgt_padding
        )
        
        # output shape: (target_seq_length, batch_size, embedding_dim), select the first sentence to be the representation
        out = self.fc_out(out[0]) # shape: (target_seq_length, batch_size, vocab_size)
        out = self.activation(out)
        return out
    
    
class TransformerModel(nn.Module):
    def __init__(self, tokenizer, args):
        super(TransformerModel, self).__init__()
        self.encoder = SentenceEncoder(tokenizer,args)
        self.decoder = SentenceDecoder(args)

        self.initialize_weights()
    def forward(self, input_ids, attention_mask, pdtb, pdtb_mask):
        # Encoder forward pass
        memory = self.encoder(input_ids, attention_mask)
        
        #return memory # This is for ablation study
        
        memory = memory.transpose(0, 1)
        # Decoder forward pass
        
        output = self.decoder(
            pdtb, 
            memory, 
            None, 
            None,  # No memory mask
            pdtb_mask, 
            None  # No memory mask
        )
        
        return output
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        print("layer initialized.")
class Args:
    pass    
def load_data(data_folder_path):
    '''
    Args:
        data_folder_path: That includes HPC and MGC encoding and mask, with its pdtb codes and mask
        
    return:
        TensorDataset: text, mask, code, code_mask, label
    
    '''
    HPC_text = torch.load(os.path.join(data_folder_path,'HPC.pt'))
    MGC_text= torch.load(os.path.join(data_folder_path,'MGC.pt'))
    Hmask = torch.load(os.path.join(data_folder_path,'HPC_mask.pt'))
    Mmask = torch.load(os.path.join(data_folder_path,'MGC_mask.pt'))
    HPC_code = torch.load(os.path.join(data_folder_path,'HPC_PDTB.pt'))
    MGC_code = torch.load(os.path.join(data_folder_path,'MGC_PDTB.pt'))
    Hcode_mask = torch.load(os.path.join(data_folder_path,'HPC_PDTB_mask.pt'))
    Mcode_mask = torch.load(os.path.join(data_folder_path,'MGC_PDTB_mask.pt'))

    text = torch.cat((HPC_text, MGC_text), dim=0)
    MGC_label = torch.zeros(MGC_text.shape[0])
    HPC_label = torch.ones(HPC_text.shape[0])
    label = torch.cat((HPC_label, MGC_label), dim=0)
    mask = torch.cat((Hmask, Mmask), dim=0)
    code = torch.cat((HPC_code, MGC_code), dim=0)
    code_mask = torch.cat((Hcode_mask, Mcode_mask), dim=0)
    dataset = TensorDataset(text, mask, code, code_mask, label)
    
    #dataset = TensorDataset(MGC_text, Mmask, MGC_code, Mcode_mask, MGC_label)#used for cases study debugging

    return dataset
    
    
    
def base_parameters(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)

    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    
    args.dropout = getattr(args, 'dropout', 0.1)
    args.Discourse_code_num = getattr(args, 'discourse_num', 16) #with padding idx -1, <CLS> -2
    args.max_seq_length = getattr(args, 'max_seq_length', 64) #This should be aligned with PDTB_predict.py
    args.max_para_length = getattr(args, 'max_para_length', 64) #This should be aligned with PDTB_predict.py


train_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/'
val_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/'
test_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/'

#train_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/lfqa/train/'
#val_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/lfqa/valid/'
#test_path = '/data/yl7622/MRes/discourse_attention_torch/data_encoding/lfqa/test/'

train_dataset = load_data(train_path)
val_dataset = load_data(val_path)
test_dataset = load_data(test_path)


# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20
batch_size = 16
best_accuracy = 0
best_epoch = 0
early_stop = 0
best_model = None

args = Args()
base_parameters(args)

tokenizer = BertTokenizer.from_pretrained('/data/yl7622/MRes/pdtb3/scripts/output/pdtb3/fold_3')
#tokenizer = BertTokenizer.from_pretrained('/data/yl7622/MRes/pdtb3/scripts/output/fold_2')

model = TransformerModel(tokenizer, args)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
model.to(device)

train_loss = []
valid_loss = []
valid_acc = []

for epoch in range(num_epochs):
    model.train()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    running_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False) as train_pbar:
        for batch in train_pbar:
            
            optimizer.zero_grad()

            input_ids, attention_mask, pdtb, pdtb_mask, labels = batch
            
            
            input_ids, attention_mask, pdtb, pdtb_mask, labels = input_ids.to(device), attention_mask.to(device), pdtb.to(device), pdtb_mask.to(device), labels.to(device)
            
            #bsz = input_ids.size()[0] // n_sent
            #labels = labels[:bsz]
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            #outputs = model(input_ids, bsz, n_sent)
            
            outputs = model(input_ids,attention_mask,pdtb,pdtb_mask)
            loss = criterion(outputs, labels.long())
           
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
           # _, predicted = torch.max(outputs, 1)

            #print(predicted)
            #print(labels)
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
    
        
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.8f}")

    train_loss.append(epoch_loss)
    running_loss = 0
    # Validation
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    correct_preds = 0
    total_preds = 0
    with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False) as val_pbar:
        for batch in val_pbar:
            input_ids, attention_mask, pdtb, pdtb_mask, labels = batch
            
            
            input_ids, attention_mask, pdtb, pdtb_mask, labels = input_ids.to(device), attention_mask.to(device), pdtb.to(device), pdtb_mask.to(device), labels.to(device)
                        

            outputs = model(input_ids,attention_mask,pdtb,pdtb_mask)
            loss = criterion(outputs, labels.long())
            _, predicted = torch.max(outputs, 1)
            
            running_loss += loss.item()
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            val_pbar.set_postfix({'accuracy': correct_preds / total_preds})
    
    epoch_loss = running_loss / len(val_dataset)
    valid_loss.append(epoch_loss)
    accuracy = correct_preds / total_preds
    #print(f"Validation Accuracy: {accuracy:.3f}")
    valid_acc.append(accuracy)
    if accuracy > best_accuracy:
        early_stop = 0
        best_accuracy = accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "/data/yl7622/MRes/discourse_attention_torch/best_M4_model_PDTB3/bestmodel_1e-6.pth")
        print("Model saved at epoch", best_epoch)
        print("Accuracy: ", accuracy)
        best_model = model
        
    else:
        early_stop += 1
        if early_stop > 3:
            break
#print("pdtb3 train_loss:",train_loss)
#print("valid_loss:",valid_loss)
#print("valid_acc:",valid_acc)

#exit()   
#test begins  
load_path = "/data/yl7622/MRes/discourse_attention_torch/best_M4_model_PDTB3/bestmodel_1e-6.pth"
model.load_state_dict(torch.load(load_path))       
#model = best_model   
model.eval()
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
correct_preds = 0
total_preds = 0
with tqdm(test_loader, desc=f"Result - Test", leave=False) as test_pbar:
        for batch in test_pbar:
            input_ids, attention_mask, pdtb, pdtb_mask, labels = batch
            
            
            input_ids, attention_mask, pdtb, pdtb_mask, labels = input_ids.to(device), attention_mask.to(device), pdtb.to(device), pdtb_mask.to(device), labels.to(device)
                        

            outputs = model(input_ids,attention_mask,pdtb,pdtb_mask)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            test_pbar.set_postfix({'accuracy': correct_preds / total_preds})
    
accuracy = correct_preds / total_preds
print(f"PDTB3: Test Accuracy: {accuracy:.3f}")
            
