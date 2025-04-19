
import nltk
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn import Parameter
import math
from tqdm import tqdm
import torch.nn.init as init
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

import numpy as np
nltk.download('punkt')  # Download the necessary tokenizer data

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}

#This is for PDTB2_Level2 labels -11 labels
label_list_2 = ['Temporal.Asynchronous', 'Temporal.Synchrony', 'Contingency.Cause',
                'Contingency.Pragmatic cause', 'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement',
                'Expansion.Alternative','Expansion.List']

#This is for PDTB3_Level2 labels -14 labels
label_list_3 = ['Temporal.Asynchronous', 'Temporal.Synchronous', 'Contingency.Cause',
                'Contingency.Cause+Belief', 'Contingency.Condition', 'Contingency.Purpose',
                'Comparison.Contrast', 'Comparison.Concession',
                'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Equivalence',
                'Expansion.Level-of-detail', 'Expansion.Manner', 'Expansion.Substitution']

label_list = label_list_3
# Function to segment paragraphs into sentences
def segment_paragraphs_into_sentences(paragraphs):
    sentences = []
    for paragraph in paragraphs:
        # Tokenize paragraph into sentences
        paragraph_sentences = nltk.sent_tokenize(paragraph)[:128]
        sentences.append(paragraph_sentences)
    return sentences

 
def read_file_lines(file_path):
    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip()) 
    return lines

def tokenize_text(texts, tokenizer, model_PDTB,
                                cls_token_at_end=False, pad_on_left=False,
                                cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', pad_token=0,
                                sequence_a_segment_id=0, sequence_b_segment_id=1,
                                cls_token_segment_id=1, pad_token_segment_id=0,
                                mask_padding_with_zero=True,
                                vector_ops=False):
    """
    Tokenize the given texts from paragraphs into sentences.

    Args:
    - texts (List[List[str]]): List of lists containing the text with shape [n_para, n_sent].

    Returns:
    - torch.Tensor: Tokenized texts with padding applied.
    - torch.Tensor: Tokenized texts with masks
    - torch.Tensor: Predicted PDTB code from pre-trained models
    - torch.Tensor: PDTB mask
    """
    model_PDTB.eval()
    max_para_len = 64  # Maximum number of sentences in any paragraph, this are different for WP (128) and LFQA(64)
    max_seq_length = 64  # Max sentence length, as the default for PDTB_Bert is 128, we have 2 sentences, and <CLS> & <SEP>

    # Pre-allocate list for all tokenized paragraphs
    tokenized_texts = []
    text_mask = []
    PDTB_code = []
    PDTB_mask = []
    for para in texts:
        if not para:
            continue
        
        if len(para) > max_para_len: #truncate if necessary
            para = para[:max_para_len]
        # Tokenize and pad/truncate sentences within each paragraph
        '''
        para_tokens = [tokenizer(sent, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')['input_ids'][0] for sent in para]
        
        # Pad paragraphs to have the same number of sentences
        para_len = len(para_tokens)
        if para_len < max_para_len:
            pad_sentences = [torch.tensor([tokenizer.pad_token_id] * max_length) for _ in range(max_para_len - para_len)]
            para_tokens.extend(pad_sentences)
        
        tokenized_texts.append(torch.stack(para_tokens))
        '''
        tokenized_texts_para = []
        tokenized_texts_para_mask = []
        PDTB_sent = []
        token_a = None
        for sent in para:
            
            if not token_a: #Set -2 as discourse code for the first sentence 
                PDTB_sent.append(-2)
                token_a = tokenizer.tokenize(sent)
                
                while(len(token_a) > max_seq_length - 2):
                    token_a.pop()
                
                token_a = [cls_token] + token_a + [sep_token]
                id_a = tokenizer.convert_tokens_to_ids(token_a)
                input_mask = [1 if mask_padding_with_zero else 0] * len(id_a)
                padding_length = max_seq_length - len(id_a)
                input_ids = id_a + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                tokenized_texts_para.append(input_ids)
                tokenized_texts_para_mask.append(input_mask)
            else:
                token_b = tokenizer.tokenize(sent)
                
                while(len(token_b) > max_seq_length - 2):
                    token_b.pop()
                
                token_b = token_b + [sep_token]
                id_b = tokenizer.convert_tokens_to_ids([cls_token] + token_b)
                input_mask = [1 if mask_padding_with_zero else 0] * len(id_b)
                padding_length = max_seq_length - len(id_b)
                input_ids = id_b + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                tokenized_texts_para.append(input_ids)
                tokenized_texts_para_mask.append(input_mask)
                
                # now to predict the PDTB code
                tokens = token_a + token_b 
                segment_ids = [sequence_a_segment_id] * len(token_a)
                segment_ids += [sequence_b_segment_id] * len(token_b)

                input_pdtb = tokenizer.convert_tokens_to_ids(tokens)
                input_pdtb_mask = [1 if mask_padding_with_zero else 0] * len(input_pdtb)
                padding_length_pdtb = max_seq_length * 2 - len(input_pdtb)
                input_pdtb = input_pdtb + ([pad_token] * padding_length_pdtb)
                input_pdtb_mask = input_pdtb_mask + ([0 if mask_padding_with_zero else 1] * padding_length_pdtb)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length_pdtb)

                inputs = {'input_ids':     torch.tensor([input_pdtb]).to(device),
                          'attention_mask': torch.tensor([input_pdtb_mask]).to(device),
                          'token_type_ids': torch.tensor([segment_ids]).to(device),  # XLM don't use segment_ids
                          'labels':         torch.tensor([[0]]).to(device)} #This is evaluation so no labels
                model_PDTB.eval()
                outputs = model_PDTB(**inputs)
                _, logits = outputs[:2]
                preds = logits.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                
                PDTB_sent.append(preds[0])
        
        pdtb_sent_mask = [1 if mask_padding_with_zero else 0] * len(PDTB_sent)
        padding_PDTB_sen = max_para_len - len(PDTB_sent)
        PDTB_sent = PDTB_sent + ([-1] * padding_PDTB_sen) # -1 is the padding as 0 is one label.
        PDTB_code.append(PDTB_sent)
        
        PDTB_para_mask = pdtb_sent_mask + ([0 if mask_padding_with_zero else 1] * padding_PDTB_sen)
        PDTB_mask.append(PDTB_para_mask)
        
        
        # Pad paragraphs to have the same number of sentences
        para_len = len(tokenized_texts_para)
        if para_len < max_para_len:
            pad_sentences = [[pad_token]* max_seq_length for _ in range(max_para_len - para_len)]
            pad_sentences_mask = [[0 if mask_padding_with_zero else 1] * max_seq_length for _ in range(max_para_len - para_len)]

            tokenized_texts_para.extend(pad_sentences)
            tokenized_texts_para_mask.extend(pad_sentences_mask)
        tokenized_texts.append(tokenized_texts_para)
        text_mask.append(tokenized_texts_para_mask)

    # Make the tokenized_texts a batch dimension [n_para, max_para_len, max_length]
    return torch.tensor(tokenized_texts), torch.tensor(text_mask), torch.tensor(PDTB_code), torch.tensor(PDTB_mask)

train_path_MGC='/data/yl7622/MRes/M4/my_data/train.MGC'
train_path_HPC='/data/yl7622/MRes/M4/my_data/train.HPC'

#train_path_MGC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/train.MGC'
#train_path_HPC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/train.HPC'
train_texts_MGC = read_file_lines(train_path_MGC) # List of input texts
train_texts_HPC = read_file_lines(train_path_HPC)
train_texts_MGC = segment_paragraphs_into_sentences(train_texts_MGC)
train_texts_HPC = segment_paragraphs_into_sentences(train_texts_HPC)


valid_path_MGC='/data/yl7622/MRes/M4/my_data/valid.MGC'
valid_path_HPC='/data/yl7622/MRes/M4/my_data/valid.HPC'
#valid_path_MGC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/valid.MGC'
#valid_path_HPC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/valid.HPC'
valid_texts_MGC = read_file_lines(valid_path_MGC)  # List of input texts
valid_texts_HPC = read_file_lines(valid_path_HPC)
valid_texts_MGC = segment_paragraphs_into_sentences(valid_texts_MGC)
valid_texts_HPC = segment_paragraphs_into_sentences(valid_texts_HPC)

#test_path_MGC='/data/yl7622/MRes/DIPPER_data/my_data/test.MGC'
#test_path_HPC='/data/yl7622/MRes/DIPPER_data/my_data/test.HPC'

#test_path_MGC='/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/revised_MGC'

#test_path_MGC='/data/yl7622/MRes/plegbench_data/MGC.test'
#test_path_HPC='/data/yl7622/MRes/plegbench_data/HPC.test'

test_path_MGC='/data/yl7622/MRes/M4/my_data/test.MGC'
test_path_HPC='/data/yl7622/MRes/M4/my_data/test.HPC'

#test_path_MGC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/test.MGC'
#test_path_HPC = '/data/yl7622/MRes/Transformer_human/fairseq/data/fairseq_data/processed_data/test.HPC'
test_texts_MGC = read_file_lines(test_path_MGC)  # List of input texts
test_texts_HPC = read_file_lines(test_path_HPC)
test_texts_MGC = segment_paragraphs_into_sentences(test_texts_MGC)
test_texts_HPC = segment_paragraphs_into_sentences(test_texts_HPC)


task_name_2 = 'PDTB2_LEVEL2'
task_name_3 = 'PDTB3_LEVEL2'

task_name = task_name_3
num_labels = len(label_list)
model_name_or_path = '/data/yl7622/MRes/pdtb3/scripts/output/pdtb3/fold_3'
#model_name_or_path = '/data/yl7622/MRes/pdtb3/scripts/output/fold_2'
config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
config = config_class.from_pretrained(model_name_or_path, num_labels=num_labels, finetuning_task=task_name)
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
model = model_class.from_pretrained(model_name_or_path, from_tf=bool('.ckpt' in model_name_or_path), config=config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print("starting PDTB_predicting")

valid_text, valid_mask, valid_PDTB, valid_PDTB_mask = tokenize_text(valid_texts_MGC, tokenizer, model)
torch.save(valid_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/MGC.pt')
torch.save(valid_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/MGC_mask.pt')
torch.save(valid_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/MGC_PDTB.pt')
torch.save(valid_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/MGC_PDTB_mask.pt')

valid_text, valid_mask, valid_PDTB, valid_PDTB_mask = tokenize_text(valid_texts_HPC, tokenizer, model)
torch.save(valid_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/HPC.pt')
torch.save(valid_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/HPC_mask.pt')
torch.save(valid_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/HPC_PDTB.pt')
torch.save(valid_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/valid/HPC_PDTB_mask.pt')
print("finish valid set")

test_text, test_mask, test_PDTB, test_PDTB_mask = tokenize_text(test_texts_MGC, tokenizer, model)


torch.save(test_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/MGC.pt')
torch.save(test_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/MGC_mask.pt')
torch.save(test_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/MGC_PDTB.pt')
torch.save(test_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/MGC_PDTB_mask.pt')

test_text, test_mask, test_PDTB, test_PDTB_mask = tokenize_text(test_texts_HPC, tokenizer, model)
torch.save(test_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/HPC.pt')
torch.save(test_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/HPC_mask.pt')
torch.save(test_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/HPC_PDTB.pt')
torch.save(test_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/test/HPC_PDTB_mask.pt')
print("finish test set.")

train_text, train_mask, train_PDTB, train_PDTB_mask = tokenize_text(train_texts_MGC, tokenizer, model)
torch.save(train_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/MGC.pt')
torch.save(train_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/MGC_mask.pt')
torch.save(train_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/MGC_PDTB.pt')
torch.save(train_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/MGC_PDTB_mask.pt')

train_text, train_mask, train_PDTB, train_PDTB_mask = tokenize_text(train_texts_HPC, tokenizer, model)
torch.save(train_text, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/HPC.pt')
torch.save(train_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/HPC_mask.pt')
torch.save(train_PDTB, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/HPC_PDTB.pt')
torch.save(train_PDTB_mask, '/data/yl7622/MRes/discourse_attention_torch/data_encoding/M4_3/train/HPC_PDTB_mask.pt')
print("finish training set")
