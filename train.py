import torch
import os
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset 
import numpy as np
import random
from tqdm.auto import tqdm
import argparse
from transformers import (
  AutoTokenizer,
  AutoModelForQuestionAnswering,
)
import wandb
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
import json

parser = argparse.ArgumentParser(description='undergraduate')

parser.add_argument('--save-dir', metavar='DIR', nargs='?', default="./save_dir",
                        help='path to save model')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
parser.add_argument('--data-dir', default="dataset/", type=str, metavar='S',
                        help='Path to the dataset')
parser.add_argument('--train-batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total ')
parser.add_argument('--validation', default=True, type=bool,
                    metavar='V',
                    help='interval for logging ')
parser.add_argument('--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--model-name', default="bert-base-chinese", type=str,
                    metavar='M', help='Model used in this project')
parser.add_argument('--logging-step', metavar='LOGS', nargs='?', default=100,
                    )
parser.add_argument('--gradient-accumulation-steps', nargs='?', default=16,
                    )
parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    type=torch.device, help='Train with gpu or cpu.' )
parser.add_argument('--fp16-training', default=True,
                    type=bool, help='Use fp16 to accerlate the training' )
parser.add_argument('--model-save-dir', default="./save_model",
                    type=str, help='Use save the model path' )
parser.add_argument('--use-wandb', default=False,
                    type=bool, help='Use wandb to plot the logging' )
args = parser.parse_args()


def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
			torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

class QA_Dataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 60
        self.max_paragraph_len = 150
        
        ##### TODO: Change value of doc_stride #####
        # self.doc_stride = 150
        self.doc_stride = 75

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])

            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            
            # paragraph_start = max(0, min(mid - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            # paragraph_end = paragraph_start + self.max_paragraph_len

            max_offset = self.max_paragraph_len / 4   # We allow up to 1/4 of the max length as offset
            random_offset = np.random.randint(-max_offset, max_offset)  # Random shift between -max_offset and +max_offset

            # Adjust paragraph start based on random offset
            paragraph_start = max(0, min(mid + random_offset - self.max_paragraph_len // 2, len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len

            # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102] 
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]		
            
            # Convert answer's start/end positions in tokenized_paragraph to start/end positions in the window  
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            
            # Pad sequence and obtain inputs to model 
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        # Validation/Testing
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i : i + self.max_paragraph_len] + [102]
                
                # Pad sequence and obtain inputs to model
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        
        return input_ids, token_type_ids, attention_mask



def train_epoch(model, train_loader, optimizer):
    model.to(args.device)
    model.train()
    step = 1
    train_loss = train_acc = 0
    for data in tqdm(train_loader):	
        with accelerator.accumulate(model):
            
            # Load all data into GPU
            data = [i.to(args.device) for i in data]
            
            # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
            # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)  
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3], end_positions=data[4])
            # Choose the most probable start position / end position
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
                
            train_loss += output.loss
            
            accelerator.backward(output.loss)
            
            step += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        ##### TODO: Apply linear learning rate decay #####

        # Print training loss and accuracy over past logging step
        if step % args.logging_step == 0:
                if args.use_wandb:
                    wandb.log({
                    "train/loss": train_loss.item() / args.logging_step,
                    "train/acc": train_acc / args.logging_step,
                })
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / args.logging_step:.3f}, acc = {train_acc / config.logging_step:.3f}")
                train_loss = train_acc = 0

    
    return model

def evaluate(data, output):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    
    for k in range(num_of_windows):
        # Obtain answer by choosing the most probable start position / end position
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        
        # Replace answer if calculated probability is larger than previous windows
        #fix the bug 
        if start_index <= end_index:
            if prob > max_prob:
                max_prob = prob
                # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
                answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
        
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(' ','')

def validate_epoch(model,dev_loader):
    
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dev_loader)):
            output = model(input_ids=data[0].squeeze(dim=0).to(args.device), token_type_ids=data[1].squeeze(dim=0).to(args.device),
                    attention_mask=data[2].squeeze(dim=0).to(args.device))
            # prediction is correct only if answer text exactly matches
            dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
        if args.use_wandb: 
            wandb.log({
                "Validation/acc": dev_acc / len(dev_loader),
            })
        print(f"Validation | Epoch {args.epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")
    


if __name__ == '__main__':

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_questions, train_paragraphs = read_data(os.path.join(args.data_dir,"train.json"))
    dev_questions, dev_paragraphs = read_data(os.path.join(args.data_dir,"dev.json"))
    test_questions, test_paragraphs = read_data(os.path.join(args.data_dir,"test.json"))

    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_questions], add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_questions], add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_questions], add_special_tokens=False) 

    train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_paragraphs, add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(test_paragraphs, add_special_tokens=False)

    train_set = QA_Dataset("train", train_questions, train_questions_tokenized, train_paragraphs_tokenized)
    dev_set = QA_Dataset("dev", dev_questions, dev_questions_tokenized, dev_paragraphs_tokenized)
    test_set = QA_Dataset("test", test_questions, test_questions_tokenized, test_paragraphs_tokenized)
    
    same_seeds(2)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)

    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.2 * total_steps)  # Set warmup steps to 20% of total steps
    # [Hugging Face] Apply linear learning rate decay with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    # Change "fp16_training" to True to support automatic mixed 
    # precision training (fp16)	
    if args.fp16_training:    
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    if args.use_wandb:
        wandb.init(project="~", name=args.savedir, config=args)

    # Documentation for the toolkit:  https://huggingface.co/docs/accelerate/
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler) 




    
    for epoch in range(args.epochs): 
        print("Start Training ...")
        model.to(args.device)
        model.train()
        #train the model
        model = train_epoch(model,train_loader,optimizer)
        #validate the model
        print("Evaluating Dev Set ...")
        model = validate_epoch(model, dev_loader)

        #save the model to dir
        print("Saving Model ...") 
        model.save_pretrained(args.model_save_dir)
    
    if args.use_wandb:
         wandb.finish()
