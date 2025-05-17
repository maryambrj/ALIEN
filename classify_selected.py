import os
import os.path as op
import time
import json
import random
random.seed(123)
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
from watermark import watermark
from datasets import load_from_disk
from transformers import AutoTokenizer, RobertaForSequenceClassification, AutoConfig
from transformers import get_linear_schedule_with_warmup
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb
from dotenv import load_dotenv

cache_dir = "./cache_dir"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

os.environ["TOKENIZERS_PARALLELISM"] = 'false'

wandb.login(key="REMOVED")
wandb.init(
    project='active_learning',
    name='roberta_base'
)

wandb_logger = WandbLogger()

E1_START_TOKEN, E1_END_TOKEN = '<e1>' , '</e1>'
E2_START_TOKEN, E2_END_TOKEN = '<e2>' , '</e2>'

label_to_label_id = {
    'no_relation': 0, 'headquartered_in': 1, 'formed_in': 2, 'title': 3, 'shares_of': 4, 
    'loss_of': 5, 'acquired_on': 6, 'agreement_with': 7, 'operations_in': 8, 'subsidiary_of': 9,'employee_of': 10, 'attended': 11,
    'cost_of': 12, 'acquired_by': 13, 'member_of': 14, 'profit_of': 15, 'revenue_of': 16, 'founder_of': 17, 'formed_on': 18,
}

wandb_logger.experiment.config["label_to_id"] = label_to_label_id

BATCH_SIZE = 16
NUM_EPOCHS = 10
WARMUP_STEPS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE = 'roberta-base'

MODEL_CONFIG = AutoConfig.from_pretrained(MODEL_TYPE)
MODEL_CONFIG.num_labels = 19

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_TYPE)
TOKENIZER.add_tokens([E1_START_TOKEN, E1_END_TOKEN, E2_START_TOKEN, E2_END_TOKEN])

MAX_LENGTH = 256 # TOKENIZER.model_max_length

wandb_logger.experiment.config["num_epochs"] = NUM_EPOCHS
wandb_logger.experiment.config["max_length"] = MAX_LENGTH
wandb_logger.experiment.config["warmup_steps"] = WARMUP_STEPS

def data_collator(batch, padding_token_id=TOKENIZER.pad_token_id):
    input_ids = [item["input_ids"][:MAX_LENGTH] for item in batch]
    attention_masks = [item["attention_mask"][:MAX_LENGTH] for item in batch]
    label = [item["label"] for item in batch]
   
    max_len = min(MAX_LENGTH, max(len(ids) for ids in input_ids))
    input_ids = torch.tensor([ids + [padding_token_id] * (max_len - len(ids)) for ids in input_ids])
    attention_masks = torch.tensor([masks + [padding_token_id] * (max_len - len(masks)) for masks in attention_masks])
    label = torch.tensor([i for i in label])
    
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_masks, 
        "labels": label,
    }

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, t_total, warmup_steps):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.model = model
        
        self.t_total = t_total
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()
        
        self.train_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=19) 
        self.val_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=19) 
        self.test_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=19)
        self.val_f1_mi = torchmetrics.F1Score(task="multiclass", average='micro', num_classes=19) 
        self.test_f1_mi = torchmetrics.F1Score(task="multiclass", average='micro', num_classes=19) 
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        # print(batch.keys())
        outputs = self(
            **batch
        )
        self.log("train_loss", outputs["loss"], prog_bar=True)
        
        with torch.no_grad():
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, -1)
            self.train_f1(predicted_labels, batch['labels'])
            self.log("train_f1", self.train_f1, on_epoch=True, on_step=True)
            
        return outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            **batch
        )
        self.log("val_loss", outputs["loss"], prog_bar=True)
        
        logits = outputs['logits']
        predicted_labels = torch.argmax(logits, -1)
        self.val_f1(predicted_labels, batch['labels'])
        self.val_f1_mi(predicted_labels, batch['labels'])
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_f1_micro", self.val_f1_mi, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        outputs = self(
            **batch
        )
        logits = outputs['logits']
        predicted_labels = torch.argmax(logits, -1)
        self.test_f1(predicted_labels, batch['labels'])
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)
        
        self.test_f1_mi(predicted_labels, batch['labels'])
        self.log("test_f1_micro", self.test_f1_mi, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.trainer.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1 # self.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.trainer.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps,
            num_training_steps=self.t_total
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]

dataset_directory = "./processed_data"
dataset = load_from_disk(dataset_directory)

tokenized_dataset = dataset.map(
    lambda examples: TOKENIZER(
        examples['text'],
        padding=True,
        max_length=MAX_LENGTH,
    ), 
    batched=True
)
all_columns = tokenized_dataset['train'].column_names
columns_to_keep = ['input_ids', "attention_mask", 'label']
tokenized_dataset = tokenized_dataset.remove_columns([col for col in all_columns if col not in columns_to_keep])

# Load selected indices from file
with open("./selection_outputs/two_stage_chunk/phase1_yes.txt", "r") as f:
    selected_indices = [int(line.strip()) for line in f.readlines()]
subset_train_dataset = tokenized_dataset['train'].select(selected_indices)

train_loader = DataLoader(
    dataset=subset_train_dataset,
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=8,
    drop_last=True,
    collate_fn=data_collator
)

dev_loader = DataLoader(
    dataset=tokenized_dataset['validation'],
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=8,
    drop_last=False,
    collate_fn=data_collator
)
test_loader = DataLoader(
    dataset=tokenized_dataset['test'],
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=8,
    drop_last=False,
    collate_fn=data_collator
)

T_TOTAL = int(len(train_loader) * NUM_EPOCHS)
wandb_logger.experiment.config["t_total"] = T_TOTAL

print("Torch CUDA Available?", torch.cuda.is_available())
torch.manual_seed(123)
wandb_logger.experiment.config["t_total"] = T_TOTAL

model = RobertaForSequenceClassification.from_pretrained(
    MODEL_TYPE, config=MODEL_CONFIG
)
model.resize_token_embeddings(len(TOKENIZER))
print('model_loaded')

wandb_logger.experiment.config["vocab_size"] = len(TOKENIZER)

LEARNING_RATE = 1e-5
wandb_logger.experiment.config["learning_rate"] = LEARNING_RATE

MODEL_SAVE_DIR = "./trained_models/roberta_base"
GRADIENT_CLIP_VALUE = 1.0
wandb_logger.experiment.config["gradient_clip_value"] = GRADIENT_CLIP_VALUE

TOKENIZER.save_pretrained(MODEL_SAVE_DIR)

lightning_model = LightningModel(model, learning_rate=LEARNING_RATE, t_total=T_TOTAL, warmup_steps=WARMUP_STEPS)
print('lightning model loaded')
callbacks = [
    ModelCheckpoint(save_top_k=1, mode="max", monitor='val_f1', dirpath=MODEL_SAVE_DIR, filename="ro_base_best"), # save top 1 model
    EarlyStopping(monitor='val_f1', patience=3, mode='max')
]

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS, callbacks=callbacks, 
    accelerator="gpu", 
    devices=[0],
    accumulate_grad_batches=2,
    precision="16-mixed",
    logger=wandb_logger, 
    log_every_n_steps=100, deterministic=True,
    gradient_clip_val=GRADIENT_CLIP_VALUE
)

start = time.time()
trainer.fit(
    model=lightning_model,
    train_dataloaders=train_loader,
    val_dataloaders=dev_loader,
)
end = time.time()
elapsed = end - start

print(f"Time elapsed : {elapsed/60:.2f} min")

trainer.validate(lightning_model, dataloaders=dev_loader, ckpt_path="best")
print(trainer.ckpt_path)

test_f1 = trainer.test(lightning_model, dataloaders=test_loader)
print(test_f1)

with open(op.join(trainer.logger.log_dir, "test_outputs.text"), "w") as f:
    f.write(f"Time elapsed {elapsed/60:.2f} min\n")
    f.write(f"Test F1 : {test_f1}")