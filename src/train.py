import torch
from torch import nn, optim
from tqdm import tqdm

from config import *
from dataset import get_dataloader
from transformers import AutoModelForSequenceClassification

from torch.utils.tensorboard import SummaryWriter
import time

def train_one_epoch(model, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for batch in tqdm(train_loader, desc='Training'):
        inputs = {k:v.to(device) for k,v in batch.items()}
        outputs = model(**inputs)
        loss_value = outputs.loss
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss_value.item()

    return total_loss / len(train_loader)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_dataloader(train=True)

    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL).to(device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print('='*15, f'EPOCH {epoch+1}', '='*15)
        this_loss = train_one_epoch(model, train_loader, optimizer, device)
        print('the loss of this epoch is : ', this_loss)

        writer.add_scalar('loss', this_loss, epoch + 1)

        if this_loss < min_loss:
            min_loss = this_loss
            model.save_pretrained(MODEL_DIR)
            print('The best model has been saved!')
        else:
            print('This model is not the best, and it has not been saved!')
    writer.close()

if __name__ == '__main__':
    train()