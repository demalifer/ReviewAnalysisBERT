import torch
from config import *
from transformers import AutoModelForSequenceClassification
from dataset import get_dataloader
from predict import predict_batch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    correct_count = 0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating: '):
            labels = batch.pop('labels').tolist()
            inputs = {k: v.to(device) for k, v in batch.items()}
            batch_results = predict_batch(model, inputs)
            for target, result in zip(labels, batch_results):
                total_count += 1
                result = 1 if result > 0.5 else 0
                if result == target:
                    correct_count += 1
    correct_rate = correct_count / total_count
    return correct_rate

def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    print('model load success!')

    test_dataloader = get_dataloader(train=False)
    correct_rate = evaluate(model, test_dataloader, device)

    print('evaluate result:')
    print('correct rate: ', correct_rate)

if __name__ == '__main__':
    run_evaluate()
