from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer
from config import *

def preprocess():
    print('The preprocessing of data is starting...')

    dataset = load_dataset('csv', data_files=str(RAW_DATA_DIR/RAW_DATA_FILE))['train']

    dataset = dataset.remove_columns(['cat'])
    dataset = dataset.filter(lambda x: x['review'] is not None)
    
    dataset = dataset.cast_column('label', ClassLabel(names=['n', 'p']))
    dataset_dict = dataset.train_test_split(test_size=0.2, stratify_by_column='label')

    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_DIR/BERT_MODEL)

    def batch_encode(example):
        inputs = tokenizer(
            example['review'],
            padding='max_length',
            max_length=128,
            truncation=True
        )
        inputs['labels'] = example['label']
        return inputs

    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['label', 'review'])
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)

    print('The preprocessing of data is done.')

if __name__ == '__main__':
    preprocess()
