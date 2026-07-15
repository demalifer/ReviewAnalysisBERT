from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

# 1.数据
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"
PRE_TRAINED_DIR = ROOT_DIR / "pretrained"

#2. 文件
RAW_DATA_FILE = 'online_shopping_10_cats.csv'
BEST_MODEL = 'best_model.pt'
BERT_MODEL = 'bert_base_chinese'

#3. 超参数
SEQ_LEN = 128
BATCH_SIZE = 64
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 768

LEARNING_RATE = 1e-5
EPOCHS = 30