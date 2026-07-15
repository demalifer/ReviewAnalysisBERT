import torch
import torch.nn as nn

from config import *

from transformers import AutoModel

class ReviewAnalysisModel(nn.Module):
    def __init__(self):
        super(ReviewAnalysisModel, self).__init__()
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_DIR/BERT_MODEL)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        cls_hidden_stat = output.pooler_output
        result = self.linear(cls_hidden_stat).squeeze(-1)
        return result

if __name__ == '__main__':
    model = ReviewAnalysisModel()
    print(model)
