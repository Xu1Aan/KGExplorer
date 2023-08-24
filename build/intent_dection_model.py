from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from datasets import ClassLabel
import torch
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from datasets import load_metric
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# 模型设置
num_labels = 13  # type of intent
model_ckpt = "bert-base-chinese"  # path or model name. there is using "bert-base-chinese"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_set = [
    '定义',
    '病因',
    '预防',
    '临床表现(病症表现)',
    '相关病症',
    '治疗方法',
    '所属科室',
    '传染性',
    '治愈率',
    '禁忌',
    '化验/体检方案',
    '治疗时间',
    '其他'
]

dataset = load_dataset("../data/intent-recognition-biomedical")
dataset = dataset.shuffle()

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)


def preprocess(sample):
    input = tokenizer(sample['text'], truncation=True)
    input['label'] = sample['label']
    return input


def compute_metrics(pred):
    logits, labels = pred
    '''
    logits: N,L,D
    labels: N,L
    '''
    predictions = np.argmax(logits, axis=-1)  # N,L
    f1 = f1_score(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1, 'recall': recall, 'precision': precision}


def main():
    tokenized_datasets = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    feat_sentiment = ClassLabel(num_classes=len(label_set), names=label_set)
    tokenized_datasets = tokenized_datasets.cast_column("label", feat_sentiment)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        num_train_epochs=100,
        save_total_limit=3,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        weight_decay=0.01,
        output_dir=f'{model_ckpt}-finetuned-intent_recognition-biomedical'
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    model.save_pretrained("./")  # 保持到当前目录
    tokenizer.save_pretrained("./")
