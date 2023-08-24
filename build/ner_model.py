from datasets import load_dataset, load_metric
import ast
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "" # path or model name .there is using "bert-base-chinese"
batch_size = 16
dataset = load_dataset("../data/chinese_biomedical_NER_dataset")
tag_set = [
    'B_手术',
    'I_疾病和诊断',
    'B_症状',
    'I_解剖部位',
    'I_药物',
    'B_影像检查',
    'B_药物',
    'B_疾病和诊断',
    'I_影像检查',
    'I_手术',
    'B_解剖部位',
    'O',
    'B_实验室检验',
    'I_症状',
    'I_实验室检验'
]

tag2id = lambda tag: tag_set.index(tag)
id2tag = lambda id: tag_set[id]

x = '[ "A","B","C" , " D"]'
x = ast.literal_eval(x)

def tokenize_and_align_labels(example):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_inputs = tokenizer(example["sequences"], truncation=True, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids()

    labels = []

    for i in word_ids:
        if i is None:
            labels.append(-100)
        else:
            labels.append(example['tag_ids'][i])

    tokenized_inputs['labels'] = labels

    return tokenized_inputs

def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [tag_set[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_set[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(tag_set))

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False,remove_columns=dataset['train'].column_names)
    model_name = model_checkpoint

    args = TrainingArguments(
        f"{model_name}-finetuned-NER-biomedical",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=100,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='recall',
        save_total_limit=3,
        push_to_hub=False
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,

    )
    trainer.train()
    trainer.evaluate()

    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [tag_set[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tag_set[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    model.save_pretrained("./")  # 保持到当前目录
    tokenizer.save_pretrained("./")
