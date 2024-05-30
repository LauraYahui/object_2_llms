import argparse
from transformers import (
    DistilBertForSequenceClassification,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DistilBertTokenizer,
    T5Tokenizer,
)
from datasets import load_dataset, load_metric
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for information retrieval or summarization")
    parser.add_argument("--task", type=str, choices=["retrieval", "summarization"], required=True, help="Task to perform")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    return parser.parse_args()

def compute_metrics_retrieval(p):
    predictions = np.argmax(p.predictions, axis=1)
    return {"accuracy": (predictions == p.label_ids).astype(np.float32).mean().item()}

def compute_metrics_summarization(pred):
    rouge = load_metric("rouge")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    
    return {
        "rouge2_precision": rouge_output.precision,
        "rouge2_recall": rouge_output.recall,
        "rouge2_fmeasure": rouge_output.fmeasure,
    }

def main():
    args = parse_args()

    if args.task == "retrieval":
        model_name = args.model_name_or_path or "distilbert-base-uncased"
        model = DistilBertForSequenceClassification.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        dataset = load_dataset("ms_marco", "v2.1")
    elif args.task == "summarization":
        model_name = args.model_name_or_path or "t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        dataset = load_dataset("samsum")

    def preprocess_function(examples):
        if args.task == "retrieval":
            inputs = examples["query"]
            targets = [item['passage_text'][0] for item in examples["passages"]]  
        elif args.task == "summarization":
            inputs = examples["dialogue"]
            targets = examples["summary"]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")['input_ids']
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    if args.task == "retrieval":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_retrieval,
        )
    elif args.task == "summarization":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_summarization,
        )

    trainer.train()

    # Evaluate the model
    print("Evaluating model...")
    eval_result = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    if args.task == "summarization":
        print(f"ROUGE Scores: {eval_result}")
    elif args.task == "retrieval":
        print(f"Accuracy: {eval_result['eval_accuracy']}")

if __name__ == "__main__":
    main()
