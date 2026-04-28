import numpy as np
import sys
from datasets import load_dataset # type: ignore
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support # type: ignore

# roberta being better model
model_name = "roberta-base"

# classification labels prepared
id2label = {0: "HUMAN", 1: "AI"}
label2id = {"HUMAN": 0, "AI": 1}

# load tokenizer for roBERTa model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# tokenize the text with tokenizer of the model
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )


# compute evaluations directly with pre-made function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    # we want to take 4 inputs: this program, train file, development file, test file, result file(or the output file)
    if len(sys.argv)!=5:
        print("Usage: python Fine_tuned_model.py train_file_name.json development_file_name.json test_file_name.json output_file")
        sys.exit(1) # we exit with error
    
    # load dataset with arguments(the filenames specific for those train, development, and test purpose)
    dataset = load_dataset(
        "json",
        data_files={
            "train": sys.argv[1],
            "validation": sys.argv[2],
            "test": sys.argv[3]
        }
    )
    
    # tokenize every sample in the dataset
    tokenized_dataset = dataset.map(tokenize, batched=True) # for purpose of process faster, we use "batched"

    # keep only the necessary columns as our model inputs
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset["train"].column_names
        if col not in ["input_ids", "attention_mask", "label"]]
    )

    # and we convert the remaining columns in dataset into the format of PyTorch (tensors)
    tokenized_dataset.set_format("torch")
    
    # create model with already made function for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    # get the training arguments needed for our model(roBERTa)
    training_args = TrainingArguments(
        output_dir="./bert-ai-detector",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs",
        report_to="none"
    )

    # Now put our prepared ingredients into the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train the model!
    trainer.train()

    # test the result
    test_results = trainer.evaluate(tokenized_dataset["test"])

    # get predictions
    pred_output = trainer.predict(tokenized_dataset["test"])

    logits = pred_output.predictions
    preds = np.argmax(logits, axis=-1)
    
    # output the result
    with open(sys.argv[4],"w") as out:
        for p in preds:
            out.write(str(p) + "\n")

    trainer.save_model("./final-bert-ai-detector")
    tokenizer.save_pretrained("./final-bert-ai-detector")

if __name__ == "__main__":
    main()
