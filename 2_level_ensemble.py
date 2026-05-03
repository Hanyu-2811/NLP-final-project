import numpy as np
from pathlib import Path
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
import torch
import torch.nn.functional as F
# import the other baseline models
from TFIDF import ModelA
from perplexity import ModelB
# import for the pipelined models - including TFIDF, Signal Featured, and RoBERTa on different datasets
from pipelined_models.baseline_pipelined import SignalFeatureExtractor, TextExtractor, datasets, make_tfmodel, make_sigmodel, run_experiments
# import for the roberta model
from pipelined_models.roberta_experiments import AIDetectionDataset, roberta_datasets, run_roberta_experiments

# voter imports
import tensorflow as tf

# add for baseline path as well
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

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

# this builds a model to learn how much percentage assign for each model - the "voter"
# n represent the number of models for ensemble
def vote_machine(n):
    # build voter
    voter = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8,activation="relu"),
        tf.keras.layers.Dropout(0.3),# avoid overfitting
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    
    #compile
    voter.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )
    
    return voter

#build voter for binary ensembling
def binary_vote():
    voter = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        # tf.keras.layers.Dense(10, activation="relu"),
        # tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8,activation="relu"),
        tf.keras.layers.Dropout(0.3),# avoid overfitting
        tf.keras.layers.Dense(1,activation="sigmoid")
    ])
    
    #compile
    voter.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )
    
    return voter

# make it easy to predict probability
def roberta_predict_prob(trainer, tokenizer, raw_data):
    texts = [x["text"] for x in raw_data]
    labels = [int(x["label"]) for x in raw_data]
    ids = [x.get("id", str(i)) for i, x in enumerate(raw_data)]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512
    )

    dataset = AIDetectionDataset(encodings, labels, ids)

    output = trainer.predict(dataset)
    logits = output.predictions

    probs = F.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return probs

"""ensemble for 9 models:
1) tfidf for experiment a,b,c
2) signal feature for experiment a,b,c
3) roberta for experiment 1,2,3
dump the voting machine at the end
"""
def main():
        # we want to take 5 inputs: this program, shared train file, shared test file, result file for word output, result file for evalution
    if len(sys.argv)!=5:
        print("Usage: python ensemble.py shared_train_file_name.json shared_test_file_name.json output_file_word output_evaluation")
        sys.exit(1) # we exit with error
    
    """preparations for roberta"""
    # load dataset with arguments(the filenames specific for those train, development, and test purpose)
    dataset = load_dataset(
        "json",
        data_files={
            "train": sys.argv[1],
            "test": sys.argv[2],
        }
    )
    shared_train = dataset["train"]
    shared_test = dataset["test"]
    
    """get datasets and relevent elements for both baseline models and roberta for training and testing"""
    # get pipelined base models' datasets
    base_datasets = datasets()
    
    # get roberta models' datasets' related things/parameters
    # exp1
    hc3_dev, hc3_train, m4_test, results, predictions, m4_train, m4_dev, cg_dev_split, cg_train_split, cg_chatgpt_test, seed1 = roberta_datasets(42)
    # exp2
    hc3_dev2, hc3_train2, m4_test2, results2, predictions2, m4_train2, m4_dev2, cg_dev_split2, cg_train_split2, cg_chatgpt_test2, seed2 = roberta_datasets(43)
    # exp3
    hc3_dev3, hc3_train3, m4_test3, results3, predictions3, m4_train3, m4_dev3, cg_dev_split3, cg_train_split3, cg_chatgpt_test3, seed3 = roberta_datasets(44)
    
    """load corresponding models
    Note: since roberta's model would already included in experiments, only pipelined models here are loaded"""
    tfmodel = make_tfmodel()
    sigmodel = make_sigmodel()
    
    """experiments for both pipelined models and roberta"""
    # base models
    exps, df_a, df_b, df_c, result_dir = run_experiments(tfmodel, sigmodel,base_datasets)
    #exps include all metrics needed for all three experiments
    
    # roberta exps
    robmodel1 = run_roberta_experiments(hc3_dev, hc3_train, m4_test, results, predictions, m4_train, m4_dev, cg_dev_split, cg_train_split, cg_chatgpt_test, seed1)
    robmodel2 = run_roberta_experiments(hc3_dev2, hc3_train2, m4_test2, results2, predictions2, m4_train2, m4_dev2, cg_dev_split2, cg_train_split2, cg_chatgpt_test2, seed2)
    robmodel3 = run_roberta_experiments(hc3_dev3, hc3_train3, m4_test3, results3, predictions3, m4_train3, m4_dev3, cg_dev_split3, cg_train_split3, cg_chatgpt_test3, seed3)
    
    """development - prepare for voter"""
    # tfidf for experiment a,b,c
    tfproba, tfprobb, tfprobc = exps["expa"]["dev"][0],exps["expb"]["dev"][0],exps["expc"]["dev"][0]
    # signal feature for experiment a,b,c
    sigproba, sigprobb, sigprobc = exps["expa"]["dev"][1],exps["expb"]["dev"][1],exps["expc"]["dev"][1]
    
    # total development for both tfidf and signal
    y_deva, y_devb, y_devc = exps["expa"]["y"][1],exps["expb"]["y"][1],exps["expc"]["y"][1]
    
    # roberta for experiment 1,2,3
    trainer1, trainer2, trainer3 = robmodel1["trainer"], robmodel2["trainer"], robmodel3["trainer"]
    robprob1, robprob2, robprob3 = robmodel1["dev_probs"], robmodel2["dev_probs"], robmodel3["dev_probs"]
    roby1, roby2, roby3 = robmodel1["dev_labels"], robmodel2["dev_labels"], robmodel3["dev_labels"]
    
    # x_dev = np.column_stack([
    #     tfproba, tfprobb, tfprobc,
    #     sigproba, sigprobb, sigprobc,
    #     robprob1, robprob2, robprob3
    # ])
    # y_dev = np.column_stack([
    #     y_deva, y_devb, y_devc,
    #     y_deva, y_devb, y_devc,
    #     roby1, roby2, roby3
    # ])
    
    # base models in binary way
    x_deva = np.column_stack([
        tfproba, sigproba
    ])
    y_dev_a = np.array(y_deva)
    
    x_devb = np.column_stack([
        tfprobb, sigprobb
    ])
    y_dev_b = np.array(y_devb)
    
    x_devc = np.column_stack([
        tfprobc, sigprobc
    ])
    y_dev_c = np.array(y_devc)
    
    # binary small ensemblings and corresponding shared train
    ena = binary_vote()
    ena.fit(x_deva, y_dev_a)
    tf_a_shared = exps["expa"]["model"][0].predict_proba(shared_train)[:, 1]
    sig_a_shared = exps["expa"]["model"][1].predict_proba(shared_train)[:, 1]
    ena_prob = ena.predict(np.column_stack([tf_a_shared, sig_a_shared])).ravel()
        
    enb = binary_vote()
    enb.fit(x_devb, y_dev_b)
    tf_b_shared = exps["expb"]["model"][0].predict_proba(shared_train)[:, 1]
    sig_b_shared = exps["expb"]["model"][1].predict_proba(shared_train)[:, 1]
    enb_prob = enb.predict(np.column_stack([tf_b_shared, sig_b_shared])).ravel()
    
    enc = binary_vote()
    enc.fit(x_devc, y_dev_c)
    tf_c_shared = exps["expc"]["model"][0].predict_proba(shared_train)[:, 1]
    sig_c_shared = exps["expc"]["model"][1].predict_proba(shared_train)[:, 1]
    enc_prob = enc.predict(np.column_stack([tf_c_shared, sig_c_shared])).ravel()
    
    # roberta predicts on shared train
    rprob1, rprob2, rprob3 = roberta_predict_prob(trainer1, robmodel1["tokenizer"], shared_train), roberta_predict_prob(trainer2, robmodel2["tokenizer"], shared_train), roberta_predict_prob(trainer3, robmodel3["tokenizer"], shared_train)
    
    # combine for final preparation on main voter
    x_trainall = np.column_stack([
        ena_prob,
        enb_prob,
        enc_prob,
        rprob1,
        rprob2,
        rprob3,
    ])
    y_trainall = [x['label'] for x in shared_train]
    
    """fit voting machine - ensembling all those models"""
    """binary ensembling and whole development"""
    voter = vote_machine(6)
    voter.fit(x_trainall,y_trainall,epochs=20,batch_size=8)
    
    """test"""
    # tfidf for experiment a,b,c
    tfproba, tfprobb, tfprobc = exps["expa"]["test"][0],exps["expb"]["test"][0],exps["expc"]["test"][0]
    # signal feature for experiment a,b,c
    sigproba, sigprobb, sigprobc = exps["expa"]["test"][1],exps["expb"]["test"][1],exps["expc"]["test"][1]
    
    # total test for both tfidf and signal
    y_testa, y_testb, y_testc = exps["expa"]["y"][2],exps["expb"]["y"][2],exps["expc"]["y"][2]

    # roberta for experiment 1,2,3
    robprob1, robprob2, robprob3 = robmodel1["test_probs"], robmodel2["test_probs"], robmodel3["test_probs"]
    roby1, roby2, roby3 = robmodel1["test_labels"], robmodel2["test_labels"], robmodel3["test_labels"]
    

    # base models in binary way
    # x_testa = np.column_stack([
    #     tfproba, sigproba
    # ])
    # y_test_a = np.array(y_testa)
    
    # x_testb = np.column_stack([
    #     tfprobb, sigprobb
    # ])
    # y_test_b = np.array(y_testb)
    
    # x_testc = np.column_stack([
    #     tfprobc, sigprobc
    # ])
    # y_test_c = np.array(y_testc)
    y_shared_test = np.array([int(x["label"]) for x in shared_test])
    
    
    """get voting probability"""
    """first for those binary ensemblings"""
    # test for a
    tf_a_shared = exps["expa"]["model"][0].predict_proba(shared_test)[:, 1]
    sig_a_shared = exps["expa"]["model"][1].predict_proba(shared_test)[:, 1]
    prob_a = ena.predict(np.column_stack([tf_a_shared, sig_a_shared])).ravel()
    pred_a = (prob_a >= 0.5).astype(int)
    
    # test for b
    tf_b_shared = exps["expb"]["model"][0].predict_proba(shared_test)[:, 1]
    sig_b_shared = exps["expb"]["model"][1].predict_proba(shared_test)[:, 1]
    prob_b = enb.predict(np.column_stack([tf_b_shared, sig_b_shared])).ravel()
    pred_b = (prob_b >= 0.5).astype(int)
    
    # test for c
    tf_c_shared = exps["expc"]["model"][0].predict_proba(shared_test)[:, 1]
    sig_c_shared = exps["expc"]["model"][1].predict_proba(shared_test)[:, 1]
    prob_c = enc.predict(np.column_stack([tf_c_shared, sig_c_shared])).ravel()
    pred_c = (prob_c >= 0.5).astype(int)
    
    """then for the main voter of the 3 emsemblings and the 3 robertas"""
    rob1 = roberta_predict_prob(trainer1, robmodel1["tokenizer"], shared_test)
    rob2 = roberta_predict_prob(trainer2, robmodel2["tokenizer"], shared_test)
    rob3 = roberta_predict_prob(trainer3, robmodel3["tokenizer"], shared_test)
    
    # set up the test sets
    x_testall = np.column_stack([
        prob_a,
        prob_b,
        prob_c,
        rob1,
        rob2,
        rob3,
    ])
    y_testall = y_shared_test
    
    # test for the major voter
    prob_vote = voter.predict(x_testall).ravel()
    pred_eventual = (prob_vote >= 0.5).astype(int) # if over half, we assign as label "1"
    
    """compute evaluation if needed"""
    # for the three small binary ensemblings
    precisiona, recalla, f1a, _ = precision_recall_fscore_support(
        y_shared_test, pred_a, average = "binary"
    )
    
    precisionb, recallb, f1b, _ = precision_recall_fscore_support(
        y_shared_test, pred_b, average = "binary"
    )
    
    precisionc, recallc, f1c, _ = precision_recall_fscore_support(
        y_shared_test, pred_c, average = "binary"
    )
    
    # for the large main voter(main ensembling)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_testall, pred_eventual, average = "binary"
    )
    
    """output the result"""
    with open(sys.argv[3],"w") as out:
        for p in pred_eventual:
            out.write(str(p) + "\n")
            
    with open(sys.argv[4],"w") as out:
        out.write("3 binary ensemblings:"+"\n")
        out.write("ensemble a:"+"\n")
        out.write(str(precisiona)+"\n")
        out.write(str(recalla)+"\n")
        out.write(str(f1a)+"\n")
        
        out.write("ensemble b:"+"\n")
        out.write(str(precisionb)+"\n")
        out.write(str(recallb)+"\n")
        out.write(str(f1b)+"\n")
        
        out.write("ensemble c:"+"\n")
        out.write(str(precisionc)+"\n")
        out.write(str(recallc)+"\n")
        out.write(str(f1c)+"\n")
        
        out.write("Main voting machine for 3 binary ensembles and 3 robertas:"+"\n")
        out.write(str(precision)+"\n")
        out.write(str(recall)+"\n")
        out.write(str(f1)+"\n")
    
    voter.save("./voting_machine.keras")

# ensemble for basic version
def main_basic():
    # we want to take 4 inputs: this program, train file, development file, test file, result file(or the output file)
    if len(sys.argv)!=5:
        print("Usage: python Fine_tuned_model.py train_file_name.json development_file_name.json test_file_name.json output_file")
        sys.exit(1) # we exit with error
    
    """preparations for roberta"""
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
        num_train_epochs=3,
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

    # # test the result
    # test_results = trainer.evaluate(tokenized_dataset["test"])

    # get relevant information for two sets: dev and test
    # development parts
    dev_output = trainer.predict(tokenized_dataset["validation"])
    dev_logits = dev_output.predictions
    dev_probs = F.softmax(torch.tensor(dev_logits), dim=1).numpy()[:,1] # probability
    dev_preds = np.argmax(dev_logits, axis=-1)
    # prediction parts
    pred_output = trainer.predict(tokenized_dataset["test"])
    logits = pred_output.predictions
    probs = F.softmax(torch.tensor(logits),dim=1).numpy()[:,1]
    preds = np.argmax(logits, axis=-1)
    
    """ensemble with TF-IDF and perplexity:
    use the sequential nn voter to compute probabilities for each model"""
    # TF-IDF
    tfmodel = ModelA()
    # perplexity plus
    pmodel = ModelB()
    
    # probabilities and predictions from two other models
    probtf, _ = tfmodel.run(dataset["train"],dataset["validation"])
    probp, _ = pmodel.run(dataset["train"],dataset["validation"])
    x_dev = np.column_stack([
        probtf,
        probp,
        dev_probs,
    ])
    y_dev = np.array(tokenized_dataset["validation"]["label"])
    
    # fit voting machine
    voter = vote_machine(3)
    voter.fit(x_dev,y_dev,epochs=20,batch_size=8)
    
    # create for x_test
    #first update for test predictsf
    probtf, predtf = tfmodel.predict(dataset["test"])
    probp, predp = pmodel.predict(dataset["test"])
    x_test = np.column_stack([
        probtf,
        probp,
        probs
    ])
    y_test = np.array(tokenized_dataset["test"]["label"])
    
    # get the voting probability
    prob_vote = voter.predict(x_test).ravel()
    pred_eventual = (prob_vote >= 0.5).astype(int) # if over half, we assign as label "1"
    
    """compute evalution if needed"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred_eventual, average = "binary"
    )
    accuracy = accuracy_score(y_test, pred_eventual)
    
    """output the result"""
    with open(sys.argv[4],"w") as out:
        for p in pred_eventual:
            out.write(str(p) + "\n")

    trainer.save_model("./final-bert-ai-detector")
    tokenizer.save_pretrained("./final-bert-ai-detector")

if __name__ == "__main__":
    main()
