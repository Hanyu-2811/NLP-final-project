Fine_tuned_model_explain

Functions:
1) tokenize(batch):
    use the tokenizer function inherent from model package to tokenize the inputs
2) compute_metrics(eval_prod):
    compute precision, recall, f1 score with the model evaluation functions
3) main():
    # prepare steps
    get arguments to process for dataset;
    tokenize the inputs from those arguments(train file);
    make the model by fitting the model with the classification function using our own label classifiers;
    # real training and result steps
    make the training arguments with training argument function;
    put the training argument and model related information into trainer;
    train the model;
    test model and save

Terminal line format:
1) First navigate to the current folder the files are directly in
2) input "python Fine_tuned_model.py" "train_file_info_name.json" "development_file_info_name.json output_file" "test_file_info_name.json""
    For instance: "python Fine_tuned_model.py train.json development.json test.json output_prac.txt"
