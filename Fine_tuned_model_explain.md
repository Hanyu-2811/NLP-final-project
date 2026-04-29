Fine_tuned_model_explain

Functions:
1) tokenize(batch):
    use the tokenizer function inherent from model package to tokenize the inputs
2) compute_metrics(eval_prod):
    compute precision, recall, f1 score with the model evaluation functions
3) vote_machine():
    build a voting machine based on sequential neural network that optimal probability assigned to different models;
    eventually return the voting machine
4) main():
    # prepare steps
    get arguments to process for dataset;
    tokenize the inputs from those arguments(train file);
    make the model by fitting the model with the classification function using our own label classifiers;

    # real training and result steps
    make the training arguments with training argument function;
    put the training argument and model related information into trainer;
    train the model;

    # ensembling
    get TFIDF and perplexity models and compute probability and predictions of both models
    fit the voter based on the probabilities computed based on those models
    predict the optimal probabilities assigned to each model

    # get results of ensembling
    get the predicted results from the voting machine by making the results over 0.5 be label 1 and below be 0
    compute potenial metrics
    save model

Terminal line format:
1) First navigate to the current folder the files are directly in
2) input "python Fine_tuned_model.py" "train_file_info_name.json" "development_file_info_name.json output_file" "test_file_info_name.json""
    For instance: "python Fine_tuned_model.py train.json development.json test.json output_prac.txt"
