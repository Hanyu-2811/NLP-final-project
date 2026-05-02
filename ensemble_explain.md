Fine_tuned_model_explain

Functions:
1) tokenize(batch):
    use the tokenizer function inherent from model package to tokenize the inputs
2) compute_metrics(eval_prod):
    compute precision, recall, f1 score with the model evaluation functions
3) vote_machine():
    build a voting machine based on sequential neural network that optimal probability assigned to different models;
    eventually return the voting machine
4) roberta_predict_prob(trainer, tokenizer, raw_data):
    build a function that make it easy to predict probability for roberta models based on the three parameters as stated in the field
5) main():
    # modules for main ensemblings: 3 binary ensemblings and 3 robertas
    ensemble for 3 ensemblings by binary ensembling each pair of tfidf and signal features and 3 robertas,
    creating 6 separate models waiting for a large ensembling

    # create a main voting machine that ensembles for the 3 ensembles and 3 robertas
    this fits the major voting machine with 6 inputs and compiles;
    the main voting machine is train on a shared train file

    # test the voting machine on shared test
    the 2-level ensemblings are tested on shared test

    # write to two output files
    first output file would be written with output words
    second output file would be written with all metrics from both the 3 small ensemblings and the major voting machine

6) main_base():
    # prepare steps
    get arguments to process for dataset;
    tokenize the inputs from those arguments(train file);
    make the model by fitting the model with the classification function using our own label classifiers;

    # real training and result steps
    make the training arguments with training argument function;
    put the training argument and model related information into trainer;
    train the model;

    # ensembling
    get TFIDF and perplexity and also the other roberta models and compute probability and predictions of both models
    fit the voter based on the probabilities computed based on those models
    predict the optimal probabilities assigned to each model

    # get results of ensembling
    get the predicted results from the voting machine by making the results over 0.5 be label 1 and below be 0
    compute potenial metrics
    save model

Terminal line format:
1) First navigate to the current folder the files are directly in
2) input "ensemble.py" "train_file_shared.json" "test_file_shared.json" "output_file_for_outputting_words" "output_file_for_evaluations"
    For instance: "python ensemble.py shared_train.json shared_test.json output_words.txt evaluation_output.txt"
