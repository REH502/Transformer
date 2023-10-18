class Hyperparams:
    """Hyperparameters"""

    # data
    source_train = "data/cn.txt"
    target_train = "data/en.txt"
    source_test = "data/cn_test.txt"
    target_test = "data/en_test.txt"

    # training
    model_dir = "./models/"  # saving directory

    # model
    maxlen = 30  # Maximum number of words in a sentence. alias = T.
    min_cnt = 0  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_epochs = 50
    num_heads = 8
    dropout_rate = 0.4
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    eval_epoch = 1  # epoch of model for eval
    eval_script = 'scripts/validate.sh'
    check_frequency = 10  # checkpoint frequency
