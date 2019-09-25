# Naive Bayes implementation to classify spam

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def quantize_features(arr, median):
    # quantize features such that values below median = 0, above median = 1
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1] - 1):
            arr[r][c] = 1 if median[c] <= arr[r][c] else 0
    return arr


def preprocess_data(train_size, file):
    # load spambase dataset
    data = np.loadtxt(file, delimiter=',')

    # shuffle and split
    np.random.shuffle(data)
    n_train = round(train_size * data.shape[0])
    train = data[:n_train, :]
    test = data[n_train:, :]

    medians = np.median(train, axis=0)
    train = quantize_features(train, medians).astype(int)
    test = quantize_features(test, medians).astype(int)
    return train, test


def train_model(train, log=False):
    # naive bayes training
    model = np.zeros((train.shape[1]-1, 2, 2))
    table_not_spam = train[train[:, -1] == 0]
    table_spam = train[train[:, -1] == 1]
    zeros_sum = np.sum(table_not_spam, axis=0)
    ones_sum = np.sum(table_spam, axis=0)

    for feature in range(train.shape[1] - 1):
        model[feature][0][1] = zeros_sum[feature] / table_not_spam.shape[0]
        model[feature][0][0] = 1 - model[feature][0][1]
        model[feature][1][1] = ones_sum[feature] / table_spam.shape[0]
        model[feature][1][0] = 1 - model[feature][1][1]
        if log:
            print("Given not spam, prob feature {} exists: {}, not exists: {}".format(feature,
                                                                                      model[feature][0][1],
                                                                                      model[feature][0][0]))
            print("Given spam, prob feature {} exists: {}, not exists: {}".format(feature,
                                                                                  model[feature][1][1],
                                                                                  model[feature][1][0]))
    return model


def get_priors(data, log=False):
    table_not_spam = data[data[:, -1] == 0]
    not_spam_prior = table_not_spam.shape[0] / data.shape[0]
    spam_prior = 1 - not_spam_prior
    if log:
        print("Prior not spam: {}, prior spam: {}".format(not_spam_prior, spam_prior))
    return not_spam_prior, spam_prior


def test_model(model, priors, test):
    # naive bayes testing
    not_spam_prior, spam_prior = priors[0], priors[1]
    n_test_samples = test.shape[0]
    predictions = np.zeros(n_test_samples, dtype=int)
    score = 0

    for idx, sample in enumerate(test):
        y_pred_not_spam = not_spam_prior
        y_pred_spam = spam_prior
        for feature_idx in range(sample.shape[0] - 1):
            y_pred_not_spam *= model[feature_idx][0][sample[feature_idx]]
            y_pred_spam *= model[feature_idx][1][sample[feature_idx]]
        prob_not_spam = y_pred_not_spam / (y_pred_not_spam + y_pred_spam)
        prob_spam = y_pred_spam / (y_pred_not_spam + y_pred_spam)
        predictions[idx] = 0 if prob_not_spam >= 0.5 else 1
        if sample[-1] == predictions[idx]:
            score += 1
    return score / n_test_samples


def main(train_size=.7):
    train, test = preprocess_data(train_size, 'data/spambase.data')
    model = train_model(train)
    priors = get_priors(train)
    accuracy = test_model(model, priors, test)
    print("Classifier accuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()
