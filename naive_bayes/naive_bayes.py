import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# load spambase dataset
spambase = np.loadtxt('naive_bayes/spambase.data', delimiter=',')

# shuffle and split
np.random.shuffle(spambase)
train = spambase[:2000, :]
test = spambase[2000:, :]


# quantize features such that values below median = 0, above median = 1
def quantize_features(arr, median):
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]-1):
            arr[r][c] = 1 if median[c] <= arr[r][c] else 0
    return arr


medians = np.median(train, axis=0)
X_train = quantize_features(train, medians).astype(int)
X_test = quantize_features(test, medians).astype(int)


# naive bayes training
model = np.zeros((57, 2, 2))
l_zeros = X_train[X_train[:, -1] == 0]
l_ones = X_train[X_train[:, -1] == 1]
zeros_sum = np.sum(l_zeros, axis=0)
ones_sum = np.sum(l_ones, axis=0)

for feature in range(X_train.shape[1]-1):
    model[feature][0][1] = zeros_sum[feature] / l_zeros.shape[0]
    model[feature][0][0] = 1 - model[feature][0][1]
    model[feature][1][1] = ones_sum[feature] / l_ones.shape[0]
    model[feature][1][0] = 1 - model[feature][1][1]
    '''
    print("Given not spam, prob feature {} exists: {}, not exists: {}".format(feature,
                                                                              model[feature][0][1],
                                                                              model[feature][0][0]))
    print("Given spam, prob feature {} exists: {}, not exists: {}".format(feature,
                                                                          model[feature][1][1],
                                                                          model[feature][1][0]))
    '''

# test classifier
not_spam_prior = l_zeros.shape[0] / X_train.shape[0]
spam_prior = 1 - not_spam_prior
print("Prior not spam: {}, prior spam: {}".format(not_spam_prior, spam_prior))
n_test_samples = X_test.shape[0]
predictions = np.zeros(n_test_samples, dtype=int)
score = 0

for idx, sample in enumerate(X_test):
    y_pred_not_spam = not_spam_prior
    y_pred_spam = spam_prior
    for feature_idx in range(sample.shape[0]-1):
        y_pred_not_spam *= model[feature_idx][0][sample[feature_idx]]
        y_pred_spam *= model[feature_idx][1][sample[feature_idx]]
    prob_not_spam = y_pred_not_spam / (y_pred_not_spam + y_pred_spam)
    prob_spam = y_pred_spam / (y_pred_not_spam + y_pred_spam)
    predictions[idx] = 0 if prob_not_spam >= 0.5 else 1
    if sample[-1] == predictions[idx]:
        score += 1

print("Classifier accuracy: {}".format(score/n_test_samples))