import numpy as np
import matplotlib.pyplot as plt

def main():
    x_train, y_train = get_data("pa2_train.csv")
    x_valid, y_valid = get_data("pa2_valid.csv")
    x_test, _ = get_data("pa2_test_no_label.csv", test=True)

    first_part(x_train, y_train, x_valid, y_valid, x_test)
    second_part(x_train, y_train, x_valid, y_valid)
    #third_part(x_train, y_train, x_valid, y_valid, x_test)

def get_data(filename, test=False):
    x = np.genfromtxt(filename, delimiter=',', dtype=float)
    y = None

    x = np.insert(x, [1], 1, axis=1)
    if not test:
        y = x[:, 0]
        y[y == 5] = -1
        y[y == 3] = 1

        x = np.delete(x, [0], 1) # remove first column with y values
    return x, y

def first_part(x_train, y_train, x_valid, y_valid, x_test):

    # train
    print "-------------------------------"
    print "Part 1. Training"
    print "-------------------------------"
    all_weights, train_accuracies = online_perceptron(x_train, y_train, 15)
    #np.savetxt('my_out.txt', weights)

    # validate
    print "-------------------------------"
    print "Part 1. Validation"
    print "-------------------------------"
    valid_accuracies = validate(all_weights, x_valid, y_valid)

    # predict
    print "-------------------------------"
    print "Part 1. Prediction"
    weights = all_weights[-2] # choose 14th iteration's weights
    filename = 'oplabel.csv'
    predictions = predict(x_test, weights)
    np.savetxt(filename, predictions)
    print "Saved to %s" % filename
    print "-------------------------------"

    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
    plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies)
    plt.show()

def second_part(x_train, y_train, x_valid, y_valid):

    # train
    print "-------------------------------"
    print "Part 2. Training"
    print "-------------------------------"
    all_weights, train_accuracies = average_perceptron(x_train, y_train, 15)
    #np.savetxt('my_out.txt', weights)

    # validate
    print "-------------------------------"
    print "Part 2. Validation"
    print "-------------------------------"
    valid_accuracies = validate(all_weights, x_valid, y_valid)

    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
    plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies)
    plt.show()

def third_part(x_train, y_train, x_valid, y_valid, x_test):
    p = 1
    # train
    print "-------------------------------"
    print "Part 3. Training"
    print "-------------------------------"
    alphas, train_accuracies = kernel_perceptron(x_train, y_train, p, 15)
    #np.savetxt('my_out.txt', weights)

    # validate
    print "-------------------------------"
    print "Part 3. Validation"
    print "-------------------------------"
    valid_accuracies = kernel_validate(x_valid, y_valid, p, alphas, 15,
            x_train, y_train)

    # predict
    print "-------------------------------"
    print "Part 1. Prediction"
    #weights = all_weights[-2] # choose 14th iteration's weights
    #filename = 'oplabel.csv'
    #predictions = predict(x_test, weights)
    #np.savetxt(filename, predictions)
    #print "Saved to %s" % filename
    print "-------------------------------"

    plt.figure()
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies)
    plt.plot(range(1, len(valid_accuracies)+1), valid_accuracies)
    plt.show()


def kernel_function(x, y, p):
    k = (1 + np.dot(x.T, y))**p
    return k

def kernel_perceptron(x, y, p, iters):
    m = x.shape[0]

    alphas = []
    accuracies = []

    kernel = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            kernel[i, j] = kernel_function(x[i, :], x[j, :], p)

    for iter_ in range(iters):
        alpha = np.zeros(m)
        correct_predictions = 0
        for i in range(m):
            u = np.dot(kernel[:, i], alpha * y)
            if y[i]*u <= 0:
                alpha[i] += 1
            else:
                correct_predictions += 1

        accuracy = correct_predictions / float(m)
        print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
        accuracies.append(accuracy)
        alphas.append(alpha)
    return alphas, accuracies

def kernel_validate(x, y, p, alphas, iters, x_train, y_train):
    m = x.shape[0]

    accuracies = []

    kernel = np.zeros((m, m))
    for i in range(x_train.shape[0]):
        for j in range(m):
            kernel[i, j] = kernel_function(x_train[i, :], x[j, :], p)

    for iter_ in range(iters):
        alpha = alphas[iter_]
        correct_predictions = 0
        for i in range(m):
            u = np.dot(kernel[:, i], alpha * y_train)
            if y[i]*u > 0:
                correct_predictions += 1

        accuracy = correct_predictions / float(m)
        print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
        accuracies.append(accuracy)
    return accuracies

def online_perceptron(x, y, iters):
    all_weights = []
    m = x.shape[0]
    accuracies = []
    weights = np.zeros(x.shape[1])

    for iter_ in range(iters):
        correct_predictions = 0
        for i in range(m):
            u = np.sign(y[i] * np.dot(x[i], weights))
            #loss += max(0, -1*u)
            if u <= 0:
                weights += y[i] * x[i]
            else:
                correct_predictions += 1

        accuracy = correct_predictions / float(m)
        print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
        accuracies.append(accuracy)

        all_weights.append(weights.copy())

    return all_weights, accuracies

def average_perceptron(x, y, iters):
    all_weights = []
    m = x.shape[0]
    accuracies = []
    weights = np.zeros(x.shape[1])
    avg_w = np.zeros(x.shape[1])
    s = 0
    c = 0

    for iter_ in range(iters):
        correct_predictions = 0
        for i in range(m):
            u = np.sign(y[i] * np.dot(x[i], weights))
            #loss += max(0, -1*u)
            if u <= 0:
                if s + c > 0:
                    avg_w = ((s * avg_w) + (c * weights)) / (s + c)
                s = s + c
                weights += y[i] * x[i]
                c = 0
            else:
                correct_predictions += 1
                c += 1

        if c > 0:
            avg_w = ((s * avg_w) + (c * weights)) / (s + c)

        accuracy = correct_predictions / float(m)
        print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
        accuracies.append(accuracy)

        all_weights.append(avg_w.copy())
    return all_weights, accuracies

def validate(all_weights, x, y):
    m = x.shape[0]
    accuracies = []
    for iter_, weights in enumerate(all_weights):
        correct_predictions = 0
        for i in range(m):
            u = np.sign(y[i] * np.dot(x[i], weights))
            #loss += max(0, -1*u)
            if u > 0:
                correct_predictions += 1

        accuracy = correct_predictions / float(m)
        print "Iteration %s, accuracy %s" % (iter_+1, accuracy)
        accuracies.append(accuracy)
    return accuracies

def predict(x, weights):
    predictions = np.sign(np.dot(x, np.matrix(weights).T))

    return predictions


if __name__ == "__main__":
    main()
