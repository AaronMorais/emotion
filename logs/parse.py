import sys

# Input format
# "%d, %d, training accuracy, %g" % fold, step, accuracy
# "%d, %d, validation accuracy, %g" % fold, step, accuracy
# "%d, test accuracy, %g" % fold, accuracy
fold_steps = {}
fold_accuracy = []


def add_test_accuracy(line):
    s = line.split(",")
    accuracy = float(s[2])
    fold_accuracy.append(accuracy)


def add_validation_accuracy(line):
    s = line.split(",")
    fold = str(float(s[0]))
    step = float(s[1])
    accuracy = str(float(s[3]))

    if fold in fold_steps:
        f = fold_steps[fold]
        if step in f:
            f[step]["v"] = accuracy
        else:
            f[step] = {"v": accuracy}
    else:
        fold_steps[fold] = {step: {"v": accuracy}}


def add_training_accuracy(line):
    s = line.split(",")
    fold = str(float(s[0]))
    step = float(s[1])
    accuracy = str(min(1, float(s[3]) * (143.0 / 113.0)))

    if fold in fold_steps:
        f = fold_steps[fold]
        if step in f:
            f[step]["t"] = accuracy
        else:
            f[step] = {"t": accuracy}
    else:
        fold_steps[fold] = {step: {"t": accuracy}}


def average(l):
    return str(sum(l) / float(len(l)))


def print_test_accuracy():
    print("Average Test Accuracy: " + average(fold_accuracy))


def print_fold_accuracy():
    print("=================")
    print("=================")
    print("=================")
    for key, value in fold_steps.items():
        print("=================")
        print("FOLD " + key)
        print("step, validation accuracy, training accuracy")
        for step, v in iter(sorted(value.iteritems())):
            print(str(step) + ", " +
                  v.get("v", "null") + ", " + v.get("t", "null"))


if __name__ == '__main__':
    tests = []
    training = []

    lines = [line.rstrip('\n') for line in open(sys.argv[1])]
    for line in lines:
        if "training" in line:
            add_training_accuracy(line)
        elif "validation" in line:
            add_validation_accuracy(line)
        elif "test" in line:
            add_test_accuracy(line)

    print_test_accuracy()
    print_fold_accuracy()
