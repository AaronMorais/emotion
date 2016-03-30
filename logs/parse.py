import sys

# Input format
# "%d, %d, training accuracy, %g" % fold, step, accuracy
# "%d, %d, validation accuracy, %g" % fold, step, accuracy
# "%d, test accuracy, %g" % fold, accuracy
training_steps = {}
validation_steps = {}
fold_accuracy = []


def add_test_accuracy(line):
    s = line.split(",")
    accuracy = float(s[2])
    fold_accuracy.append(accuracy)


def add_validation_accuracy(line):
    s = line.split(",")
    step = str(float(s[1]))
    accuracy = float(s[3])
    if step in validation_steps:
        validation_steps[step].append(accuracy)
    else:
        validation_steps[step] = [accuracy]


def add_training_accuracy(line):
    s = line.split(",")
    step = str(float(s[1]))
    accuracy = float(s[3])
    if step in training_steps:
        training_steps[step].append(accuracy)
    else:
        training_steps[step] = [accuracy]


def average(l):
    return str(sum(l) / float(len(l)))


def print_test_accuracy():
    print("Average Test Accuracy: " + average(fold_accuracy))


def print_validation_accuracy():
    print("Validation Accuracy")
    print("step, accuracy")
    for key, value in validation_steps.items():
        print(key + ", " + average(value))


def print_training_accuracy():
    print("Training Accuracy")
    print("step, accuracy")
    for key, value in training_steps.items():
        print(key + ", " + average(value))

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
    print_validation_accuracy()
    print_training_accuracy()
