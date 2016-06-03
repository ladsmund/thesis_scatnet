import numpy as np
import sys


def conduct_experiment(**kwargs):
    training_data = kwargs.pop('training_data')
    training_labels = kwargs.pop('training_labels')
    test_data = kwargs.pop('test_data')
    test_labels = kwargs.pop('test_labels')
    classifier = kwargs.pop('classifier')
    n_training_samples = kwargs.pop('n_training_samples')
    max_repetitions = kwargs.pop('max_repetitions', 4)

    print "conduct_experiment"

    n_training_samples_total = training_labels.shape[0]

    classes = {int(np.round(c)) for c in set(training_labels)}
    class_indexes = [np.where(training_labels == c)[0] for c in classes]

    for n in n_training_samples:
        n_per_class = n // len(classes)
        n = n_per_class * len(classes)
        repetitions = min(max_repetitions, n_training_samples_total // n)

        sys.stdout.write("Testing for size: %5i which is %4i per class." % (n, n_per_class))
        sys.stdout.write(" Repeat %2i times\n" % repetitions)

        training_indexes = [[] for _ in range(repetitions)]
        for i in range(repetitions):

            permuted_labels = {c:0 for c in classes}
            for c in classes:
                permuted_labels[c] = list(np.random.permutation(class_indexes[c]))

            c = 0
            for _ in range(n):
                while len(permuted_labels[c]) == 0:
                    c = (c + 1) % len(classes)
                training_indexes[i].append(permuted_labels[c].pop())
                c = (c + 1) % len(classes)

        scores = []
        for indexes in training_indexes:
            c = classifier
            c.fit(training_data[indexes], training_labels[indexes], find_dim=True)
            scores.append(c.score(test_data, test_labels))
        print "\n".join(["Got: %5.3f%%" % (100 * s) for s in scores])


if __name__ == '__main__':
    training_data = np.random.random(60000)
    training_labels = np.sort(np.arange(60000) % 10)
    test_data = np.random.random(10000)
    test_labels = np.sort(np.arange(10000) % 10)
    # n_training_samples = [300]
    n_training_samples = [300, 1000, 2000, 5000, 10000, 20000, 40000, 60000]


    class classifier:
        def fit(self, data, labels):
            pass

        def score(self, data, labels):
            return np.random.random(1)


    conduct_experiment(
        training_data=training_data,
        training_labels=training_labels,
        test_data=test_data,
        test_labels=test_labels,
        n_training_samples=n_training_samples,
        max_repetitions=4,
        classifier=classifier)
