from numpy import array, zeros, where
from sklearn import svm
import argparse
from matplotlib import pyplot, cm
import matplotlib as mpl

# Using the KNN homework code to utilize SVM
class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))
    # DONE
    for ii in range(len(x)):
      w += x[ii] * y[ii] * alpha[ii]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()
    # DONE
    # Inspired https://www.cs.cmu.edu/~guestrin/Class/15781/recitations/r7/20071018svm.pdf
    for ii in range(len(x)):
        slope     = sum(w*x[ii]) + b
        intercept = slope * y[ii]
        if (intercept > 1 - tolerance) and (intercept < 1 + tolerance):
            support.add(ii)
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()
    # DONE
    for ii in range(len(x)):
        slope     = sum(w*x[ii]) + b
        intercept = slope * y[ii]
        if intercept < 1.0:
            slack.add(ii)
    return slack

def majority(self, item_indices):
    """
    Given the indices of training examples, return the majority label.  If
    there's a tie, return the median value (as implemented in numpy).
    :param item_indices: The indices of the k nearest neighbors
    """
    assert len(item_indices) == self._k, "Did not get k inputs"

    # Finish this function to return the most common y value for
    # these indices
    #
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
    # Goes through list comparing xi with xi+1
    major = [x[0] for x in Counter(self._y[item_indices]).most_common() 
                if x[1] == max(Counter(self._y[item_indices]).values())]

    return median(major)

def classify(self, example):
    """
    Given an example, classify the example.
    :param example: A representation of an example in the same
    format as training data
    """

    # Finish this function to find the k closest points, query the
    # majority function, and return the value.

    _, point = self._kdtree.query(example, self._k) #query(X[, k, return_distance, dualtree, ...])  query the tree for the k nearest neighbors

    return self.majority(point.flatten()) #flatten multidimensional arrays into 1D

def confusion_matrix(self, test_x, test_y):
    """
    Given a matrix of test examples and labels, compute the confusion
    matrixfor the current classifier.  Should return a dictionary of
    dictionaries where d[ii][jj] is the number of times an example
    with true label ii was labeled as jj.
    :param test_x: Test data representation
    :param test_y: Test data answers
    """

    # Finish this function to build a dictionary with the
    # mislabeled examples.  You'll need to call the classify
    # function for each example.

    d = defaultdict(dict)
    data_index = 0
    for xx, yy in zip(test_x, test_y):
        try:
            d[yy][self.classify(xx)] += 1
        except KeyError:
            d[yy][self.classify(xx)] = 1
        data_index += 1
        if data_index % 100 == 0:
            print("%i/%i for confusion matrix" % (data_index, len(test_x)))

    return d

@staticmethod
def acccuracy(confusion_matrix):
    """
    Given a confusion matrix, compute the accuracy of the underlying classifier.
    """

    # You do not need to modify this function

    total = 0
    correct = 0
    for ii in confusion_matrix:
        total += sum(confusion_matrix[ii].values())
        correct += confusion_matrix[ii].get(ii, 0)

    if total:
        return float(correct) / float(total)
    else:
        return 0.0

def show(result, numbers):
  #Examples, adapted online implementations
  #https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot
  col  = 3
  row  = len(numbers)
  for ii, val in enumerate(numbers):
      for jj in range(col):
          pyplot.subplot(row, col, jj+1+(ii*col))
          pyplot.subplots_adjust(wspace=0.5, hspace=0.5)
          pyplot.title(str(val))
          idx = where(data.train_y[result.support_]==val)
          train_sv = data.train_x[result.support_]
          pyplot.imshow(train_sv[idx[0][jj]].reshape((28,28)), cmap=cm.gray, interpolation='nearest')
  pyplot.show()


if __name__ == "__main__":
  # Adapted from KNN
  # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
  parser = argparse.ArgumentParser(description='SVM classifier options')
  parser.add_argument('--C', type=float, default=1.0,
          help="Penalty parameter C of the error term.")
  parser.add_argument('--d', type=int, default=1,
          help="Degree of the polynomial. Options: any float")
  parser.add_argument('--k', type=str, default='rbf',
          help="Specifies kernel. Options: linear, poly, rbf, sigmoid")
  parser.add_argument('--g', type=float, default=0.0,
          help="Kernel coefficient gamma. Options: any float")
  parser.add_argument('--c0', type=float, default=0.0,
          help="Independent term in Kernel function. Options: any float")
  parser.add_argument('--limit', type=int, default=-1,
          help="Restrict training to this many examples")

  # setup
  args = parser.parse_args()
  data = Numbers("../data/mnist.pkl.gz")

  result = svm.SVC(
                C      = args.C,
                kernel = args.k,
                degree = args.d,
                coef0  = args.c0,
                gamma  = args.g)
  #Training data
  if args.limit > 0:
      print("Data limit: %i" % args.limit)
      result.fit(data.train_x[:args.limit], data.train_y[:args.limit])
  else:
      result.fit(data.train_x, data.train_y)

  #Test
  Tester = result.predict(data.test_x)

  # evaluation
  accuracy = sum(Tester==data.test_y)/float(len(data.test_y)) * 100.000000

  print "Accuracy: ", accuracy, ", C:", args.C, ", Kernel:", args.k, ", Degree:", args.d, ", Gamma:", args.g, ", Coef0:", args.c0
  # show(result, [3,8,9])