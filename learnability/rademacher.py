from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, tan, acos, hypot, pi
from numpy import array
from numpy.linalg import norm
from itertools import combinations
from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"
        # DONE
        sum   = 0.0
        j     = 0
        for i in data:
            if(self.classify(i)):
                a = 1.0 
            else:
                a = -1.0
            sum += float(labels[j]) * a
            j +=1
        datasize = float(len(data))
        return (sum/datasize)


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """
    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    # DONE
    tt = list() #Make list of thetas
    for ii in dataset:
        if norm not in tt: #If it's not in the thetas set, then we append it to it damn it
            if ii[0] == 0:
                tt.append(0.00001)
            else:
                tt.append(atan(ii[1]/ii[0]))
    tt.sort() #get all the closest thetas next to each other

    theta = tt[0]
    y = tan(theta)
    hp = OriginPlaneHypothesis(1, y)
    hn = OriginPlaneHypothesis(-1,-y)
    classified = list()
    planed = list()
    classified.append([hp.classify(x) for x in dataset]), classified.append([hn.classify(x) for x in dataset])
    planed.append(hp), planed.append(hn)

    for i in range(len(tt)-1):
        theta = (tt[i]+tt[i+1])/2 + pi/2 #bisect the 2 thetas nearest each other
        hp = OriginPlaneHypothesis(1, tan(theta)) # +1 decision boundaries
        hn  = OriginPlaneHypothesis(-1, -tan(theta)) # -1 decision boundaries

        for d in dataset: #classify the data based on the computed boundaries
            p = [hp.classify(d)] #get i
            n = [hn.classify(d)]

        if (p not in classified): #if it's not already classified, add it and yield hp and hn
            classified.append(p)
            planed.append(hp)

        if (n not in classified):
            classified.append(n)
            planed.append(hn)

    return planed

def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # Complete this for extra credit
    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """
    # DONE
    # Empty rectangle first classifier
    empty =  AxisAlignedRectangle(0.0, 0.0, 0.0, 0.0)
    classified =  list()
    classified.append([empty.classify(d) for d in dataset])

    # Add all rectangles containing just one point aka squares
    squares = list()
    for ii in dataset:
        squares.append(AxisAlignedRectangle(ii[0],ii[1],ii[0],ii[1]))

    for x in range(1,len(dataset)):
        combos = combinations(dataset,x) #create tuple of combos of data and n
        for ii in combos:
            x = []
            y = []
            for i in ii:
                x.append(i[0])
                y.append(i[1]) 
            #creates rectangle with lower bound of minimum values of x and y, upper bound with the largest xs and ys 
            rectangle = AxisAlignedRectangle(min(x),min(y),max(x),max(y))
            for ii in dataset:
                rectangle.classify(ii)
            if rectangle not in squares:
                squares.append(rectangle)
    return squares


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """
    # DONE
    hypotheses = list(hypothesis_generator(dataset))
    sums=[]
    for ii in range(num_samples):
        if ii == 0:
            sigma = coin_tosses(len(dataset),random_seed)
        else:
            sigma = coin_tosses(len(dataset))
        sums.append(max([hypothesis.correlation(dataset, sigma) for hypothesis in hypotheses]))
    sums = sum(sums)
    return sums/num_samples


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
