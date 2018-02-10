from __future__ import division
import itertools
import random
import scipy.spatial
import numpy as np
from matplotlib import pyplot as plt

# TODO: DOCUMENT!


class Point(object):

    def __init__(self, value, label=None):
        self.value = value
        self.label = label


class Space(object):

    def __init__(self, points, dist, zero_point):
        self.points = points
        self.dist = dist
        self.zero = zero_point

    def min_dist(self, pt, ls):
        return min(self.dist(pt.value, p2.value) for p2 in ls)

    def avg_dist(self, p, ls):
        return sum(self.dist(p.value, p2.value) for p2 in ls) / len(ls)

    def max_dist(self, p, ls):
        return max(self.dist(p.value, p2.value) for p2 in ls)


class Partition(object):

    def __init__(self, space, labels, temp=0.01, conv=1.0):
        self.partition = {label: [] for label in labels}
        self.centroids = {label: space.zero for label in labels}
        self.labels = labels
        self.space = space
        self.temp = temp
        self.conv = conv
        # TODO: different modes of generation?
        self._generate()

    def assign_point(self, pt, label):
        pt.label = label
        # update centroid before .append so that weights are correct
        self.centroids[label] = np.average(
            [self.centroids[label], pt.value],
            axis=0,
            weights=[len(self.partition[label]), 1])
        self.partition[label].append(pt)

    def remove_point(self, pt, label):
        # update centroid first
        self.centroids[label] -= pt.value / len(self.partition[label])
        self.partition[label].remove(pt)
        pt.label = None

    def relabel_point(self, pt, label):
        self.remove_point(pt, pt.label)
        self.assign_point(pt, label)

    # TODO: fix this so that when proportion=1.0, the resulting system is
    # guaranteed to be convex? Or good enough just to have almost always?
    def convexify_region(self, label):

        region = self.partition[label]
        all_points = self.space.points
        # NOTE: sometimes the iterative calls to this method result in a cell of
        # the partition being empty.  We need to pass over it in those cases.  But
        # maybe we also need better logic to prevent this from happening?
        if len(region) == 0:
            return

        # TODO: fix two errors in ConvexHull: (i) tuple index out of range; (ii)
        # not enough points to construct initial simplex; input is less than
        # two-dimensional since it has the same x coordinate.
        # (i) occurs when the number of points passed to ConvexHull is 0; this only
        # happens when an entire region has been "gobbled up" in the relabeling
        # process of this method.  Is there a way around this?!
        # Note: (ii) happens
        # just if a very small number of points is generated, so we could hard-wire
        # a minimum size of each cell of the partition. (iii) has a similar origin:
        # when there are few points, they are more likely to be on a line.
        convex_hull = scipy.spatial.ConvexHull([point.value for point in region])
        misclassified = [point for point in all_points
                         if point_in_hull(point.value, convex_hull)
                         and point not in region]
        if len(misclassified) > 0:
            # NOTE: the current method of choosing which points to move works
            # much better than the ones commented out below
            """
            to_move = np.random.choice(misclassified,
                                       size=int(self.conv*len(misclassified)),
                                       replace=False)
            misclassified.sort(
                key=lambda point: distance_to_convex_hull(point.value,
                                                          convex_hull))
            """
            num_to_move = int(self.conv*len(misclassified))
            misclassified.sort(
                key=lambda point: self.space.min_dist(point, region))
            to_move = misclassified[:num_to_move]
            for point in to_move:
                self.relabel_point(point, label)

    def degree_of_convexity_of_cell(self, label):
        # empty regions have "degree" 1.0
        region = self.partition[label]
        if len(region) == 0:
            return 1.0

        convex_hull = scipy.spatial.ConvexHull([point.value for point in region])
        num_inside_hull = sum(int(point_in_hull(pt.value, convex_hull)) for pt in
                              self.space.points)
        return len(region) / num_inside_hull

    def degree_of_convexity(self):
        partition = self.partition
        return np.average(
            [self.degree_of_convexity_of_cell(label) for label in partition],
            # should the weights be different -- uniform? -- for this mean?
            weights=[len(partition[label]) for label in partition])

    def _generate(self):

        points = self.space.points
        unlabeled = range(len(points))
        min_dist = self.space.min_dist

        # initialize with one seed point for each label
        seeds = random.sample(unlabeled, len(labels))
        for label in labels:
            unlabeled.remove(seeds[label])
            self.assign_point(points[seeds[label]], label)

        while len(unlabeled) > 0:
            # get random point
            new_idx = np.random.choice(unlabeled)
            to_add = points[new_idx]

            # choose cell based on how close it is to the other cells
            # TODO: parameterize the f in f(min_dist(pt, label))?
            weights = [1 / min_dist(to_add, self.partition[label])**0.25 for label in labels]
            # weights = [1 / self.space.dist(to_add.value, self.centroids[label]) for label in labels]
            probs = softmax(weights, self.temp)
            cell = np.random.choice(labels, p=probs)

            # add both to partition and labels array
            self.assign_point(to_add, cell)

            # mark as labeled
            unlabeled.remove(new_idx)

        if self.conv:

            # iterate through labels, starting with smallest, so that they are less
            # likely to get gobbled up in the convexify-ing process
            sorted_labels = sorted(labels,
                                   key=lambda label:
                                   self.degree_of_convexity_of_cell(label),
                                   reverse=True)
            # sorted_labels = sorted(labels, key=lambda label:
            #                        width(partition[label]))
            for label in sorted_labels:
                self.convexify_region(label)

    def _get_mislabeled(self):

        points = self.space.points
        mislabeled = set()
        for label in labels:
            convex_hull = scipy.spatial.ConvexHull([point.value for point in
                                                    self.partition[label]])
            not_in_hull = set([point for point in points if not
                               point_in_hull(point.value, convex_hull)])
            mislabeled |= not_in_hull
        return mislabeled


# sum of squares distance
def dist(p1, p2):
    """Distance of n-d coordinates. """
    return np.sum((p1 - p2)**2)


def width(ls):
    max_len = 0
    for idx1 in range(len(ls)):
        for idx2 in range(idx1, len(ls)):
            new_dist = dist(ls[idx1].value, ls[idx2].value)
            max_len = max_len or new_dist
    return max_len


def softmax(weights, temp=1.0):
    exp = np.exp(np.array(weights) / temp)
    return exp / np.sum(exp)


# see https://stackoverflow.com/a/42165596
def point_in_hull(pt, hull, eps=1e-12):
    return all(
        (np.dot(eq[:-1], pt) + eq[-1] <= eps)
        for eq in hull.equations)


# see https://stackoverflow.com/a/42254318
def distance_to_convex_hull(point, hull):
    distances = []
    for eq in hull.equations:
        t = -(eq[-1] + np.dot(eq[:-1], point))/(np.sum(eq[:-1]**2))
        projection = point + eq[:-1]*t
        distances.append(dist(point, projection))
    return min(distances)


def partition_to_img(partition):
    img = np.zeros([AXIS_SIZE, AXIS_SIZE])
    for label in partition:
        for point in partition[label]:
            img[point.value[0], point.value[1]] = label
    return img


if __name__ == '__main__':

    AXIS_SIZE = 40
    points = list(Point(np.array(pt))
                  for pt in itertools.product(range(AXIS_SIZE), repeat=2))
    labels = range(7)

    # TODO: wrap generate_partition in a try block? manually ensure that each cell
    # has enough points and is not a line? something else?
    for idx in range(4):
        space = Space(points, dist, (0, 0))
        partition = Partition(space, labels, conv=1.0)
        print partition.degree_of_convexity()
        img = plt.imshow(partition_to_img(partition.partition))
        plt.show()
        # plt.savefig('partition_{}-n7-sm-0.01-conv-1.0.png'.format(idx))
