from __future__ import division
import itertools
import random
import scipy.spatial
import colour
from sklearn.utils.extmath import cartesian
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

# TODO: DOCUMENT!

aRGB = colour.models.ADOBE_WIDE_GAMUT_RGB_COLOURSPACE
D50 = colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']


class Point(object):

    def __init__(self, value, label=None):
        self.value = value
        self.label = label


class Space(object):

    def __init__(self, points, dist, zero_point, JND_threshold=0.0):
        self.points = points
        self.dist = dist
        self.zero = zero_point
        self.JND_threshold = JND_threshold

    def min_dist(self, pt, ls):
        return min(self.dist(pt.value, p2.value) for p2 in ls)

    def avg_dist(self, p, ls):
        return sum(self.dist(p.value, p2.value) for p2 in ls) / len(ls)

    def max_dist(self, p, ls):
        return max(self.dist(p.value, p2.value) for p2 in ls)


def RGB_to_Lab(space, pt):
    XYZ = colour.RGB_to_XYZ(pt,
                            space.whitepoint,
                            D50,
                            space.RGB_to_XYZ_matrix)
    return colour.XYZ_to_Lab(XYZ)


def generate_CIELab_space(rgb_space=aRGB, axis_stride=0.1):
    # 3 axes, equal strides along each
    axes = [np.arange(0, 1+axis_stride, axis_stride)]*3
    rgb_points = cartesian(axes)
    lab_points = []
    for row in range(len(rgb_points)):
        lab_points.append(Point(RGB_to_Lab(rgb_space, rgb_points[row, :])))
    # dist is squared euclidean, so JND threshold is 0.23^2
    return Space(lab_points, dist, np.zeros(3), 0.23**2)


class Partition(object):

    def __init__(self, space, labels, temp=0.01, conv=1.0):
        self.partition = {label: [] for label in labels}
        self.centroids = {label: space.zero for label in labels}
        self.labels = labels
        self.space = space
        self.temp = temp
        self.conv = conv
        # TODO: write to / read from file?
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

    def convexify_region(self, label):

        region = self.partition[label]
        all_points = self.space.points
        # NOTE: sometimes the iterative calls to this method result in a cell of
        # the partition being empty.  We need to pass over it in those cases.  But
        # maybe we also need better logic to prevent this from happening?
        if len(region) < len(self.space.zero)+1:
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
        # 'Qs' option: search until non-coplanar input is found...
        # 'QJ' option: QHull joggles input so that it will be full dimensional.
        # If points happen to be co-planar, this will get around it.
        # Problem with QJ: it excludes some points from being inside their own
        # convex hull.  This effects the degree calculations quite a bit...
        convex_hull = scipy.spatial.ConvexHull(
            [point.value for point in region],
            qhull_options='QJ')
        misclassified = [point for point in all_points
                         if point_in_hull(point.value, convex_hull)
                         and point not in region]
        if len(misclassified) > 0:
            num_to_move = int(self.conv*len(misclassified))
            misclassified.sort(
                key=lambda point: self.space.min_dist(point, region))
            to_move = misclassified[:num_to_move]
            for point in to_move:
                self.relabel_point(point, label)

    def degree_of_convexity_of_cell(self, label):
        # empty regions have "degree" 1.0
        region = self.partition[label]
        if len(region) < len(self.space.zero)+1:
            return 1.0

        convex_hull = scipy.spatial.ConvexHull(
            [point.value for point in region],
            qhull_options='QJ')
        num_inside_hull = sum(int(point_in_hull(pt.value, convex_hull))
                              for pt in self.space.points)
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
        labels = self.labels

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
            dists = [min_dist(to_add, self.partition[label])
                     for label in labels]
            below_JND = np.where(dists < self.space.JND_threshold)[0]
            if len(below_JND) > 0:
                print 'below JND'
                cell = labels[np.random.choice(below_JND)]
            else:
                # TODO: parameterize the f in f(min_dist(pt, label))?
                norm_dists = dists / max(dists)
                weights = -np.array(norm_dists)
                probs = softmax(weights, self.temp)
                cell = np.random.choice(labels, p=probs)

            # add both to partition and labels array
            self.assign_point(to_add, cell)
            # mark as labeled
            unlabeled.remove(new_idx)

        if self.conv:
            print 'Convexifying...'
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

        # TODO: record stats about partition here
        # degree of convexity, size of each label, ...

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


def partition_to_img(partition, axes):
    img = np.zeros([axes[0][1], axes[1][1]])
    for label in partition:
        for point in partition[label]:
            img[point.value[0], point.value[1]] = label
    return img


def generate_2D_grid(temps, convs, axis_length):

    my_axes = [(0, axis_length, 1), (0, axis_length, 1)]
    points = list(Point(np.array(pt))
                  for pt in itertools.product(
                      *[range(*axis) for axis in my_axes]))
    labels = range(7)

    fig, axes = plt.subplots(nrows=len(temps), ncols=len(convs))

    for row in range(len(temps)):
        for col in range(len(convs)):
            temp = temps[row]
            conv = convs[col]
            print '{}, {}'.format(temp, conv)
            space = Space(points, dist, np.zeros(2))
            partition = Partition(space, labels, temp, conv)
            print partition.degree_of_convexity()
            img = partition_to_img(partition.partition, my_axes)
            ax = axes[row, col]
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow(img, aspect='equal', cmap='Set2')

    for ax, col in zip(axes[0], convs):
        ax.set_title('c = {}'.format(col), fontsize=12)

    for ax, row in zip(axes[:, 0], temps):
        ax.set_ylabel('t = {}'.format(row), fontsize=12)

    fig.tight_layout(h_pad=0.001, w_pad=0.001)
    plt.show()


if __name__ == '__main__':

    generate_2D_grid([1, 0.1, 0.01, 0.001, 0.0005], [0, 0.25, 0.5, 0.75, 1.0], 50)
    # generate_2D_grid([0.001, 0.0005], [0.75, 1.0], 20)

    """
    space = generate_CIELab_space(axis_stride=0.075)
    print len(space.points)
    labels = range(7)
    for idx in range(4):
        partition = Partition(space, labels, temp=0.0005, conv=1.0)
        print partition.degree_of_convexity()
        xs, ys, zs, color = [], [], [], []
        part = partition.partition
        for label in part:
            print len(part[label])
            for point in part[label]:
                xs.append(point.value[0])
                ys.append(point.value[1])
                zs.append(point.value[2])
                color.append(point.label)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, c=color)
        plt.show()
    """


    """

    AXES = [(0, 50, 1), (0, 50, 1)]
    #AXES = [(0, 12, 1), (0, 12, 1), (0, 12, 1)]
    points = list(Point(np.array(pt))
                  for pt in itertools.product(*[range(*axis) for axis in AXES]))
    print len(points)
    labels = range(7)

    # TODO: wrap generate_partition in a try block? manually ensure that each cell
    # has enough points and is not a line? something else?
    for idx in range(4):
        space = Space(points, dist, [0 for ax in AXES])
        partition = Partition(space, labels, temp=0.0005, conv=1.0)
        print partition.degree_of_convexity()

        # TODO: clean all this up
        if len(AXES) == 2:
            img = plt.imshow(partition_to_img(partition.partition))
            plt.show()
            # plt.savefig('partition_{}-n7-sm-0.01-conv-1.0.png'.format(idx))

        if len(AXES) == 3:
            xs, ys, zs, color = [], [], [], []
            part = partition.partition
            for label in part:
                print len(part[label])
                for point in part[label]:
                    xs.append(point.value[0])
                    ys.append(point.value[1])
                    zs.append(point.value[2])
                    color.append(point.label)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, c=color)
            plt.show()
    """
