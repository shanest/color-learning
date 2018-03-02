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
        lab_points.append(RGB_to_Lab(rgb_space, rgb_points[row, :]))
    # dist is squared euclidean, so JND threshold is 0.23^2
    return np.array(lab_points)


class Partition(object):

    def __init__(self, points, labels, zero, temp=0.01, conv=1.0):
        self.partition = {label: [] for label in labels}
        self.centroids = {label: zero for label in labels}
        self.labelled_pts = [None]*len(points)
        self.labels = labels
        self.points = points
        self.zero = zero
        self.temp = temp
        self.conv = conv
        # TODO: write to / read from file?
        self._generate()

    def assign_point(self, pt, label):
        # pt.label = label
        self.labelled_pts[pt] = label
        # update centroid before .append so that weights are correct
        self.centroids[label] = np.average(
            [self.centroids[label], self.points[pt]],
            axis=0,
            weights=[len(self.partition[label]), 1])
        self.partition[label].append(pt)

    def remove_point(self, pt, label):
        # update centroid first
        self.centroids[label] -= self.points[pt] / len(self.partition[label])
        self.partition[label].remove(pt)
        self.labelled_pts[pt] = None
        # pt.label = None

    def relabel_point(self, pt, label):
        self.remove_point(pt, self.labelled_pts[pt])
        self.assign_point(pt, label)

    def convexify_region(self, label):

        region = self.partition[label]
        all_points = self.points
        # NOTE: sometimes the iterative calls to this method result in a cell of
        # the partition being empty.  We need to pass over it in those cases.  But
        # maybe we also need better logic to prevent this from happening?
        if len(region) < len(self.zero)+1:
            return

        # TODO: fix two errors in ConvexHull: (i) tuple index out of range; (ii)
        # not enough points to construct initial simplex; input is less than
        # two-dimensional since it has the same x coordinate.
        # (i) occurs when the number of points passed to ConvexHull is 0; this only
        # happens when an entire region has been "gobbled up" in the relabeling
        # process of this method.  Is there a way around this?!
        # Note: (ii) happens
        # just if a very small number of points is generated, so we could hard-wire
        # a minimum size of each cell of the partition.
        # (iii) has a similar origin:
        # when there are few points, they are more likely to be on a line.
        # 'Qs' option: search until non-coplanar input is found...
        # 'QJ' option: QHull joggles input so that it will be full dimensional.
        # If points happen to be co-planar, this will get around it.
        # Problem with QJ: it excludes some points from being inside their own
        # convex hull.  This effects the degree calculations quite a bit...
        # SOLUTION: add 'noise' to points in space, when making the space, so
        # that it's very unlikely that any will actually be linear!
        convex_hull = scipy.spatial.ConvexHull(
            [all_points[point] for point in region])
        misclassified = [point for point in range(len(all_points))
                         if point_in_hull(all_points[point], convex_hull)
                         and point not in region]
        if len(misclassified) > 0:
            num_to_move = int(self.conv*len(misclassified))
            misclassified.sort(
                key=lambda point: min_dist(point, region))
            to_move = misclassified[:num_to_move]
            for point in to_move:
                self.relabel_point(point, label)

    def degree_of_convexity_of_cell(self, label):
        # empty regions have "degree" 1.0
        region = self.partition[label]
        if len(region) < len(self.zero)+1:
            return 1.0

        convex_hull = scipy.spatial.ConvexHull(
            [self.points[point] for point in region])
        num_inside_hull = sum(int(point_in_hull(pt, convex_hull))
                              for pt in self.points)
        return len(region) / num_inside_hull

    def degree_of_convexity(self):
        partition = self.partition
        return np.average(
            [self.degree_of_convexity_of_cell(label) for label in partition],
            # should the weights be different -- uniform? -- for this mean?
            weights=[len(partition[label]) for label in partition])

    def _generate(self, init_n=5):

        points = self.points
        unlabeled = range(len(points))
        # min_dist = self.space.min_dist
        labels = self.labels

        # initialize with one seed point for each label
        seeds = random.sample(unlabeled, len(labels))
        for label in labels:
            unlabeled.remove(seeds[label])
            #self.assign_point(points[seeds[label]], label)
            self.assign_point(seeds[label], label)

        while len(unlabeled) > 0:
            # get random point
            new_idx = np.random.choice(unlabeled)
            to_add = points[new_idx]

            # choose cell based on how close it is to the other cells
            dists = [min_dist(to_add, self.partition[label])
                     for label in labels]

            # TODO: parameterize the f in f(min_dist(pt, label))?
            norm_dists = dists / max(dists)
            weights = -np.array(norm_dists)
            probs = softmax(weights, self.temp)
            cell = np.random.choice(labels, p=probs)

            # add both to partition and labels array
            #self.assign_point(to_add, cell)
            self.assign_point(new_idx, cell)
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

        points = self.points
        mislabeled = set()
        for label in self.labels:
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


def min_dist(pt, ls):
    return min(dist(pt, p2) for p2 in ls)


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
    part = partition.partition
    points = partition.points
    for label in part:
        for point in part[label]:
            # de-noise point value
            pt = np.around(points[point]).astype(int)
            img[pt[0], pt[1]] = label
    return img


def plot_3D_partition(partition):
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


def generate_2D_grid(temps, convs, axis_length):

    my_axes = [(0, axis_length, 1), (0, axis_length, 1)]
    points = np.array([np.array(pt, dtype=np.float_)
                       for pt in itertools.product(
                           *[range(*axis) for axis in my_axes])])
    # add noise so that co-linear points are very unlikely
    noise = np.random.random(points.shape) * 1e-5
    points += noise
    # points = [Point(points[row, :]) for row in range(len(points))]
    labels = range(7)

    fig, axes = plt.subplots(nrows=len(temps), ncols=len(convs))

    for row in range(len(temps)):
        for col in range(len(convs)):
            temp = temps[row]
            conv = convs[col]
            print '{}, {}'.format(temp, conv)
            partition = Partition(points, labels, np.zeros(2), temp, conv)
            print partition.degree_of_convexity()
            img = partition_to_img(partition, my_axes)
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

    generate_2D_grid([1, 0.1, 0.01, 0.001, 0.0005], [0, 0.25, 0.5, 0.75, 1.0],
                    40)

    """
    space = generate_CIELab_space(axis_stride=0.1)
    labels = range(7)
    for idx in range(4):
        partition = Partition(space, labels, temp=0.001, conv=1.0)
        print partition.degree_of_convexity()
        plot_3D_partition(partition)
    """
