'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# n = 100

# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

# for c, m, zlow, zhigh in [('b', 'o', -50, -25), ('b', 'o', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#     # for i in range(0, len(xs)):
#     	# print("%s %s %s" %(xs[i], ys[i], zs[i]))

# ax.set_xlabel('Variable 1')
# ax.set_ylabel('Variable 2')
# ax.set_zlabel('Variable 3')

# # plt.show()
# #
# #


a = 2.5 * np.random.randn(3, 100) + 3


for i in range(0, len(a[0])):
	print("%s %s %s" % (a[0][i], a[1][i], a[2][i]))
