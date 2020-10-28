#!/usr/bin/python

from __future__ import division

import copy

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def animate(frame):
    global positions

    update_boids(velocities)

    scatter.set_offsets(positions.transpose())


def init_positions(lower_limits, upper_limits):
    width = upper_limits - lower_limits

    return lower_limits[:, np.newaxis] + np.random.rand(2, BOIDS_COUNT) * width[:, np.newaxis]


def calculate_centers(square_distances, distance):
    global positions

    list_neighbors = np.where(square_distances < distance, 1, 0)
    list_neighbors -= np.identity(list_neighbors.shape[0], dtype="int64")
    nb_neighbors = np.sum(list_neighbors, axis=1, dtype="float64")
    nb_neighbors = nb_neighbors[:, np.newaxis]

    ind_has_neighbors = tuple(np.where(nb_neighbors != 0)[0])

    nb_neighbors[ind_has_neighbors, 0] = np.divide(np.ones((len(ind_has_neighbors)), dtype="float64"),
                                                   np.array(nb_neighbors[ind_has_neighbors, 0], dtype="float64"))

    centers = (np.array(list_neighbors, dtype=float)
               @ positions.T) * nb_neighbors

    return copy.deepcopy(centers)


def update_boids(vel):
    global positions
    # we look at the separation distances
    # 2*N*N separation matrix
    separations = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]

    squared_displacements = separations * separations

    # N*N matrix with x^2 + y^2
    square_distances = np.sum(squared_displacements, 0)

    middle = calculate_centers(square_distances, ATTRACTION_DISTANCE)
    # we get a N*2 Matrix with middles

    # cohesion:
    # 2*N - 2*N matrix
    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    # For boids that have neighbors near, we apply cohesion
    direction_to_middle = (middle.T - positions) * has_neighbors

    vel += direction_to_middle * MOVE_TO_MIDDLE_STRENGTH

    # separation
    # for boids that need to separate
    middle = calculate_centers(square_distances, ALERT_DISTANCE)

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    direction_to_middle = (middle.T - positions) * has_neighbors
    vel -= direction_to_middle * SEPARATION_STRENGTH

    # alignment
    # 2*N*N matrix of velocity differences
    velocity_differences = vel[:, np.newaxis, :] - vel[:, :, np.newaxis]

    very_far = square_distances > FORMATION_FLYING_DISTANCE
    # N * N matrix with True if the neighbor is away False else

    velocity_differences_if_close = np.copy(velocity_differences)

    velocity_differences_if_close[0, :, :][very_far] = 0
    velocity_differences_if_close[1, :, :][very_far] = 0

    vel -= np.mean(velocity_differences_if_close, 1) * FORMATION_FLYING_STRENGTH

    vel = limit_vel(vel)

    positions = positions + vel
    wrap()


def limit_vel(vel):  # cart2pol and pol2cart cost a lot, to optimize

    rho, phy = cart2pol(vel[0, :], vel[1, :])

    rho = np.where(rho > MAX_FORCE, MAX_FORCE, rho)
    rho = np.where(rho < MIN_FORCE, MIN_FORCE, rho)

    vel[0, :], vel[1, :] = pol2cart(rho, phy)

    return vel


# allow the boids to stay in the simulation
def wrap():
    global positions
    # we update the x positions to stay in the simulation
    pos_x = np.where(positions[0, :] < 0, positions[0, :] + X_LIMIT, positions[0, :])
    pos_x = np.where(pos_x > X_LIMIT, pos_x - X_LIMIT, pos_x)
    pos_x = pos_x[np.newaxis, :]

    # we update the y positions to stay in the simulation
    pos_y = np.where(positions[1, :] < 0, positions[1, :] + Y_LIMIT, positions[1, :])
    pos_y = np.where(pos_y > Y_LIMIT, pos_y - Y_LIMIT, pos_y)
    pos_y = pos_y[np.newaxis, :]

    positions = np.concatenate((pos_x, pos_y), axis=0)


# initial parameter
# cohesion
ATTRACTION_DISTANCE = 90000
MOVE_TO_MIDDLE_STRENGTH = 0.03

# separation
ALERT_DISTANCE = 3000
SEPARATION_STRENGTH = 0.035

# alignment
FORMATION_FLYING_DISTANCE = 100000
FORMATION_FLYING_STRENGTH = 0.015

# force
MAX_FORCE = 40
MIN_FORCE = 5

# General parameters
BOIDS_COUNT = 1000
X_LIMIT = 2000
Y_LIMIT = 2000

limits = np.array([X_LIMIT, Y_LIMIT])

# initial positions and velocities

positions = init_positions(np.array([10, 10]), np.array([1990, 1990]))

velocities = init_positions(np.array([10, -20]), np.array([15, 20]))

figure = plt.figure(figsize=[30, 30])
axes = plt.axes(xlim=(0, limits[0]), ylim=(0, limits[1]))
scatter = axes.scatter(positions[0, :], positions[1, :],
                       marker='o', edgecolor='k', lw=0.5)
anim = animation.FuncAnimation(figure, animate, frames=200, interval=30)
plt.show()
