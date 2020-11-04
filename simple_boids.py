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

    update_boids()

    scatter.set_offsets(positions.transpose())


def init_positions(lower_limits, upper_limits):
    """
    :param lower_limits:
    :param upper_limits:
    :return: array with positions initialised
    """
    width = upper_limits - lower_limits

    return lower_limits[:, np.newaxis] + np.random.rand(2, BOIDS_COUNT) * width[:, np.newaxis]


def calculate_centers(list_ind_1, list_ind_2, square_distances, distance):
    """
    Calculate the centers of list_ind_1 with neighborhood list_ind_2 and distance
    :param list_ind_1:
    :param list_ind_2:
    :param square_distances:
    :param distance:
    :return: centers
    """
    global positions

    list_neighbors = np.where(square_distances < distance, 1, 0)
    list_neighbors -= np.identity(list_neighbors.shape[0], dtype="int64")

    list_neighbors = list_neighbors[list_ind_1, :]
    list_neighbors = list_neighbors[:, list_ind_2]
    # matrix n_1 * n_2

    nb_neighbors = np.sum(list_neighbors, axis=1, dtype="float64")
    nb_neighbors = nb_neighbors[:, np.newaxis]
    # matrix n_1 * 1

    ind_has_neighbors = tuple(np.where(nb_neighbors != 0)[0])

    nb_neighbors[ind_has_neighbors, 0] = np.divide(np.ones((len(ind_has_neighbors)), dtype="float64"),
                                                   np.array(nb_neighbors[ind_has_neighbors, 0], dtype="float64"))

    centers = (np.array(list_neighbors, dtype=float)  # (n_1, n_2) @ (n_2, 1) = (n_1, 1)
               @ positions[:, list_ind_2].T) * nb_neighbors  # * (n_1, 1)

    return copy.deepcopy(centers)


def get_ind(species):
    """
    Get the list of indices for one species.
    :param species: int from 0 to n_species
    :return: list of indices
    """
    assert 0 <= species <= RELATIONS.shape[0], 'this species doesnt exist'

    list_indices = list()
    # particular cases:
    # if there is only one species or it is first species
    if species == 0:
        return list(np.arange(0, IND[0]))
    # other species
    else:
        return list(np.arange(np.sum(IND[:species]), np.sum(IND[:species]) + IND[species]))


def cohesion_separation_alignment(species, square_distances, velocity_differences):
    global velocities

    ind_species = get_ind(species)
    # cohesion:

    middle = calculate_centers(ind_species, ind_species, square_distances, ATTRACTION_DISTANCE)
    # we get a n_i*2 Matrix with middles

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    # For boids that have neighbors near, we apply cohesion
    direction_to_middle = (middle.T - positions[:, ind_species]) * has_neighbors

    velocities[:, ind_species] += direction_to_middle * MOVE_TO_MIDDLE_STRENGTH

    # separation
    middle = calculate_centers(ind_species, ind_species, square_distances, ALERT_DISTANCE)

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    direction_to_middle = (middle.T - positions[:, ind_species]) * has_neighbors
    velocities[:, ind_species] -= direction_to_middle * SEPARATION_STRENGTH

    # alignment
    dist = square_distances[ind_species, :]
    dist = dist[:, ind_species]

    very_far = dist > FORMATION_FLYING_DISTANCE
    # N * N matrix with True if the neighbor is away False else

    velocity_differences_if_close = np.copy(velocity_differences[:, :, ind_species])
    velocity_differences_if_close = velocity_differences_if_close[:, ind_species, :]
    # matrix (2, n_i, n_i)

    velocity_differences_if_close[0, :, :][very_far] = 0
    velocity_differences_if_close[1, :, :][very_far] = 0

    # (2,n_i)              (2, n_i)
    velocities[:, ind_species] -= np.mean(velocity_differences_if_close, 1) * FORMATION_FLYING_STRENGTH


def update_boids():
    global positions, velocities

    # 2*N*N separation matrix
    separations = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]
    # we look at the separation distances
    squared_displacements = separations * separations
    # N*N matrix with x^2 + y^2
    square_distances = np.sum(squared_displacements, 0)

    velocity_differences = velocities[:, np.newaxis, :] - velocities[:, :, np.newaxis]

    for i in range(RELATIONS.shape[0]):
        # for species n°i, we apply cohesion, separation, alignment
        ind_i = get_ind(i)
        cohesion_separation_alignment(i, square_distances, velocity_differences)
        flee = np.where(RELATIONS[i, :] == -1)[0]
        chase = np.where(RELATIONS[i, :] == 1)[0]
        ind_flee = []
        ind_chase = []
        for ind in flee:
            ind_flee += get_ind(ind)
        for ind in chase:
            ind_chase += get_ind(ind)

        # ATTRACTION_DISTANCE to replace here (temporary) by a new parameter
        chase_boids(ind_i, ind_flee, square_distances, ATTRACTION_DISTANCE, chase=False)
        chase_boids(ind_i, ind_chase, square_distances, ATTRACTION_DISTANCE, chase=True)

    limit_vel()
    positions = positions + velocities
    wrap()


def chase_boids(ind_chase, ind_prey, square_distances, distance, chase=True):
    global velocities
    middle = calculate_centers(ind_chase, ind_prey, square_distances, distance)
    # we get a n_i*2 Matrix with middles

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    # For boids that have neighbors near, we apply cohesion
    direction_to_middle = (middle.T - positions[:, ind_chase]) * has_neighbors

    if chase:
        velocities[:, ind_chase] += direction_to_middle * MOVE_TO_MIDDLE_STRENGTH
    else:
        velocities[:, ind_chase] -= direction_to_middle * MOVE_TO_MIDDLE_STRENGTH


def limit_vel():  # cart2pol and pol2cart cost a lot, to optimize
    global velocities

    rho, phy = cart2pol(velocities[0, :], velocities[1, :])

    rho = np.where(rho > MAX_FORCE, MAX_FORCE, rho)
    rho = np.where(rho < MIN_FORCE, MIN_FORCE, rho)

    velocities[0, :], velocities[1, :] = pol2cart(rho, phy)


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


# Boids parameters
# cohesion
ATTRACTION_DISTANCE = 70000  # 70000
MOVE_TO_MIDDLE_STRENGTH = 0.025  # 0.03

# separation
ALERT_DISTANCE = 3000  # 3000
SEPARATION_STRENGTH = 0.02

# alignment
FORMATION_FLYING_DISTANCE = 80000  # 80000
FORMATION_FLYING_STRENGTH = 0.023

# force
MAX_FORCE = 30
MIN_FORCE = 12

# Window size
X_LIMIT = 2000
Y_LIMIT = 2000

# relations between boids
RELATIONS = np.array([
    [0, 1, 1, -1],
    [-1, 0, -1, -1],
    [-1, 1, 0, 1],
    [-1, -1, -1, 0]
])

# number of boids for each species
N1 = 120
N2 = 120
N3 = 120
N4 = 120

IND = [N1, N2, N3, N4]

BOIDS_COUNT = np.sum(IND)

# colors
colors = ['r', 'g', 'b', 'c']  # , 'k'

list_colors = colors[0] * N1 + colors[1] * N2 + colors[2] * N3 + colors[3] * N4  

limits = np.array([X_LIMIT, Y_LIMIT])

# initial positions and velocities

positions = init_positions(np.array([10, 10]), np.array([1990, 1990]))

velocities = init_positions(np.array([5, -20]), np.array([12, 20]))

# animation

figure = plt.figure(figsize=[30, 30])
axes = plt.axes(xlim=(0, limits[0]), ylim=(0, limits[1]))
scatter = axes.scatter(positions[0, :], positions[1, :], c=list_colors,
                       marker='o', lw=0.5)

anim = animation.FuncAnimation(figure, animate, frames=200, interval=30)
plt.show()
