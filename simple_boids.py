#!/usr/bin/python

from __future__ import division

import copy

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt

import constants

from utils import cart2pol, pol2cart


def animate(frame):
    global positions

    update_boids()

    scatter.set_offsets(positions.transpose())


def init_positions(lower_limits, upper_limits):
    """
    return array with positions initialised
    """
    width = upper_limits - lower_limits

    return lower_limits[:, np.newaxis] + np.random.rand(2, constants.BOIDS_COUNT) * width[:, np.newaxis]


def calculate_centers(list_ind_1, list_ind_2, square_distances, distance):
    """
    Calculate the centers of list_ind_1 with neighborhood list_ind_2 and distance
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
    """
    assert 0 <= species <= constants.RELATIONS.shape[0], 'this species doesnt exist'

    # particular cases:
    # if there is only one species or it is first species
    if species == 0:
        return list(np.arange(0, constants.IND[0]))
    # other species
    else:
        return list(
            np.arange(np.sum(constants.IND[:species]), np.sum(constants.IND[:species]) + constants.IND[species]))


def cohesion_separation_alignment(species, square_distances, velocity_differences):
    """
    Apply cohesion-separation-alignment for a species
    """
    global velocities

    ind_species = get_ind(species)
    # cohesion:

    middle = calculate_centers(ind_species, ind_species, square_distances, constants.ATTRACTION_DISTANCE[species])
    # we get a n_i*2 Matrix with middles

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    # For boids that have neighbors near, we apply cohesion
    direction_to_middle = (middle.T - positions[:, ind_species]) * has_neighbors

    velocities[:, ind_species] += direction_to_middle * constants.MOVE_TO_MIDDLE_STRENGTH[species]

    # separation
    middle = calculate_centers(ind_species, ind_species, square_distances, constants.ALERT_DISTANCE[species])

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    direction_to_middle = (middle.T - positions[:, ind_species]) * has_neighbors
    velocities[:, ind_species] -= direction_to_middle * constants.SEPARATION_STRENGTH[species]

    # alignment
    dist = square_distances[ind_species, :]
    dist = dist[:, ind_species]

    very_far = dist > constants.FORMATION_FLYING_DISTANCE[species]
    # N * N matrix with True if the neighbor is away False else

    velocity_differences_if_close = np.copy(velocity_differences[:, :, ind_species])
    velocity_differences_if_close = velocity_differences_if_close[:, ind_species, :]
    # matrix (2, n_i, n_i)

    velocity_differences_if_close[0, :, :][very_far] = 0
    velocity_differences_if_close[1, :, :][very_far] = 0

    # (2,n_i)              (2, n_i)
    velocities[:, ind_species] -= np.mean(velocity_differences_if_close, 1) \
                                  * constants.FORMATION_FLYING_STRENGTH[species]


def update_boids():
    """
    Update the boids each frame
    """
    global positions, velocities

    # 2*N*N separation matrix
    separations = positions[:, np.newaxis, :] - positions[:, :, np.newaxis]
    # we look at the separation distances
    squared_displacements = separations * separations
    # N*N matrix with x^2 + y^2
    square_distances = np.sum(squared_displacements, 0)

    velocity_differences = velocities[:, np.newaxis, :] - velocities[:, :, np.newaxis]

    for i in range(constants.RELATIONS.shape[0]):
        # for species n°i, we apply cohesion, separation, alignment
        ind_i = get_ind(i)
        cohesion_separation_alignment(i, square_distances, velocity_differences)
        flee = np.where(constants.RELATIONS[i, :] == -1)[0]
        chase = np.where(constants.RELATIONS[i, :] == 1)[0]
        ind_flee = []
        ind_chase = []
        for ind in flee:
            ind_flee += get_ind(ind)
        for ind in chase:
            ind_chase += get_ind(ind)

        # ATTRACTION_DISTANCE to replace here (temporary) by a new parameter
        chase_boids(ind_i, ind_flee, square_distances, constants.FLEE_DISTANCE[i], constants.FLEE_STRENGTH[i],
                    chase=False)
        chase_boids(ind_i, ind_chase, square_distances, constants.CHASE_DISTANCE[i], constants.CHASE_STRENGTH[i],
                    chase=True)

    limit_vel()
    positions = positions + velocities
    wrap()


def chase_boids(ind_chase, ind_prey, square_distances, distance, strength, chase):
    """
    boids ind_chase chase or flee boids from ind_prey
    """
    global velocities
    middle = calculate_centers(ind_chase, ind_prey, square_distances, distance)
    # we get a n_i*2 Matrix with middles

    has_neighbors = np.where(middle[:, 0] != 0, 1, 0)
    has_neighbors = has_neighbors[np.newaxis, :]

    # For boids that have neighbors near, we apply cohesion
    direction_to_middle = (middle.T - positions[:, ind_chase]) * has_neighbors

    if chase:
        velocities[:, ind_chase] += direction_to_middle * strength
    else:
        velocities[:, ind_chase] -= direction_to_middle * strength


def limit_vel():  # cart2pol and pol2cart cost a lot, to optimize
    """
    Limit the velocity
    """
    global velocities

    rho, phy = cart2pol(velocities[0, :], velocities[1, :])

    rho = np.where(rho > constants.MAX_FORCE, constants.MAX_FORCE, rho)
    rho = np.where(rho < constants.MIN_FORCE, constants.MIN_FORCE, rho)

    velocities[0, :], velocities[1, :] = pol2cart(rho, phy)


def wrap():
    """
    Make sure that the boids stay into the simulation
    """
    global positions
    # we update the x positions to stay in the simulation
    pos_x = np.where(positions[0, :] < 0, positions[0, :] + constants.X_LIMIT, positions[0, :])
    pos_x = np.where(pos_x > constants.X_LIMIT, pos_x - constants.X_LIMIT, pos_x)
    pos_x = pos_x[np.newaxis, :]

    # we update the y positions to stay in the simulation
    pos_y = np.where(positions[1, :] < 0, positions[1, :] + constants.Y_LIMIT, positions[1, :])
    pos_y = np.where(pos_y > constants.Y_LIMIT, pos_y - constants.Y_LIMIT, pos_y)
    pos_y = pos_y[np.newaxis, :]

    positions = np.concatenate((pos_x, pos_y), axis=0)


# initial positions and velocities
positions = init_positions(np.array([10, 10]), np.array([constants.X_LIMIT - 10, constants.Y_LIMIT - 10]))
velocities = init_positions(np.array([5, -50]), np.array([12, 50]))

# animation
figure = plt.figure(figsize=[35, 35])
axes = plt.axes(xlim=(0, constants.LIMITS[0]), ylim=(0, constants.LIMITS[1]))
scatter = axes.scatter(positions[0, :], positions[1, :], c=constants.LIST_COLOR,
                       marker='o', lw=0.5)

anim = animation.FuncAnimation(figure, animate, frames=200, interval=30)
plt.show()
