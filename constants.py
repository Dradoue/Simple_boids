import yaml
import numpy as np

PARAMS_FILE = "params.yaml"


def load_params(file):
    stream = open(file, 'r')
    dictionary = list(yaml.load_all(stream))
    win = dictionary[0]
    param_boids_ = dictionary[1]
    vectors = [pair for pair in param_boids_["relations"]]
    param_boids_["relations"] = np.array(vectors)
    return win, param_boids_


param_window, param_boids = load_params(PARAMS_FILE)

MAX_FORCE = param_boids["max_vel_norm"]
MIN_FORCE = param_boids["min_val_norm"]
# Window size
X_LIMIT = param_window["x_limit"]
Y_LIMIT = param_window["y_limit"]

LIMITS = np.array([X_LIMIT, Y_LIMIT])

# relations between boids
RELATIONS = param_boids["relations"]

ATTRACTION_DISTANCE = param_boids["attraction_distance"]

ALERT_DISTANCE = param_boids["alert_distance"]
FORMATION_FLYING_DISTANCE = param_boids["formation_flying_distance"]

MOVE_TO_MIDDLE_STRENGTH = param_boids["move_to_middle_strength"]
SEPARATION_STRENGTH = param_boids["separation_strength"]
FORMATION_FLYING_STRENGTH = param_boids["formation_flying_strength"]

CHASE_DISTANCE = param_boids["chase_distance"]
FLEE_DISTANCE = param_boids["flee_distance"]

CHASE_STRENGTH = param_boids["chase_strength"]
FLEE_STRENGTH = param_boids["flee_strength"]

IND = param_boids["indiv"]
BOIDS_COUNT = np.sum(IND)
# colors
COLORS = param_boids["colors"]
LIST_COLOR = []
for i in range(len(IND)):
    LIST_COLOR += COLORS[i] * IND[i]
