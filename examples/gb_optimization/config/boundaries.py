import numpy as np

theta5  = 2 * np.arctan(1 / 3)
theta13 = 2 * np.arctan(1 / 5)

BOUNDARIES = {
    "sigma3_coherent_twin": {
        "misorientation": np.array([3*np.pi/4, np.arccos(-1/3), np.pi/4, np.pi/4, -np.arctan(1/np.sqrt(2))]),
        "repeat_factor": (2, 3),
    },
    "sigma5_210_11_2bar_0_ATGB": {
        "misorientation": np.array([theta5, 0, 0, 0, -np.arctan(1/2)]),
        "repeat_factor": (2, 3),
    },
    "sigma5_310_STGB": {
        "misorientation": np.array([theta5, 0, 0, 0, -theta5/2]),
        "repeat_factor": (2, 3),
    },
    "sigma5_mixed": {
        "misorientation": np.array([theta5, 0, 0, np.pi/4, -np.arctan(1/np.sqrt(2))]),
        "repeat_factor": (2, 3),
    },
    "sigma5_twist": {
        "misorientation": np.array([0, theta5, 0, 0, 0]),
        "repeat_factor": (2, 3),
    },
    "sigma13_510_STGB": {
        "misorientation": np.array([theta13, 0, 0, 0, -theta13/2]),
        "repeat_factor": (2, 3),
    },
}
