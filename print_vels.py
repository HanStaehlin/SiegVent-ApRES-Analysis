import json
import numpy as np
with open('data/apres/ImageP2_python.mat', 'rb') as f:
    from scipy.io import loadmat
    mat = loadmat(f)
    print("Time bounds:", mat['TimeInDays'].flatten()[0], mat['TimeInDays'].flatten()[-1])

with open('output/radon_velocity.json', 'r') as f:
    pass # Wait, what is the JSON file path?
