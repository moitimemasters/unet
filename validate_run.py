import numpy as np
from PIL import Image


if __name__ == "__main__":
    binary = np.load("predictions.npz")
    keys = binary.files
    element = binary[keys[0]].astype(np.bool)
    i = Image.fromarray(element)
    i.show()
    
