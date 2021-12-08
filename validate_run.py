import numpy as np
from PIL import Image


if __name__ == "__main__":
    binary = np.load("predictions.npz")
    keys = binary.files
    element = binary[keys[2]].astype(np.bool)
    i = Image.fromarray(element.squeeze(0))
    i.show()
    
