import sys
from predict import main as predict


if __name__ == "__main__":
    weights_path = "model_final.pth"
    images_path, predictions_fname = sys.argv[1:]
    predict(weights_path, images_path, predictions_fname)


