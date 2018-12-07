from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])

labels = model.predict(image)[0]
labels = labels.tolist()

output = imutils.resize(orig, width=400)
cv2.putText(output, "{}: {:.2f}%".format(labels.index(max(labels)) + 1, max(labels) * 100), (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (0, 255, 0), 1)

cv2.imshow("Output", output)
cv2.waitKey(0)
