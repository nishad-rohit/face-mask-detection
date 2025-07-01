import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Adjust paths as needed
directory = "face-mask-detection-dataset/Medical mask/Medical Mask/annotations"
image_directory = "face-mask-detection-dataset/Medical mask/Medical Mask/images"
df = pd.read_csv("train.csv")
df_test = pd.read_csv("submission.csv")



cvNet = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')


def getJSON(filePathandName):
    with open(filePathandName, 'r') as f:
        return json.load(f)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))



jsonfiles = [getJSON(os.path.join(directory, f)) for f in os.listdir(directory)]
data = []
img_size = 124
mask = ['face_with_mask']
non_mask = ['face_no_mask']
labels = {'mask': 0, 'without mask': 1}

for i in df["name"].unique():
    json_file = i + ".json"
    json_path = os.path.join(directory, json_file)

    if not os.path.exists(json_path):
        print(f"⚠️ JSON file not found: {json_path}")
        continue

    annotations = getJSON(json_path).get("Annotations", [])

    for j in annotations:
        if j["classname"] in mask + non_mask:
            x, y, w, h = j["BoundingBox"]

            # Read image
            image_path = os.path.join(image_directory, i)
            img = cv2.imread(image_path, 1)

            if img is None:
                print(f"⚠️ Could not read image: {image_path}")
                continue

            # Clamp bounding box to image size to avoid slicing errors
            height, width = img.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(0, min(w, width))
            h = max(0, min(h, height))

            if w <= x or h <= y:
                print(f"⚠️ Invalid bounding box in {image_path}: {x, y, w, h}")
                continue

            face = img[y:h, x:w]

            if face.size == 0:
                print(f"⚠️ Empty crop in {image_path}")
                continue

            face = cv2.resize(face, (img_size, img_size))
            label = labels["mask"] if j["classname"] in mask else labels["without mask"]
            data.append([face, label])

# Shuffle the data after collecting it
random.shuffle(data)


p = ["Mask" if label == 0 else "No Mask" for _, label in data]
sns.countplot(x=p)
plt.title("Mask vs No Mask Image Distribution")
plt.show()


X = np.array([item[0] for item in data]) / 255.0
Y = np.array([item[1] for item in data])
X = X.reshape(-1, img_size, img_size, 3)


xtrain, xval, ytrain, yval = train_test_split(X, Y, train_size=0.8, random_state=0)

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(xtrain)


model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(img_size, img_size, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    datagen.flow(xtrain, ytrain, batch_size=32),
    steps_per_epoch=len(xtrain) // 32,
    epochs=50,
    verbose=1,
    validation_data=(xval, yval)
)


test_images = ['1114.png', '1504.jpg', '0072.jpg', '0012.jpg', '0353.jpg', '1374.jpg']
assign = {0: 'Mask', 1: 'No Mask'}
gamma = 2.0

fig = plt.figure(figsize=(14, 14))
rows, cols = 3, 2

for j, img_name in enumerate(test_images):
    image = cv2.imread(os.path.join(image_directory, img_name), 1)
    image = adjust_gamma(image, gamma=gamma)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    cvNet.setInput(blob)
    detections = cvNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            frame = image[startY:endY, startX:endX]
            if frame.size == 0:
                continue
            im = cv2.resize(frame, (img_size, img_size))
            im = np.array(im) / 255.0
            im = im.reshape(1, img_size, img_size, 3)
            result = model.predict(im)
            label_Y = int(result > 0.5)
            label_text = assign[label_Y]
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (36, 255, 12), 2)

    ax = fig.add_subplot(rows, cols, j + 1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.axis('off')

plt.tight_layout()
plt.show()

