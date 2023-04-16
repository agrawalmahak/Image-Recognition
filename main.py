import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_train.csv')
print(train.head())
labels = train['label'].values
unique_val = np.array(labels)
print(np.unique(unique_val))
plt.figure(figsize = (18,8))
print(sns.countplot(x = labels))
train.drop('label',axis = 1,inplace = True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array ([i.flatten() for i in images])
# hot one encode our labels
from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)
labels
# Inspect an image
index = 2
print(labels [index])
plt.imshow(images [index].reshape (28,28))
# Use OpencV to view 10 random images from our training data
import cv2
import numpy as np
for i in range(0, 10):
    rand = np.random. randint (0, len (images))
    input_im = images [rand]
    sample = input_im.reshape (28, 28) .astype(np.uint8)
    sample = cv2. resize (sample, None, fx=10, fy=10, interpolation = cv2. INTER_CUBIC)
    cv2. imshow("sample image", sample)
    cv2.waitKey (0)
cv2.destroyAllWindows()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state=101)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
batch_size = 128
num_classes =24

epochs =10
x_train = x_train / 255
x_test = x_test / 255
x_train = x_train. reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test. reshape (x_test.shape[0], 28, 28, 1)
plt.imshow(x_train[0].reshape(28,28))
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

model = Sequential()
model. add(Conv2D (64, kernel_size= (3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model. add(Conv2D (64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D (pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten ())
model. add(Dense (128, activation = 'relu'))
model.add(Dropout (0.20))

model.add (Dense (num_classes, activation = "softmax"))
model.compile(loss = 'categorical_crossentropy',
              optimizer= Adam(),
              metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)
# Save our Model
model.save ("sign_mnist_cnn_50_Epochs.h5")
print("Model Saved" )
# View our training history graphically
plt.plot (history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel ('epoch')
plt.ylabel ('accuracy')
plt. legend (['train', 'test'])
plt.show()
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
test_images.shape
y_pred = model.predict (test_images)
# Get our accuracy score
import mediapipe as mp

mphands = mp.solutions.hands
hands = mphands.Hands()
from sklearn.metrics import accuracy_score

accuracy_score(test_labels, y_pred.round())
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape
# Create function to match label to letter
def getletter (result):
    classLabels = { 0: 'A',
                    1: 'B',
                    2: 'C',
                    3: 'D',
                    4: 'E',
                    5: 'F',
                    6: 'G',
                    7: 'H',
                    8: 'I',
                    9: 'J',
                    10: 'K',
                    11: 'L',
                    12: 'M',
                    13: 'N',
                    14: 'O',
                    15: 'P',
                    16: 'Q',
                    17: 'R',
                    18: 'S',
                    19: 'T',
                    20: 'U',
                    21: 'V',
                    22: 'W',
                    23: 'X',
                    24: 'Y',
                    25: 'Z'}
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # SPACE pressed
        analysisframe = frame
        showframe = analysisframe
        cv2.imshow("Frame", showframe)
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20

        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe, (28, 28))

        nlist = []
        rows, cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i, j]
                nlist.append(k)

        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1, 28, 28, 1)
        prediction = model.predict(pixeldata)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        for key, value in letter_prediction_dict.items():
            if value == high1:
                print("Predicted Character 1: ", key)
                print('Confidence 1: ', 100 * value)
            elif value == high2:
                print("Predicted Character 2: ", key)
                print('Confidence 2: ', 100 * value)
            elif value == high3:
                print("Predicted Character 3: ", key)
                print('Confidence 3: ', 100 * value)
        time.sleep(5)
