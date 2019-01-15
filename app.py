from flask import Flask
from flask import Flask, request
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

im = Image.open("non_contrarie.bmp")

x_start = 70
y_start = 50
x_delta = 100
y_delta = 100
row_max = 20
col_max = 11

labels_array = []
models_array = []
raw_labels_array = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]
raw_models_array = []

for i in range(row_max):
    for j in range(col_max):
        current_box = (x_start, y_start, x_start + x_delta, y_start + y_delta)
        current_region = im.crop(current_box)
        current_region = current_region.convert("L")

        pix = np.array(current_region)
        pix = pix.reshape((current_region.size[0] * current_region.size[1],))

        models_array.append(pix)
        labels_array.append(raw_labels_array[i])
        raw_models_array.append(pix)

        x_start = x_start + 120

    x_start = 70
    y_start = y_start + 120

models_array = np.array(models_array)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, min_samples_leaf=1)
clf.fit(models_array, labels_array)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route("/add", methods=["POST"])
def add_example():
    json = request.get_json(force=True)

    # print(json["data"])
    # print(json["label"])

    new_model = np.array(json["data"])
    new_label = json["label"]

    raw_models_array.append(new_model)
    labels_array.append(new_label)

    new_arr = np.array(raw_models_array)
    clf.fit(new_arr, labels_array)

    return new_label


@app.route("/predict", methods=["POST"])
def predict():
    json = request.get_json(force=True)
    new = np.array(json["data"])
    new = new.reshape((1, -1))
    predict = clf.predict(new)
    print(predict)
    predString = "{}".format(predict)
    print(predString)
    #return predString[3:-2]
    return predict


if __name__ == '__main__':
    app.run()
