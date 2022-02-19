import os.path
from PIL import Image
from flask import Flask, render_template, request, jsonify
import cv2
import rgb2hsi
import numpy as np
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import DistanceMetric
import pickle

app= Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    prediction = None
    image_path = None


    if request.method == 'POST':
        #Load model
        model = pickle.load(open('knn_model', 'rb'))

        imagefile = request.files['image']
        image_path = "./static/images" + imagefile.filename
        imagefile.save(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        height = 500
        width = 500
        rsize = cv2.resize(img, (width, height))
        # Cropping
        jumbaris, jumkolom = rsize.shape[0], rsize.shape[1]
        xtengah = jumkolom // 2
        ytengah = jumbaris // 2
        titiktengah = [xtengah, ytengah]

        titikawalx = xtengah - 100
        titikawaly = ytengah - 100
        titikakhirx = xtengah + 100
        titikakhiry = ytengah + 100
        titikawal = (titikawalx, titikawaly)
        titikakhir = (titikakhirx, titikakhiry)
        crop = rsize[titikawaly:titikakhiry, titikawalx:titikakhirx]

        #HSI
        r, g, b = crop[100, 100]
        img2 = r, g, b
        H, S, I = rgb2hsi.RGB2HSI(img2)

        #LBP
        numPoints = 24
        radius = 3
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

        imean = np.mean(lbp)
        grayf = lbp.astype(np.float32)
        grayf2 = grayf * grayf
        imeanf2 = np.mean(grayf2)
        variance = imeanf2 - imean ** 2
        p = np.array([(lbp == v).sum() for v in range(256)])
        p = p / p.sum()
        entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()

        #Data Hasil Ekstraksi Fitur
        data = [H, S, I, imean, variance, entropy]

        # Prediction
        prediction = model.predict([data])

    #return data
    return render_template('index.html', prediction=prediction, image_loc=image_path)


if __name__ == '__main__':
    app.run(debug=True,  port=8000)
