from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.xception import preprocess_input
import gdown
import warnings
from PIL import Image, ImageDraw, ImageFont
import uuid


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])

def hello_world():
    request_type_str = request.method

    if request_type_str == "GET":
        return render_template("index.html", href = "static/cat_overview.png")

    else:
        text = request.form["text"]
        try:
            # Download image
            # url = "https://drive.google.com/file/d/1OLwvX6R1ufJwyv5cCBFsO1qsS-Qt3K3E/view?usp=share_link"
            random_str = uuid.uuid4().hex
            output = "static/"
            random_name = output+random_str+".png"
            gdown.download(url=text, output=random_name, quiet=True, fuzzy=True)

            # Image preprocessing
            my_image = load_img(random_name, target_size=(299, 299))
            my_image = img_to_array(my_image)
            my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
            my_image = preprocess_input(my_image)

            model = load_model('xception')
            # Prediction
            prediction = model.predict(my_image)
            result = np.argmax(prediction)


            # Get the label
            my_dict = {0:'Bombay',1:'Calico',2:'Persian',3:'Ragdoll',4:'Tuxedo'}
            breed = my_dict[result]

            ## img = mpimg.imread(random_name)
            font = ImageFont.truetype("Arial.ttf", size=10)
            # Open an Image
            img = Image.open(random_name)
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(img)
            I1.text((0,0),"The predicted breed is "+breed, fill=(255,0,0), font=font)
            img.save(random_name)
            content = "The predicted breed is "+breed+"."
        except:
            random_name = "static/cat_overview.png"
            content = "Invalid input url, please try again with valid google drive link with access to the picture."
        return render_template("index.html", href = random_name, content=content)
