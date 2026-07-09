# Importing libraries
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from model import model


model = tf.keras.models.load_model("my_model.keras")

# image uploader function
def imageUploader():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        img = Image.open(path)
        img = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img)

        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("560x300")
        label.config(image=pic)
        label.image = pic
        model.predict(pic)
        img_array = tf.keras.utils.img_to_array(img)

        # Add batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
        img_array = tf.expand_dims(img_array, 0)

        # Predict
        prediction = model.predict(img_array)
        score = prediction[0][0]

        label = "Fake" if score > 0.5 else "REAL"
        confidence = score if score > 0.5 else 1 - score
        label.config(text = f"Prediction: {label} ({confidence:.2%} confidence)")
        print(f"Prediction: {label} ({confidence:.2%} confidence)")  

    # if no file is selected, then we are displaying below message
    else:
        print("No file is Choosen !! Please choose a file.")




# defining tkinter object
app = tk.Tk()

# setting title and basic size to our App
app.title("AI Image Detector")
app.geometry("560x270")

# adding background image
imgLabel = Label(app)
imgLabel.place(x=0, y=0)

# adding background color to our upload button
app.option_add("*Label*Background", "white")
app.option_add("*Button*Background", "lightgreen")

label = tk.Label(app)
label.pack(pady=10)

# defining our upload buttom
uploadButton = tk.Button(app, text="Locate Image", command=imageUploader)
uploadButton.pack(side=tk.BOTTOM, pady=20)

predictButton

app.mainloop()