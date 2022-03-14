import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from labels import labels




def create_model():
    # name
    base_model = keras.applications.Xception(input_shape=(300, 300, 3), include_top=False )
    base_model.trainable = False
    inputs = keras.Input(shape=(300, 300, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    outputs = keras.layers.Dense(133, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model



def preprocess(uploaded_file):
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array)
    scaled_image_tensor = tf.math.divide(image_tensor, 255)
    processed_image = tf.image.resize(scaled_image_tensor, (300, 300), preserve_aspect_ratio=False)
    height, width, channel = processed_image.shape
    return tf.reshape(processed_image, shape=(-1, height, width, channel), name=None), image_array


def predict(processed_image):
    model = create_model()
    model.load_weights('weights.h5')
    y_pred = model.predict(processed_image)
    idx = int(tf.argmax(y_pred, 1))
    probability = y_pred[0][idx]
    dog_label = labels[idx]
    return probability, dog_label


def output(image, probability, dog_label):
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.title(dog_label)
    plt.axis('off')

    return print(f'Your dog breed is:{dog_label} with {probability:.2f} probability')



def main():
    st.title("BreedNet - Dog Breed Classifier 🐕")
    st.write("Identify the breed of your Dog")

    menu = ["Home", "How it is made", "About me"]
    choice = st. sidebar.selectbox(" Choice", menu)
    if choice == "Home":
        uploaded_file = st.file_uploader("upload image of a dog", accept_multiple_files=False )
        if st.button("Classify"):
            processed_image_tensor, image_array = preprocess(uploaded_file)
            probability, dog_label = predict(processed_image_tensor)
            st.image(image_array)
            st.write(f'Your dog breed is: {dog_label}  with  {probability:.2f}  probability')

if __name__ == "__main__":
    main()
