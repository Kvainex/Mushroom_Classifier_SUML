import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def main():
    st.title("Mushroom Strain Classifier")

    class_names = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius',
                    'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

    #Load the model using tensorflow

    loaded_model = load_model("classification_model")

    uploaded_file = st.file_uploader("Upload an image of the mushroom", 
                                     type=["jpg", "png"])

    if uploaded_file is not None:

        st.image(uploaded_file, caption='Uploaded Image', 
                 use_column_width=True)

        if st.button('Classify'):

            img_height = 224
            img_width = 224

            image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(img_height, img_width))

            image_array = tf.keras.utils.img_to_array(image)
            image_array = tf.expand_dims(image_array, 0) # Create a batch

            predictions = loaded_model.predict(image_array)
            score = tf.nn.softmax(predictions[0])

            st.success(
                    "This {} most likely belongs to {} with a {:.2f} percent confidence."
                    .format("Mushroom", class_names[np.argmax(score)], 100 * np.max(score))
    )       

if __name__ == "__main__":
    main()