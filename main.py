import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras import layers



labels = ['Affenpinscher' 'Afghan_hound' 'Airedale_terrier' 'Akita'
 'Alaskan_malamute' 'American_eskimo_dog' 'American_foxhound'
 'American_staffordshire_terrier' 'American_water_spaniel'
 'Anatolian_shepherd_dog' 'Australian_cattle_dog' 'Australian_shepherd'
 'Australian_terrier' 'Basenji' 'Basset_hound' 'Beagle' 'Bearded_collie'
 'Beauceron' 'Bedlington_terrier' 'Belgian_malinois' 'Belgian_sheepdog'
 'Belgian_tervuren' 'Bernese_mountain_dog' 'Bichon_frise'
 'Black_and_tan_coonhound' 'Black_russian_terrier' 'Bloodhound'
 'Bluetick_coonhound' 'Border_collie' 'Border_terrier' 'Borzoi'
 'Boston_terrier' 'Bouvier_des_flandres' 'Boxer' 'Boykin_spaniel' 'Briard'
 'Brittany' 'Brussels_griffon' 'Bull_terrier' 'Bulldog' 'Bullmastiff'
 'Cairn_terrier' 'Canaan_dog' 'Cane_corso' 'Cardigan_welsh_corgi'
 'Cavalier_king_charles_spaniel' 'Chesapeake_bay_retriever' 'Chihuahua'
 'Chinese_crested' 'Chinese_shar-pei' 'Chow_chow' 'Clumber_spaniel'
 'Cocker_spaniel' 'Collie' 'Curly-coated_retriever' 'Dachshund'
 'Dalmatian' 'Dandie_dinmont_terrier' 'Doberman_pinscher'
 'Dogue_de_bordeaux' 'English_cocker_spaniel' 'English_setter'
 'English_springer_spaniel' 'English_toy_spaniel'
 'Entlebucher_mountain_dog' 'Field_spaniel' 'Finnish_spitz'
 'Flat-coated_retriever' 'French_bulldog' 'German_pinscher'
 'German_shepherd_dog' 'German_shorthaired_pointer'
 'German_wirehaired_pointer' 'Giant_schnauzer' 'Glen_of_imaal_terrier'
 'Golden_retriever' 'Gordon_setter' 'Great_dane' 'Great_pyrenees'
 'Greater_swiss_mountain_dog' 'Greyhound' 'Havanese' 'Ibizan_hound'
 'Icelandic_sheepdog' 'Irish_red_and_white_setter' 'Irish_setter'
 'Irish_terrier' 'Irish_water_spaniel' 'Irish_wolfhound'
 'Italian_greyhound' 'Japanese_chin' 'Keeshond' 'Kerry_blue_terrier'
 'Komondor' 'Kuvasz' 'Labrador_retriever' 'Lakeland_terrier' 'Leonberger'
 'Lhasa_apso' 'Lowchen' 'Maltese' 'Manchester_terrier' 'Mastiff'
 'Miniature_schnauzer' 'Neapolitan_mastiff' 'Newfoundland'
 'Norfolk_terrier' 'Norwegian_buhund' 'Norwegian_elkhound'
 'Norwegian_lundehund' 'Norwich_terrier'
 'Nova_scotia_duck_tolling_retriever' 'Old_english_sheepdog' 'Otterhound'
 'Papillon' 'Parson_russell_terrier' 'Pekingese' 'Pembroke_welsh_corgi'
 'Petit_basset_griffon_vendeen' 'Pharaoh_hound' 'Plott' 'Pointer'
 'Pomeranian' 'Poodle' 'Portuguese_water_dog' 'Saint_bernard'
 'Silky_terrier' 'Smooth_fox_terrier' 'Tibetan_mastiff'
 'Welsh_springer_spaniel' 'Wirehaired_pointing_griffon' 'Xoloitzcuintli'
 'Yorkshire_terrier']

# Original: EfficientNetB0 feature vector (version 1)
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
filepath = "weights.h5"
def create_model(model_url, num_classes=10):

  # Download the pretrained model and save it as Keras Layer
  feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, name='feature_extraction_layer', input_shape=(300,300,3))
  # create your own model
  model = tf.keras.Sequential([feature_extractor_layer,
                               layers.Dense(num_classes, activation='softmax', name='output_layer')])
  return model








def preprocess(uploaded_file):
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    image_tensor = tf.convert_to_tensor(image_array)
    scaled_image_tensor = tf.math.divide(image_tensor, 255)
    processed_image = tf.image.resize( scaled_image_tensor, (300, 300), preserve_aspect_ratio=True)

    return tf.reshape(processed_image, shape= (-1, 200, 300, 3), name=None)

def predict(processed_image):
    efficientnet_model = create_model(efficientnet_url, num_classes=133)
    efficientnet_model.load_weights(filepath)
    y_pred = efficientnet_model.predict(processed_image)
    label = tf.argmax(y_pred, 1)
    probability = y_pred[0][label]
    dog_label = labels[label]
    return probability, dog_label

def output(image, probability,dog_label):
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.title(dog_label)
    plt.axis('off')
    print(f'Your dog breed is:{dog_label} with {probability} probability')




def main():
    st.title("BreedNet - Dog Breed Classifier 🐕")
    st.write("Identify the breed of your Dog")

    menu = ["Home", "How it is made", "About me"]
    choice = st. sidebar.selectbox(" Choice", menu)
    if choice == "Home":
        uploaded_file = st.file_uploader("upload image of a dog", accept_multiple_files=False )
        if st.button("Classify"):
            processed_image_tensor = preprocess(uploaded_file)
            probability, dog_label = predict(processed_image_tensor)
            prediction = output(uploaded_file, probability, dog_label)
            st.write(prediction)









if __name__ == "__main__":
    main()