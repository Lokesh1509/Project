import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from PIL import Image
from tkinter import filedialog, Tk

# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Base Classifier class
class Classifier:
    def __init__(self):
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):   
        self.model.load_weights(path)

# Define the Meso4 class that inherits from Classifier
class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

# Instantiate the Meso4 model and load weights
meso = Meso4()
meso.load("C:\\Meso4_DF.h5")  # Ensure the path to the model weights is correct

# Function to open a file dialog and load an image
def load_image():
    # Open a file dialog to select the image file
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return None

    # Load the image using PIL and ensure it's in RGB format
    img = Image.open(file_path).convert('RGB').resize((image_dimensions['height'], image_dimensions['width']))
    return img

# Function to preprocess the image
def preprocess_image(img):
    # Convert image to numpy array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch shape
    img_array /= 255.0  # Normalize to range [0,1]
    return img_array

# Load the image from user input
img = load_image()
if img is not None:
    # Preprocess the image for prediction
    img_array = preprocess_image(img)

    # Predict using the model
    prediction = meso.predict(img_array)[0][0]

    # Determine the label based on the prediction probability
    predicted_label = 'fake' if 0.4 <= prediction <= 0.69 else 'real'

    # Display the prediction result
    print(f"Predicted likelihood: {prediction:.4f}")
    print(f"Predicted label: {predicted_label}")

    # Show the image with the prediction
    plt.imshow(np.squeeze(img_array))
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
else:
    print("No image selected.")
