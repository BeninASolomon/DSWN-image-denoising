import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, Add, Input
from tensorflow.keras.models import Model
import pywt
import os
# Load the image


# Data paths
dataset_path = 'Data//'

# Load images 
images_ct = []

org_img=[]
ground1=[]
for r, d, f in os.walk(dataset_path):
    for file in f:
        images_ct.append(os.path.join(r, file))


for i in range(0,len(images_ct)):
    img = cv.imread(images_ct[i])
    resized_image = cv.resize(img, (256, 256))
    gray_img = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    org_img.append(gray_img)
    # Apply OpenCV denoising for GroundTruth Image
    dst_opencv = cv.fastNlMeansDenoisingColored(resized_image, None, 10, 10, 7, 21)
    gray_dst_opencv = cv.cvtColor(dst_opencv, cv.COLOR_BGR2GRAY)
    ground1.append(gray_dst_opencv)
# Normalize and reshape images
org_img=np.array(org_img)
gray_img = org_img.astype(np.float32) / 255.0
ground1=np.array(ground1)
gray_dst_opencv = ground1.astype(np.float32) / 255.0

gray_img = np.expand_dims(gray_img, axis=-1)  # Add channel dimension
gray_dst_opencv = np.expand_dims(gray_dst_opencv, axis=-1)  # Add channel dimension


#Densely Self-guided wavelet Network (DSWN) for image Denoising
def dwt2d(image, wavelet='haar'):
    coeffs = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs
    return np.stack([cA, cH, cV, cD], axis=-1)

def idwt2d(coeffs, wavelet='haar'):
    cA, cH, cV, cD = np.split(coeffs, 4, axis=-1)
    coeffs = (cA.squeeze(-1), (cH.squeeze(-1), cV.squeeze(-1), cD.squeeze(-1)))
    return pywt.idwt2(coeffs, wavelet)
def dcr_block(x):
    residual = x
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Add()([x, residual])
    return x
def dsw_network(input_shape):
    inputs = Input(shape=input_shape)

    # Top-level
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = PReLU()(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = PReLU()(x)
    x = dcr_block(x)
    
    # Middle-level (Example for one level)
    x_middle = Conv2D(64, (1, 1), padding='same')(x)
    x_middle = PReLU()(x_middle)
    
    # Full resolution level
    x_full = Conv2D(64, (3, 3), padding='same')(x_middle)
    x_full = PReLU()(x_full)
    x_full = dcr_block(x_full)
    
    # Two branches
    residual_branch = Conv2D(1, (1, 1))(x_full)
    end2end_branch = Conv2D(1, (1, 1))(x_full)
    end2end_branch = tf.keras.layers.Activation('tanh')(end2end_branch)

    # Average output
    output = tf.keras.layers.Average()([residual_branch, end2end_branch])
    
    model = Model(inputs, output)
    return model
model = dsw_network(input_shape=(256, 256, 1))
model.compile(optimizer='adam', loss='mean_absolute_error')

# Assuming x_train and y_train are your training data and labels
history = model.fit(gray_img, gray_dst_opencv, epochs=100, batch_size=32)
model.save_weights('denoise_sample.weights.h5')
#model.load_weights('denoise_sample.weights.h5')
denoise_images = model.predict(gray_img)
# Enhanced image

plt.imshow(gray_img[0])  # Use index 0 for the first image
plt.show()
# Enhanced image

plt.imshow(denoise_images[0])  # Use index 0 for the first image
plt.show()

