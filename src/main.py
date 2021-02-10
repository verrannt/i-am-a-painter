# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
#
# Train a network for the "I'm Something of a Painter Myself" Kaggle challenge.
#
# NOTE: Run this script from the root of the repository with:
# > python src/main.py
#
# AUTHORS (Github handles)
# - @aerigon
# - @thomasroodnl
# - @verrannt
#
# CREDITS
# - Amy Jang's CycleGAN tutorial 
#   (https://www.kaggle.com/amyjang/monet-cyclegan-tutorial)
# - Joaquín Bengochea's extension using data augmentation
#   (https://www.kaggle.com/joackobengochea/cyclegan-with-data-augmentation
#
# LICENSE
# All code is licensed under GNU General Public License (find at root of repo)
# 
# % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %


# System modules
import os
import shutil

# 3rd party modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import numpy as np
import PIL
from tqdm import tqdm
from random import random

# Own modules
from data_loader import DataLoader, get_filenames
from models.generators import Generators
from models.discriminators import Discriminators
from models.full import CycleGan
from models.losses import *

print('Tensorflow Version:', tf.__version__)

def initialize_TPUs():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except Exception as e:
        print(e)
        strategy = tf.distribute.get_strategy()
    print('Number of replicas:', strategy.num_replicas_in_sync)


if __name__=='__main__':

    try:
        os.mkdir('../../kaggle/images')
        print('Created `images` directory.')
    except FileExistsError:
        print('`images` directory already exists.')

    # Enable using TPUs (only on Kaggle and Google CoLab)
    USE_TPUS = False
    # Use a custom data path. If empty string, uses '/kaggle/input/gan-getting-started/'
    DATA_PATH = '../../kaggle/input/gan-getting-started/'
    # Set global image size constant
    IMAGE_SIZE = [256, 256]
    # Configure validation split
    VAL_SPLIT=0.2
    # Configure batch size for training
    BATCH_SIZE=6

    if USE_TPUS:
        initialize_TPUs()

    monet_filenames, photo_filenames = get_filenames(USE_TPUS, DATA_PATH)

    # Get datasets
    dataLoader = DataLoader(IMAGE_SIZE)

    monet_train, monet_val = dataLoader.load_dataset(
        monet_filenames, batch_size=BATCH_SIZE, vsplit=VAL_SPLIT, augment=True)
    photo_train, photo_val = dataLoader.load_dataset(
        photo_filenames, batch_size=BATCH_SIZE, vsplit=VAL_SPLIT, augment=True)
    photo_test, _ = dataLoader.load_dataset(
        photo_filenames, batch_size=1, augment=False)

    # Create model
    monet_generator = Generators.unet() 
    photo_generator = Generators.unet()
    monet_discriminator = Discriminators.default() 
    photo_discriminator = Discriminators.default()

    monet_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    monet_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    photo_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    cycle_gan_model = CycleGan(
        monet_generator, photo_generator, 
        monet_discriminator, photo_discriminator
    )

    cycle_gan_model.compile(
        m_gen_optimizer = monet_generator_optimizer,
        p_gen_optimizer = photo_generator_optimizer,
        m_disc_optimizer = monet_discriminator_optimizer,
        p_disc_optimizer = photo_discriminator_optimizer,
        gen_loss_fn = generator_loss,
        disc_loss_fn = discriminator_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

    # Fit model
    history = cycle_gan_model.fit(
        tf.data.Dataset.zip((monet_train, photo_train)),
        validation_data=tf.data.Dataset.zip((monet_val, photo_val)),
        validation_freq=5,
        epochs=1,
        verbose=1
    )

    # Plot exemplary model output
    #_, ax = plt.subplots(5, 2, figsize=(60, 60))
    #for i, img in enumerate(photo_train.take(5)):
    #    prediction = monet_generator(img, training=False)[0].numpy()
    #    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    #    img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

    #    ax[i, 0].imshow(img)
    #    ax[i, 1].imshow(prediction)
    #    ax[i, 0].set_title("Input Photo")
    #    ax[i, 1].set_title("Monet-esque")
    #    ax[i, 0].axis("off")
    #    ax[i, 1].axis("off")
    #plt.show()

    # Save outputs
    i = 1
    for img in photo_test:
        print('Processing image: {}\r'.format(i), end='',)
        prediction = monet_generator(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        im = PIL.Image.fromarray(prediction)
        im.save("../../kaggle/images/" + str(i) + ".jpg")
        i += 1

    shutil.make_archive("../../kaggle/working/images", 'zip', "../../kaggle/images")