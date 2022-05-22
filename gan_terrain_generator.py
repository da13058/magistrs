# Dmitrijs Arkašarins, da16019
# UZ ATVĒRTIEM DATIEM BALSTĪTA REĢIONAM LĪDZĪGĀ RELJEFA ĢENERĒŠANA, maģistra darbs

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt

# Attēla izšķirtspējas parametri
RESOLUTION_TRIGGER = 16 # Parametrs, kas ietekmē izšķirtspēju
EDGE_LEN = 32 * RESOLUTION_TRIGGER # kvadrāts ar vismaz 32 pikseļu garām malām
IMAGE_CHANNELS = 1 # krāsainiem RGB attēliem izmanto 3, melnbaltiem 1

# Konfigurācijas parametri
MAP_ID = '4411'
MAP_TYPE = 'hillshade' # full vai hillshade, apzīmē kartes ieguves metodi
DATA_PATH = f'C:/Users/dmitr/OneDrive/Documents/{MAP_ID}/{MAP_TYPE}'
EPOCHS = 100 # Apmācību reizes

# Ģenerēšanas parametri
BATCH_SIZE = 32
BUFFER_SIZE = 100000
SEED_SIZE = 100
ROWS = 1 # nepieciešams, lai uzģenerētu vairākus attēla paraugus vienā datnē pie noteiktā apmācību reižu skaita
COLUMNS = 1
MARGIN = 0 # nepieciešams, ja vairākus attēlus ģenerē vienā mēģinājumā un vairākās rindās un kolonnās

def getHMS(elapsed):
    h = int(elapsed / (60 * 60))
    m = int((elapsed % (60 * 60)) / 60)
    s = elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

print(f"Tiek ģenerētas {EDGE_LEN}px {MAP_ID} kartes")

trainingBinary = os.path.join(DATA_PATH, f'training_data_{EDGE_LEN}_{MAP_ID}_{MAP_TYPE}.npy')

print(f"Tiek meklēts iepriekšējais trenēšanas ieraksts {trainingBinary}")

if not os.path.isfile(trainingBinary):
  start = time.time()
  print("Uzgaidiet, tiek ielādēti ievaddati")

  trainingData = []
  facesPath = os.path.join(DATA_PATH)
  for filename in tqdm(os.listdir(facesPath)):
      path = os.path.join(facesPath, filename)
      time.sleep(1)
      image = Image.open(path).resize((EDGE_LEN, EDGE_LEN), Image.Resampling.LANCZOS)
      trainingData.append(np.asarray(image))
  trainingData = np.reshape(trainingData,(EDGE_LEN, EDGE_LEN, IMAGE_CHANNELS))
  trainingData = trainingData.astype(np.float32)
  trainingData = trainingData / 127.5 - 1.


  print(f"Uzgaidiet, progress tiek saglabāts trenēšanas ierakstā {trainingBinary}")
  np.save(trainingBinary, trainingData)
  elapsed = time.time()-start
  print ('Attēlu apstrādes laiks: ' + getHMS(elapsed))
else:
  print("Uzgaidiet, tiek ielādēts iepriekšējais trenēšanas ieraksts")
  trainingData = np.load(trainingBinary)

trainDataset = tf.data.Dataset.from_tensor_slices(trainingData) \
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def buildGenerator(seedSize, channels):
    model = Sequential()

    model.add(Dense(4*4*256,activation="relu",input_dim=seedSize))
    model.add(Reshape((4,4,256)))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
   
    # Output resolution, additional upsampling
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    if RESOLUTION_TRIGGER>1:
      model.add(UpSampling2D(size=(RESOLUTION_TRIGGER, RESOLUTION_TRIGGER)))
      model.add(Conv2D(128,kernel_size=3,padding="same"))
      model.add(BatchNormalization(momentum=0.8))
      model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    return model


def buildDiscriminator(imageShape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=imageShape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def saveImages(imgID, noise):
  imgArray = np.full(( 
      MARGIN + (ROWS * (EDGE_LEN + MARGIN)), 
      MARGIN + (COLUMNS * (EDGE_LEN + MARGIN)), 3), 
      255, dtype=np.uint8)
  
  generatedImages = generator.predict(noise)

  generatedImages = 0.5 * generatedImages + 0.5

  imgCount = 0
  for row in range(ROWS):
      for col in range(COLUMNS):
        r = row * (EDGE_LEN+16) + MARGIN
        c = col * (EDGE_LEN+16) + MARGIN
        imgArray[r:r+EDGE_LEN,c:c+EDGE_LEN] \
            = generatedImages[imgCount] * 255
        imgCount += 1

          
  outputPath = os.path.join(DATA_PATH,'output')
  if not os.path.exists(outputPath):
    os.makedirs(outputPath)
  
  filename = os.path.join(outputPath,f"train-{imgID}.png")
  im = Image.fromarray(imgArray)
  im.save(filename)

generator = buildGenerator(SEED_SIZE, IMAGE_CHANNELS)

noise = tf.random.normal([1, SEED_SIZE])
generatedImage = generator(noise, training=False)

plt.imshow(generatedImage[0, :, :, 0])

imageShape = (EDGE_LEN, EDGE_LEN, IMAGE_CHANNELS)

discriminator = buildDiscriminator(imageShape)
decision = discriminator(generatedImage)
print (decision)

crossEntropy = tf.keras.losses.BinaryCrossentropy()

def discriminatorLoss(realOutput, fakeOutput):
    realLoss = crossEntropy(tf.ones_like(realOutput), realOutput)
    fakeLoss = crossEntropy(tf.zeros_like(fakeOutput), fakeOutput)
    totalLoss = realLoss + fakeLoss
    return totalLoss

def generatorLoss(fakeOutput):
    return crossEntropy(tf.ones_like(fakeOutput), fakeOutput)

genOptimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discOptimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

@tf.function
def train_step(images):
  seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

  with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
    generatedImages = generator(seed, training=True)

    realOutput = discriminator(images, training=True)
    fakeOutput = discriminator(generatedImages, training=True)

    genLoss = generatorLoss(fakeOutput)
    discLoss = discriminatorLoss(realOutput, fakeOutput)
    

    genGradients = genTape.gradient(\
        genLoss, generator.trainable_variables)
    discGradients = discTape.gradient(\
        discLoss, discriminator.trainable_variables)

    genOptimizer.apply_gradients(zip(
        genGradients, generator.trainable_variables))
    discOptimizer.apply_gradients(zip(
        discGradients, 
        discriminator.trainable_variables))
  return genLoss, discLoss

def train(dataset, epochs):
  fixedSeed = np.random.normal(0, 1, (ROWS * COLUMNS, SEED_SIZE))
  start = time.time()

  for epoch in range(epochs):
    genLossList = []
    discLossList = []

    for imgBatch in dataset:
      t = train_step(imgBatch)
      genLossList.append(t[0])
      discLossList.append(t[1])

    genLoss = sum(genLossList) / len(genLossList)
    discLoss = sum(discLossList) / len(discLossList)

    print (f'{epoch+1}. iterācija: ģenerātora trūkums {genLoss}, diskriminatora trūkums {discLoss}')
    saveImages(epoch, fixedSeed)

  elapsed = time.time() - start
  print ('Kopējais trenēšanas laiks: ' + getHMS(elapsed))

train(trainDataset, EPOCHS)

