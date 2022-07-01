
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

image = pickle.load(open('Allimg160x128_Dataset.p', 'rb'))
with open('All2DGaussian160x128_Dataset.p', 'rb') as fp:
    GroundTruth = pickle.load(fp)

# test set
test_image = image[:416,:,:,:]
test_coord_xy = GroundTruth ['coord_xy'][:416,:]
print(test_image.shape)
print(test_coord_xy.shape)

#Validation_dataset
val_image = image[416:832,:,:,:]
val_coord_xy = GroundTruth ['coord_xy'][416:832,:]
print(val_image.shape)
print(val_coord_xy.shape)

# training data
train_image = image[832:4192,:,:,:]
train_prob_xy = GroundTruth['prob_xy'][832:4192,:,:]
print(train_image.shape)
print(train_prob_xy.shape)

#Data Loader
train_dataset = tf.data.Dataset.from_tensor_slices((train_image,  train_prob_xy ))
train_dataset = train_dataset.shuffle(buffer_size=len(train_image)).batch(8)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_image, val_coord_xy ))
validation_dataset = validation_dataset.batch(8)

test_dataset = tf.data.Dataset.from_tensor_slices((test_image , test_coord_xy))
test_dataset = test_dataset.batch(8)

# Create Block CNN
def ConvBlock(inputs,out_channels, kernel_size=3, dropout_p= 0.5):
    x = layers.Conv2D(out_channels, kernel_size, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_p)(x)

    return x

def DeconBlock(inputs, skip_connection, out_channels,kernel_size = 3):
    x = layers.Conv2DTranspose(out_channels, kernel_size, strides = 2, padding='same')(inputs)
    Concat = layers.Concatenate(axis = 3)([ x, skip_connection])
    x = layers.Conv2D(out_channels, kernel_size, padding='same')(Concat)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def Unet(input = layers.Input(shape=(128, 160, 3),name ='Input_Image')):

    # Downconvolution Block
    convblock1 = ConvBlock(input, 16)
    x = layers.MaxPool2D(pool_size=(2, 2))(convblock1)
    convblock2 = ConvBlock(x, 32)
    x = layers.MaxPool2D(pool_size=(2, 2))(convblock2)
    convblock3 = ConvBlock(x, 64)
    x = layers.MaxPool2D(pool_size=(2, 2))(convblock3)
    convblock4 = ConvBlock(x,128)
    x = layers.MaxPool2D(pool_size=(2, 2))(convblock4)
    BottleNeck = ConvBlock(x,256)

    # Upconvolution Block
    DeconvBLock1 = DeconBlock( BottleNeck, convblock4, 128)
    DeconvBLock2 = DeconBlock( DeconvBLock1, convblock3, 64)
    DeconvBLock3 = DeconBlock( DeconvBLock2, convblock2, 32)
    DeconvBLock4 = DeconBlock( DeconvBLock3, convblock1, 16)

    #Conv2D layer with filter, kernel size of 1
    Conv = layers.Conv2D(filters= 1, kernel_size= 1,padding='same',activation = "sigmoid")(DeconvBLock4)

    #Create Model
    model = keras.Model(inputs = input, outputs =Conv )

    return model

model= Unet()
model.summary()
def exponential_decay(epoch, lr, decay_rate=0.96):
    if epoch < 20:
        return lr
    else:
        return lr * decay_rate ** (epoch)

class CustomLearningRateScheduler(keras.callbacks.Callback):

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch):

        learning_rate = float(tf.keras.backend.get_value(optimizer.learning_rate))
        scheduled_lr = self.schedule(epoch, learning_rate)
        tf.keras.backend.set_value(optimizer.learning_rate, scheduled_lr)
        print("\nLearning rate is %6.8f." % (scheduled_lr))
        return scheduled_lr

def accuracy (y_true, y_predicted, radius=5):
    metric_accuracy = []
    for xy_true, xy_pred in zip(y_true, y_predicted):
        accuracy_radius = np.sum(np.square(np.subtract(xy_pred, xy_true)))
        if accuracy_radius < radius**2:
            score = int(1)
            metric_accuracy.append(score)
        else:
            score = int(0)
            metric_accuracy.append(score)
    return np.mean(metric_accuracy)

callback = CustomLearningRateScheduler(exponential_decay)
loss_func = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Adam(1e-3)
num_epochs = 30


@tf.function
def train_step(x_train, y_train):

    with tf.GradientTape() as tape:
        y_hat = model(x_train, training=True)
        loss =loss_func (y_train, y_hat)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return y_hat, loss

def test_step(x_test, y_test):
    y_hat = model(x_test, training=False)
    position_xy_pred = []
    for y in y_hat:
        position_xy = np.unravel_index(np.argmax(y), y.shape)
        position_xy_pred.append([int(position_xy[1])+1,int(position_xy[0])+1])
    test_accuracy = accuracy(y_test, position_xy_pred)
    return y_hat, test_accuracy


Training_loss = []
learning_rate = []
validation_accuracy = []

initialize_acuuracy = [0]
for epoch in range(num_epochs):
    print("\nEpoch [%d/%d]" % (epoch + 1, num_epochs), )
    lr = callback.on_epoch_begin(epoch)
    learning_rate.append(lr)

    for (x_batch_train, y_batch_train) in train_dataset:
        y_pred, loss = train_step(x_batch_train, y_batch_train)
    Training_loss.append(np.mean(loss))
    print("training loss: " + str(np.mean(loss)))

    validation_score = []
    for x_batch_val, y_batch_val in validation_dataset:
        y_pred_val, score_val = test_step(x_batch_val, y_batch_val)
        validation_score.append(score_val)
    validation_accuracy.append(np.mean(validation_score))
    print("validation accuracy: " + str(np.mean(validation_score)))

    if np.mean(validation_score) > initialize_acuuracy[0]:
        model.save('save model/Unet_160x128_1st.h5')
        model.save_weights('save model/'+str(np.mean(validation_score))+'Unet_Weight_160x128_1st.h5')
        initialize_acuuracy[0]= np.mean(validation_score)
    else:
        initialize_acuuracy[0] = initialize_acuuracy[0]

epochs = range(1, num_epochs + 1)
plt.plot(epochs, Training_loss, 'g', label='Training Loss')
plt.plot(epochs, validation_accuracy, 'b', label='Validation Accuracy')
plt.title('Training Loss and Validation Accuracy')
plt.legend()
plt.figure()
plt.show()


with open('save model//Unet_learning_rate_1st.p', 'wb') as fp:
    pickle.dump(learning_rate , fp, protocol=pickle.HIGHEST_PROTOCOL)
with open('save model/Unet_loss160x128_1st.p', 'wb') as fp:
    pickle.dump(Training_loss, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('save model/Unet_validation_accuracy160x128_1st.p', 'wb') as fp:
    pickle.dump(validation_accuracy , fp, protocol=pickle.HIGHEST_PROTOCOL)

test_score = []
for x_batch_test, y_batch_test in test_dataset:
    y_pred_test, score_test = test_step(x_batch_test, y_batch_test)
    test_score.append(score_test)
print("Test accuracy: " + str(np.mean(test_score)))
