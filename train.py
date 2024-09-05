import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
import matplotlib.pyplot as plt


# Loading
base_dir = 'dataset'
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + '/train',
    image_size=(224, 224),
    batch_size=32,
    seed=100
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + '/val',
    image_size=(224, 224),
    batch_size=32,
    seed=100
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    base_dir + '/test',
    image_size=(224, 224),
    batch_size=32,
    seed=100
)

# Normalizing
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Loading the pretrained MobileNet V3 Small model
base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Creating the model
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Setting up the learning rate, batch size, and evaluation metrics
checkpoint = ModelCheckpoint( 'best_model.keras', monitor="val_loss", save_best_only=True, mode='min')
early_stopping = EarlyStopping( patience=20, restore_best_weights=True )
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[checkpoint, early_stopping])

# Evaluating the model
score, acc = model.evaluate(train_ds)
print('Test Loss =', score)
print('Test Accuracy =', acc)

# Exporting the model
model.export('SavedModel')
# Also use these commands to generate the actual .onnx file
# pip install tf2onnx
# python -m tf2onnx.convert --saved-model SavedModel --output mobilenetv3_model.onnx --opset 13


hist_=pd.DataFrame(hist.history)
print(hist_)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist_['loss'],label='Train_Loss')
plt.plot(hist_['val_loss'],label='Validation_Loss')
plt.title('Train_Loss & Validation_Loss',fontsize=20)
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist_['accuracy'],label='Train_Accuracy')
plt.plot(hist_['val_accuracy'],label='Validation_Accuracy')
plt.title('Train_Accuracy & Validation_Accuracy',fontsize=20)
plt.legend()
plt.show()
