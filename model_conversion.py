import tensorflow as tf

model = tf.keras.models.load_model("best_vgg16_model.h5")
model.save("best_vgg16_model.keras")   # New Keras 3 format
