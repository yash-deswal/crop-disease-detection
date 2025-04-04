import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open("paddy_disease_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format.")
