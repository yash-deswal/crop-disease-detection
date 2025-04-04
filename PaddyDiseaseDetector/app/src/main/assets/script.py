from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata_schema_py_generated as _metadata_fb

# === IMPORTANT: Define your normalization parameters here ===
# Example for scaling to [0.0, 1.0] (dividing by 255.0)
NORM_MEAN = [0.0]
NORM_STD = [255.0]

# Example for scaling to [-1.0, 1.0] (dividing by 127.5, subtracting 1.0)
# NORM_MEAN = [127.5]
# NORM_STD = [127.5]

# Choose the correct NORM_MEAN and NORM_STD based on your training!

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# Define Model Information
model_path = "paddy_disease_model.tflite"  # Your existing TFLite file
label_file_path = "labels.txt"      # Your labels file
output_model_with_metadata_path = "paddy_disease_model_with_metadata.tflite" # New output file name

# Create the metadata writer
writer = image_classifier.MetadataWriter.create_for_inference(
    writer_utils.load_file(model_path),
    # Input metadata (Normalization)
    input_norm_mean=NORM_MEAN,
    input_norm_std=NORM_STD,
    # Output metadata (Labels)
    label_file_paths=[label_file_path]
)

# Populate the metadata into the output file
writer_utils.save_file(writer.populate(), output_model_with_metadata_path)

print(f"Metadata added successfully. New model saved to: {output_model_with_metadata_path}")

# Optional: Display the metadata added
# displayer = image_classifier.MetadataDisplayer.with_model_file(output_model_with_metadata_path)
# print("Metadata populated:")
# print(displayer.get_metadata_json())
# print("Associated file(s) added:")
# print(displayer.get_associated_file_buffer_list())
