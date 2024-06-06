import pandas as pd
import tensorflow as tf
import os

def process_parquet_to_tfrecord(parquet_file_path, tfrecord_file_path):
    """ Function to process a single Parquet file and write it to a TFRecord file """

    # Step 1: Read the Parquet file into a Pandas DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Extract the byte arrays from the dictionaries
    df['original_image_bytes'] = df['original_image'].apply(lambda x: x['bytes'])
    df['edited_image_bytes'] = df['edited_image'].apply(lambda x: x['bytes'])

    # Step 2: Convert the DataFrame to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices({
        'original_prompt': df['original_prompt'].values,
        'original_image_bytes': df['original_image_bytes'].values,
        'edit_prompt': df['edit_prompt'].values,
        'edited_prompt': df['edited_prompt'].values,
        'edited_image_bytes': df['edited_image_bytes'].values
    })

    # Step 3: Define a function to serialize the data
    def serialize_example(original_prompt, original_image_bytes, edit_prompt, edited_prompt, edited_image_bytes):
        feature = {
            'original_prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[original_prompt.encode()])),
            'original_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[original_image_bytes])),
            'edit_prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[edit_prompt.encode()])),
            'edited_prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[edited_prompt.encode()])),
            'edited_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[edited_image_bytes]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # Step 4: Write the TFRecord file
    try:
        with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
            for row in dataset:
                example = serialize_example(
                    row['original_prompt'].numpy().decode(),
                    row['original_image_bytes'].numpy(),
                    row['edit_prompt'].numpy().decode(),
                    row['edited_prompt'].numpy().decode(),
                    row['edited_image_bytes'].numpy()
                )
                writer.write(example)

        # Delete the original Parquet file after successful conversion
        os.remove(parquet_file_path)
        print(f"Deleted original file: {parquet_file_path}")
    except Exception as e:
        print(f"Error processing file {parquet_file_path}: {e}")

if __name__ == "__main__":

    # Directory containing the Parquet files
    input_directory = os.path.join(os.path.expanduser("~"), "data")

    # Process each Parquet file in the directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.parquet'):
            parquet_file_path = os.path.join(input_directory, file_name)
            tfrecord_file_path = os.path.join(input_directory, f"{os.path.splitext(file_name)[0]}.tfrecord")
            
            # Check if the TFRecord file already exists
            if not os.path.exists(tfrecord_file_path):
                process_parquet_to_tfrecord(parquet_file_path, tfrecord_file_path)
            else:
                print(f"TFRecord file {tfrecord_file_path} already exists. Skipping conversion.")

