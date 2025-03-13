import os
import time
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


class ImageProcessor(FileSystemEventHandler):
    def __init__(self, directory_path):
        # Keep track of processed files
        self.processed_files = set()

        # Process existing files first
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                self.process_image(file_path)

    def process_image(self, file_path):
        # Skip if already processed or not an image
        if file_path in self.processed_files or not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return None

        try:
            # Load and preprocess the image
            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            processed_img = preprocess_input(img_array)

            # Mark as processed and report
            self.processed_files.add(file_path)
            print(f"Processed: {os.path.basename(file_path)}")

            return processed_img

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def on_created(self, event):
        """Handle new files"""
        if not event.is_directory:
            print(f"New file detected: {event.src_path}")
            self.process_image(event.src_path)

    def on_modified(self, event):
        """Handle modified files (only if not processed before)"""
        if not event.is_directory and event.src_path not in self.processed_files:
            print(f"File modified: {event.src_path}")
            self.process_image(event.src_path)


# Main function to set up and run the file monitor
def monitor_directory(directory_path):
    # Create event handler and observer
    event_handler = ImageProcessor(directory_path)
    observer = Observer()
    observer.schedule(event_handler, directory_path, recursive=False)
    observer.start()

    print(f"Monitoring directory: {directory_path}")
    print("Press Ctrl+C to stop")

    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    directory_path = r"C:\Users\kamun\imagePractice"  # Use your directory path here
    monitor_directory(directory_path)