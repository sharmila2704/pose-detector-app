!/bin/bash

# Script to download sample pose images for your app

# Create the samples directory if it doesn't exist
mkdir -p samples

echo "Downloading sample yoga and fitness pose images..."

# Download sample yoga pose images from pexels or other free image sources
# Using curl to download images

# Sample 1: Yoga pose
curl -L "https://images.pexels.com/photos/374101/pexels-photo-374101.jpeg" -o samples/yoga_warrior.jpg

# Sample 2: Another yoga pose
curl -L "https://images.pexels.com/photos/3822167/pexels-photo-3822167.jpeg" -o samples/yoga_stretch.jpg

# Sample 3: Fitness pose
curl -L "https://images.pexels.com/photos/4057839/pexels-photo-4057839.jpeg" -o samples/fitness_squat.jpg

# Sample 4: Another fitness pose
curl -L "https://images.pexels.com/photos/6550839/pexels-photo-6550839.jpeg" -o samples/push_up_pose.jpg

# Sample 5: Side plank
curl -L "https://images.pexels.com/photos/6111616/pexels-photo-6111616.jpeg" -o samples/side_plank.jpg

echo "Downloaded 5 sample images to the samples directory!"
echo "You can now use these images in your Streamlit app."
