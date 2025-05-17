import streamlit as st
import cv2
import numpy as np
from detector import PoseDetector
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="AI Fitness & Pose Detection")

st.title("💪 AI Fitness & Pose Detection")

# Initialize detector
detector = PoseDetector()

# Function to load and process sample images
def load_sample_image(sample_path):
    image = cv2.imread(sample_path)
    processed_image, results = detector.detect(image)
    return image, processed_image, results

# Create a tab-based interface
tab1, tab2, tab3 = st.tabs(["Upload Video", "Upload Image", "Try Samples"])

with tab1:
    st.header("Video Analysis")
    uploaded_video = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'], key="video_uploader")
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()
        
        # Process the video
        video_placeholder = st.empty()
        st.write("Processing video...")
        
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            stframe = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = detector.detect(frame)
                
                # Display the processed frame
                stframe.image(processed_frame, channels="BGR")
            
            cap.release()
        
        # Clean up
        os.unlink(tfile.name)

with tab2:
    st.header("Image Analysis")
    uploaded_image = st.file_uploader("Upload an image file", type=['jpg', 'jpeg', 'png'], key="image_uploader")
    
    if uploaded_image is not None:
        # Convert uploaded image to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Process image
        processed_frame, results = detector.detect(frame)
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with col2:
            st.subheader("Detected Pose")
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        
        # Display pose metrics or analysis results
        st.subheader("Analysis Results")
        
        # Example: Display confidence scores
        if results and hasattr(results, 'pose_landmarks'):
            st.success("Pose detected successfully!")
            # Add more metrics as needed
        else:
            st.warning("No pose detected in the image.")

with tab3:
    st.header("Sample Images")
    st.write("Try our pose detection with these sample images:")
    
    # Look for sample images directory
    sample_dir = Path("samples")
    
    # Check if directory exists
    if sample_dir.exists() and sample_dir.is_dir():
        # Get all image files
        sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg")) + list(sample_dir.glob("*.png"))
        
        if sample_images:
            # Create a selectbox to choose samples
            selected_sample = st.selectbox(
                "Select a sample image:",
                options=sample_images,
                format_func=lambda x: x.name
            )
            
            if selected_sample:
                # Load and process the selected sample
                try:
                    original, processed, results = load_sample_image(str(selected_sample))
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                    
                    with col2:
                        st.subheader("Detected Pose")
                        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                    
                    # Display pose metrics
                    st.subheader("Analysis Results")
                    if results and hasattr(results, 'pose_landmarks'):
                        st.success("Pose detected successfully!")
                        # Add metrics display here
                    else:
                        st.warning("No pose detected in the sample image.")
                        
                except Exception as e:
                    st.error(f"Error processing sample image: {str(e)}")
        else:
            st.warning("No sample images found in the samples directory.")
    else:
        st.info("""
        Sample images not found. Create a 'samples' directory in your project root 
        and add some image files (.jpg, .jpeg, .png) to see them here.
        """)
        
        # Display instructions for adding samples
        with st.expander("How to add sample images"):
            st.markdown("""
            1. Create a directory named `samples` in your project root
            2. Add image files (jpg, jpeg, png) to this directory
            3. Restart your Streamlit app
            
            Your directory structure should look like:
            ```
            ├── app.py
            ├── detector.py
            ├── requirements.txt
            ├── packages.txt
            └── samples/
                ├── yoga_pose1.jpg
                ├── exercise1.jpg
                └── ...
            ```
            """)

# Add informational section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app detects and analyzes human poses in images and videos using computer vision.
    
    **Instructions:**
    1. Upload an image or video
    2. View the pose detection results
    3. Check the analysis for fitness metrics
    
    Note: For best results, ensure the person is fully visible in the frame.
    """
)

# Display example thumbnails in the sidebar
st.sidebar.subheader("Example Poses")
# Check if sample directory exists
if Path("samples").exists():
    # Get first few samples
    thumbnail_samples = list(Path("samples").glob("*.jpg"))[:3]
    
    # Create columns for thumbnails
    if thumbnail_samples:
        cols = st.sidebar.columns(len(thumbnail_samples))
        
        for i, sample in enumerate(thumbnail_samples):
            try:
                img = cv2.imread(str(sample))
                if img is not None:
                    # Resize for thumbnail
                    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
                    # Display thumbnail
                    cols[i].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                 use_column_width=True,
                                 caption=sample.name)
            except Exception:
                pass
else:
    st.sidebar.write("Add sample images to see thumbnails here.")
