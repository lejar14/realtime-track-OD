import streamlit as st
from streamlit_tags import st_tags_sidebar
from detection import perform_detection, label_dict
from moviepy.editor import VideoFileClip
import os

# Set Streamlit page configuration
st.set_page_config(
    page_title="Realtime Object-Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for file uploader and class input
with st.sidebar:
    st.header("File Uploader")
    video_file = st.file_uploader("Upload Video", type="mp4")

    # Allow user to input/select classes manually using st_tags
    selected_classes = st_tags_sidebar(
        label='Enter classes:',
        text='Press enter to add more',
        value=['car', 'truck'],  # Initialize with an empty list
        suggestions=list(label_dict.values()),  # Use labels from the dictionary
        maxtags=len(label_dict)  # Maximum number of tags can be all labels
    )
    
    # Map selected class names to class indices
    selected_class_indices = [list(label_dict.keys())[list(label_dict.values()).index(class_name)] for class_name in selected_classes]
    st.write(selected_class_indices)

# Main content
st.title("Realtime Object Detection")

col1, col2 = st.columns(2)

with col1:
    if video_file is not None:
        st.write("Original Video")
        st.video(video_file)

try:
    if st.button("Detect Objects"):
        # Check if video_file is not None
        if video_file:
            # Create a temporary file to store the video content
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as temp_file:
                temp_file.write(video_file.read())

            # Use st.spinner to show loading state
            with st.spinner('Performing object detection...'):
                # Perform object detection without cancel button
                perform_detection(temp_video_path, "result.mp4", selected_class_indices)
                clip = VideoFileClip("result.mp4")
                clip.write_videofile("result.mp4".replace('.mp4', '_converted.mp4'), codec='libx264', audio_codec='aac')

                # Display a success message
                st.success("Object detection completed successfully!")
                os.remove(temp_video_path)

        else:
            st.warning("Please upload a video file before detecting objects.")
except Exception as ex:
    st.error("Error during object detection.")
    st.error(str(ex))  # Convert exception to string for better error display

with col2:
    if st.button("Show Detected Video"):
        st.write("Detected Video")
        st.video("result_converted.mp4")
