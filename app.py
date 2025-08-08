# Importing the 'Path' class from the 'pathlib' module, which is used to handle file system paths in a clean and cross-platform way
from pathlib import Path

# Importing the Streamlit library as 'st', which is used to build interactive web apps in Python
import streamlit as st

# Importing a custom module called 'helper' – usually contains utility functions like loading models, webcam detection, etc.
import helper

# Importing another custom module called 'settings' – typically stores configuration like model path, webcam settings, etc.
import settings


# Setting the configuration of the Streamlit web app
# page_title: The title that appears on the browser tab
st.set_page_config(
    page_title="Waste Detection",
)


# Creating a sidebar title in the Streamlit UI that says "Detect Console"
st.sidebar.title("Detect Console")


# Getting the path to the trained YOLO model (best.pt) from the settings module
# The Path() function is used to make sure the path works across operating systems
model_path = Path(settings.DETECTION_MODEL)


# Setting the main title of the page in bold using Markdown-like syntax
st.title("Intelligent waste segregation system")

# Displaying a brief explanation to the user about how to start and stop object detection using the webcam
st.write("Start detecting objects in the webcam stream by clicking the button below. To stop the detection, click stop button in the top right corner of the webcam stream.")


# Embedding custom CSS styles into the Streamlit app to style the classification labels visually
# st.markdown is used to include raw HTML and CSS
st.markdown(
"""
<style>
    /* Style for Recyclable waste block (Yellow background) */
    .stRecyclable {
        background-color: rgba(233,192,78,255);
        padding: 1rem 0.75rem; /* Padding inside the box */
        margin-bottom: 1rem;   /* Space below each box */
        border-radius: 0.5rem; /* Rounded corners */
        margin-top: 0 !important;
        font-size:18px !important;
    }

    /* Style for Non-Recyclable waste block (Blue background) */
    .stNonRecyclable {
        background-color: rgba(94,128,173,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }

    /* Style for Hazardous waste block (Red background) */
    .stHazardous {
        background-color: rgba(194,84,85,255);
        padding: 1rem 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        margin-top: 0 !important;
        font-size:18px !important;
    }

</style>
""",
# unsafe_allow_html=True is necessary to allow raw HTML/CSS to render in Streamlit
unsafe_allow_html=True
)


# Try to load the model using the custom helper function
# If there's an error (like file not found), show an error message to the user
try:
    # Load the YOLO model for waste detection using the path defined earlier
    model = helper.load_model(model_path)

# If something goes wrong during loading (e.g., model file missing), handle the error gracefully
except Exception as ex:
    # Show a user-friendly error in the Streamlit app
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    # Also display the actual Python error for debugging
    st.error(ex)


# Once the model is loaded, call the function to start the webcam and perform object detection
# The webcam stream is displayed live in the browser
helper.play_webcam(model)


# Add a description in the sidebar to tell users this is just a demo of the detection model
st.sidebar.markdown("This is a demo of the waste detection model.", unsafe_allow_html=True)
