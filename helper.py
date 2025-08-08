from ultralytics import YOLO  # Import the YOLO class from the ultralytics library
import time  # Used for adding delay or tracking time
import streamlit as st  # Importing streamlit for creating web apps
import cv2  # OpenCV for image/video processing
import settings  # Custom Python file that contains constants like RECYCLABLE, WEBCAM_PATH, etc.
import threading  # Used to run tasks (like clearing the sidebar) asynchronously

# Function to wait for 3 seconds and then clear the sidebar placeholders
def sleep_and_clear_success():
    time.sleep(3)  # Wait for 3 seconds
    st.session_state['recyclable_placeholder'].empty()  # Clear recyclable items placeholder
    st.session_state['non_recyclable_placeholder'].empty()  # Clear non-recyclable items placeholder
    st.session_state['hazardous_placeholder'].empty()  # Clear hazardous items placeholder

# Load the YOLO model from a given path
def load_model(model_path):
    model = YOLO(model_path)  # Load YOLO model using the provided model path (e.g., best.pt)
    return model

# Classify detected items into recyclable, non-recyclable, and hazardous
def classify_waste_type(detected_items):
    recyclable_items = set(detected_items) & set(settings.RECYCLABLE)  # Items found in both detected and recyclable
    non_recyclable_items = set(detected_items) & set(settings.NON_RECYCLABLE)  # Intersection with non-recyclable
    hazardous_items = set(detected_items) & set(settings.HAZARDOUS)  # Intersection with hazardous list
    
    return recyclable_items, non_recyclable_items, hazardous_items  # Return all three categories

# Remove underscores in class names for better readability
def remove_dash_from_class_name(class_name):
    return class_name.replace("_", " ")  # Convert 'plastic_bottle' -> 'plastic bottle'

# Display detection results frame-by-frame and update sidebar based on the category
def _display_detected_frames(model, st_frame, image):
    image = cv2.resize(image, (640, int(640*(9/16))))  # Resize image to 16:9 aspect ratio (640x360)
    
    # Initialize session state variables if not already set
    if 'unique_classes' not in st.session_state:
        st.session_state['unique_classes'] = set()

    if 'recyclable_placeholder' not in st.session_state:
        st.session_state['recyclable_placeholder'] = st.sidebar.empty()
    if 'non_recyclable_placeholder' not in st.session_state:
        st.session_state['non_recyclable_placeholder'] = st.sidebar.empty()
    if 'hazardous_placeholder' not in st.session_state:
        st.session_state['hazardous_placeholder'] = st.sidebar.empty()

    if 'last_detection_time' not in st.session_state:
        st.session_state['last_detection_time'] = 0

    res = model.predict(image, conf=0.6)  # Run YOLO model on the image with 0.6 confidence threshold
    names = model.names  # Get class label names
    detected_items = set()  # To collect detected item labels

    # Process each result (frame prediction)
    for result in res:
        new_classes = set([names[int(c)] for c in result.boxes.cls])  # Get class names for detected boxes
        if new_classes != st.session_state['unique_classes']:  # Update only if new classes are detected
            st.session_state['unique_classes'] = new_classes  # Save new classes to session
            st.session_state['recyclable_placeholder'].markdown('')  # Clear previous recyclable markdown
            st.session_state['non_recyclable_placeholder'].markdown('')  # Clear non-recyclable markdown
            st.session_state['hazardous_placeholder'].markdown('')  # Clear hazardous markdown
            detected_items.update(st.session_state['unique_classes'])  # Update detected item list

            # Classify detected items
            recyclable_items, non_recyclable_items, hazardous_items = classify_waste_type(detected_items)

            # Update sidebar with recyclable items
            if recyclable_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in recyclable_items)
                st.session_state['recyclable_placeholder'].markdown(
                    f"<div class='stRecyclable'>Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            # Update sidebar with non-recyclable items
            if non_recyclable_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in non_recyclable_items)
                st.session_state['non_recyclable_placeholder'].markdown(
                    f"<div class='stNonRecyclable'>Non-Recyclable items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )
            # Update sidebar with hazardous items
            if hazardous_items:
                detected_items_str = "\n- ".join(remove_dash_from_class_name(item) for item in hazardous_items)
                st.session_state['hazardous_placeholder'].markdown(
                    f"<div class='stHazardous'>Hazardous items:\n\n- {detected_items_str}</div>",
                    unsafe_allow_html=True
                )

            # Start a new thread to clear the sidebar after 3 seconds
            threading.Thread(target=sleep_and_clear_success).start()
            st.session_state['last_detection_time'] = time.time()  # Save detection timestamp

    res_plotted = res[0].plot()  # Plot bounding boxes on the frame
    st_frame.image(res_plotted, channels="BGR")  # Display the image with boxes in Streamlit

# Function to access the webcam, detect objects, and show output
def play_webcam(model):
    source_webcam = settings.WEBCAM_PATH  # Get webcam source path from settings
    if st.button('Detect Objects'):  # When user clicks 'Detect Objects'
        try:
            vid_cap = cv2.VideoCapture(source_webcam)  # Open video stream
            st_frame = st.empty()  # Placeholder for updating video frames in Streamlit
            while (vid_cap.isOpened()):  # Continuously read frames
                success, image = vid_cap.read()  # Capture frame-by-frame
                if success:
                    _display_detected_frames(model,st_frame,image)  # Process and display detected frame
                else:
                    vid_cap.release()  # Release camera when no frames
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))  # Show error if something goes wrong
