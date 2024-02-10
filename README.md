# Realtime Object-Detection Application

  

## Overview

  

This is a real-time object detection application using Streamlit, YOLO (You Only Look Once) model from Ultralytics, and a custom tracker from the `supervision` library. The application allows users to upload a video, select specific object classes for detection, and visualize the results.

  

## Setup

  

### Requirements

  

Ensure you have the required dependencies installed. You can install them using the following command:

  

```bash

pip  install  -r  requirements.txt

```

  

### Database Setup

  

1. Make sure you have MySQL installed and running on your machine.

  

2. Update the MySQL connection details in `detection.py`:
	```python
	mydb  =  pymysql.connect(
		host="localhost",
		user="root",
		password="",
		database="detection" 
		)
	```
	make sure you have changed the database setup according to your MySQL.
  

## Running the Application

  

To run the application, execute the following command:

  

```bash

streamlit run app.py

```

  

The application will open in your default web browser.

  

## Usage

  

1. Upload a video using the "Upload Video" section in the sidebar.

  

2. Enter/select the desired object classes using the tag input.

  

3. Click the "Detect Objects" button to perform real-time object detection.

  

4. View the original video in one column and the detected video in the other.

  

## File Descriptions

  

-  **app.py**: The main Streamlit application script.

-  **detection.py**: Script containing the object detection logic using YOLO and supervision.

-  **requirements.txt**: List of Python dependencies required for the application.

  

## Acknowledgements

  

We would like to express our gratitude to the following individuals and projects:

  

-  **[Streamlit](https://www.streamlit.io/)**: A fantastic framework for building interactive web applications with Python.

  

-  **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: The Ultralytics team for providing a powerful YOLO model implementation.

  

-  **[Roboflow Supervision ](https://github.com/roboflow/supervision)**: Contributors to the supervision library for their work on computer vision annotation and processing.

  

-  **[MySQL](https://www.mysql.com/)**: The developers of MySQL for providing a robust and widely-used relational database system.

  

-  **[pymysql](https://pypi.org/project/pymysql/)**: Python MySQL database connector.

  

-  **[MoviePy](https://zulko.github.io/moviepy/)**: Video editing library for video file processing.

-  **[streamlit_tags](https://pypi.org/project/streamlit-tags/)** : Custom Streamlit component for tag input.

  

## License

  

This project is licensed under the [MIT License](LICENSE).