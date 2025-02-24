# Object Detection Web Application

Welcome to the Object Detection Web Application repository! This project implements a Flask web application for object detection using YOLOv8 model trained on the COCO dataset.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Object Detection Web Application allows users to upload images or provide downloadable links to images or videos. The application performs object detection using the YOLOv8 model and displays the results with bounding boxes and labels for detected objects.

## Features

- Upload images or provide downloadable links for object detection.
- Real-time object detection for images and videos.
- Frame-by-frame analysis for video files.
- Detection of objects from 1000 classes provided by the ImageNet-1K dataset.
- Easy deployment on Google Cloud Platform or any other hosting service.

## Installation

To run the Object Detection Web Application locally, follow these steps:

### For Windows or Mac

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Kamlesh364/EE297-Project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd EE297-Project/
   ```

3. Install the required Python packages:
   
   ```bash
   pip3 install -r requirements.txt
   ```

### For Ubuntu (>18.04)

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Kamlesh364/EE297-Project.git
   ```

2. Navigate to the project directory:

   ```bash
   cd EE297-Project
   ```

3. make `setup.sh` executable:
   
   ```bash
   chmod +x setup.sh
   ```
4. setup the repository:
   
   ```bash
   ./setup.sh
   ```

## Usage

1. Start the Flask app:

   ```bash
   python3 real-time.py
   ```

2. Open a web browser and navigate to `http://localhost:8000` to access the application.

3. Upload an image/video file or provide a accessible link (example given below) to an image or video for object detection-
   ```bash
   https://www.youtube.com/ddTV12hErTc
   ```

### Note: Most of the Youtube videos require special permissions to be accesssible via python API. Please make sure the video is accessible before testing.

## Demo

Coming Soon!

## Contributing

Contributions to the Object Detection Web Application are welcome! If you find any bugs, have feature requests, or want to contribute improvements, please submit an issue or pull request.

## License

All copyrights are reserved by [Kamlesh364](https://github.com/kamlesh364).
