# Real-Time Emotion Detection

This project detects **human emotions in real time** from a webcam using a pre-trained CNN model (`mini_XCEPTION`) and OpenCV's Haar cascade for face detection

> Built with Python 3.12 and TensorFlow 2.19+

---

## Features

- Real-time video capture from your webcam
- Face detection using Haar cascades
- Emotion classification using a pre-trained CNN
- Displays bounding boxes and emotion labels live on screen

---

## Requirements

Install Python 3.12+ and create a virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt