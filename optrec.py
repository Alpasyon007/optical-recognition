import pytesseract
import cv2 
import time
import numpy as np
import sys
import signal
import os

import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

from picamera.array import PiRGBArray
from picamera import PiCamera

from PIL import Image

from gtts import gTTS

cap = cv2.VideoCapture(0)

PIN_TRIGGER = 7
PIN_ECHO = 11

def ObjectRecognition(cap, model: str, max_results: int, score_threshold: float, num_threads: int,
        enable_edgetpu: bool, camera_id: int, width: int, height: int) -> None:

  # Initialize the image classification model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)

  # Enable Coral by this setting
  classification_options = processor.ClassificationOptions(
      max_results=max_results, score_threshold=score_threshold)
  options = vision.ImageClassifierOptions(
      base_options=base_options, classification_options=classification_options)

  classifier = vision.ImageClassifier.create_from_options(options)

  success, image = cap.read()
  if not success:
    sys.exit(
        'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    )

  # counter += 1
  image = cv2.flip(image, 1)

  # Convert the image from BGR to RGB as required by the TFLite model.
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Create TensorImage from the RGB image
  tensor_image = vision.TensorImage.create_from_array(rgb_image)
  # List classification results
  categories = classifier.classify(tensor_image)

  # Show classification results on the imagei
  for idx, category in enumerate(categories.classifications[0].categories):
    category_name = category.category_name
    score = round(category.score, 2)
    result_text = category_name + ' (' + str(score) + ')'
    print(result_text)
    return category_name

def OpticalCharacterRecognition(cap):
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )
    
    return pytesseract.image_to_string(image)

def DistanceMeasure():
  try:
        GPIO.setmode(GPIO.BOARD)

        GPIO.setup(PIN_TRIGGER, GPIO.OUT)
        GPIO.setup(PIN_ECHO, GPIO.IN)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        print ("Waiting for sensor to settle")

        time.sleep(2)

        print ("Calculating distance")

        GPIO.output(PIN_TRIGGER, GPIO.HIGH)

        time.sleep(0.00001)

        GPIO.output(PIN_TRIGGER, GPIO.LOW)

        while GPIO.input(PIN_ECHO)==0:
              pulse_start_time = time.time()
        while GPIO.input(PIN_ECHO)==1:
              pulse_end_time = time.time()

        pulse_duration = pulse_end_time - pulse_start_time
        distance = round(pulse_duration * 17150, 2)
        print ("Distance:",distance,"cm")

  finally:
        GPIO.cleanup()

def handler(signum, frame):
    cap.release()
    sys.exit("Done")

def main():
  language = 'en'

  signal.signal(signal.SIGINT, handler)  # prevent "crashing" with ctrl+C

  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  prev_ocr_text = ""
  prev_obj_text = ""

  while cap.isOpened():
    ocr_text = OpticalCharacterRecognition(cap)
    if ocr_text and (not ocr_text.isspace()) and (ocr_text != prev_ocr_text):
      prev_ocr_text = ocr_text
      print(ocr_text)
      gTTS(text=ocr_text, lang=language, slow=False).save("Text.mp3")
      os.system("mpg321 -q Text.mp3")

    obj_text = ObjectRecognition(cap, 'efficientnet_lite0.tflite', int(3), 0.2, int(4), bool(False), int(0), 640, 480)
    if obj_text and (not obj_text.isspace()) and (obj_text != prev_obj_text):
      prev_obj_text = obj_text
      gTTS(text=obj_text, lang=language, slow=False).save("Text.mp3")
      os.system("mpg321 -q Text.mp3")

    # DistanceMeasure()

if __name__ == '__main__':
  main()