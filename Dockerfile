FROM ghcr.io/luxonis/robothub-base-app:ubuntu-depthai-main


ADD script.py .

ADD age-gender-recognition.blob .
ADD face-detection.blob .

ARG FILE=app.py
ADD $FILE run.py
