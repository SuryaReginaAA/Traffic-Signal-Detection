import streamlit as st
import os
import time
import cv2
import numpy as np
from PIL import Image
import tempfile
from model.yolo_model import YOLO
from tensorflow.keras.models import load_model

classifier = load_model('model/tl_classifier.h5')
from keras.preprocessing import image


def process_image(img):
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(img):
    test_image = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    test_image = np.array(test_image, dtype='float32')
    test_image = np.expand_dims(test_image, axis=0)# rgb 3d => expand rgb=>4D => maybe classifier needs 4d
    result = classifier.predict(test_image) # 2d array => [[0.]] or [[1.]]
    return result


def get_classes(file):
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def draw(image, boxes, scores, classes, all_classes):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        if cl == 9: # traffic light
            crop_area = image[left:bottom, top:right]
            result = predict_image(crop_area)
            if result[0][0] == 1:
                cv2.rectangle(image, (top, left), (right, bottom), ( 0, 0,255), 2)
                cv2.putText(image, 'Red - Stop', (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255), 2,
                            cv2.LINE_AA)
            elif result[0][0] == 0:
                cv2.rectangle(image, (top, left), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, 'Green - Proceed',
                            (top, left - 6),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2,
                            cv2.LINE_AA)
        else:
            continue

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    pimage = process_image(image)
    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    # print(image.shape)
    end = time.time()
    print('time: {0:.2f}s'.format(end - start))
    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_video(tfile, yolo, all_classes):
    camera = cv2.VideoCapture(tfile.name)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'H264') #'mpeg'

    # vout = cv2.VideoWriter()
    # wfile = tempfile.NamedTemporaryFile(delete=False)
    # vout.open(wfile.name, fourcc, 20, sz, True)

    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", "s1.mp4"), fourcc, 20, sz, True)

    stframe = st.empty()

    while camera.isOpened():
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        rgbimage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # cv2.imshow("detection", image)
        stframe.image(rgbimage)
        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(30) & 0xff == ord('q'):
            break

    # video_file = open("videos/res/s1.mp4", "rb").read()
    # st.video(open("videos/res/s1.mp4", "rb").read())

    vout.release()
    camera.release()

    cv2.destroyAllWindows()

    video_file = open("videos/res/s1.mp4", "rb").read()
    st.video(video_file)




@st.cache
def load_image(img):
    im = Image.open(img)
    return im


def main():
    st.title("Traffic Light Detection")
    choice = st.sidebar.selectbox("Choose file type", ["Image", "Video"])

    thresh = st.sidebar.slider('Confidence Threshold',0.00,1.00,0.4)

    yolo = YOLO(thresh, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)


    if choice == 'Image':
        image_file = st.file_uploader("Upload Image", type=['jpg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Uploaded Image")
            st.image(our_image)

        if st.button("Detect"):
            new_image = np.array(our_image.convert('RGB'))# uploaded image is stored in some other format we need to convert it into RGB
            # new_image = cv2.cvtColor(new_image,cv2.COLOR_RGB2BGR)
            processed_image = detect_image(new_image, yolo, all_classes)
            st.text("Detections")
            st.image(processed_image)
    elif choice == 'Video':
        video_file = st.file_uploader("Upload Video", type=['mp4'])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

        if st.button("Detect"):
            detect_video(tfile,yolo,all_classes)

if __name__ == '__main__':
    main()