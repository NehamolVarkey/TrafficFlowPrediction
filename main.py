import cv2
import datetime
import imutils
import numpy as np
from nms import non_max_suppression_fast
from centroidtracker import CentroidTracker
from datetime import date, time

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

import pandas as pd

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def main():
    cap = cv2.VideoCapture('video/Traffic3.mp4')
    opc_count = 0
    object_id_list = []
    my_dict = {"Counter": [], "In_time": []}

    while True:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=600)

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        get_hour = datetime.datetime.now()
        detector.setInput(blob)
        car_detections = detector.forward()
        rects = []
        for i in np.arange(0, car_detections.shape[2]):
            confidence = car_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(car_detections[0, 0, i, 1])

                if CLASSES[idx] != "car":
                    continue

                car_box = car_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = car_box.astype("int")
                rects.append(car_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if (datetime.datetime.now().minute > 0):

                if objectId not in object_id_list:
                    my_dict["Counter"].append(objectId)
                    now = datetime.datetime.now()
                    # hr = datetime.strftime("11/03/22 14:23", "%d/%m/%y %H:%M")
                    time = now.strftime("%y-%m-%d %H:%M:%S")
                    # print(hr)
                    my_dict["In_time"].append(str(time))

                    object_id_list.append(objectId)

        opc_count = len(object_id_list)

        opc_txt = "Car Count: {}".format(opc_count)
        # dict2.update({: "Scala"})

        cv2.putText(frame, opc_txt, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        cv2.imshow("Application", frame)
        # frame1 = frame2
        key = cv2.waitKey(1)
        if key == ord('q'):
            print(my_dict)
            print(opc_txt)
            df = pd.DataFrame.from_dict(my_dict)
            # df.set_index('In_time', inplace=True)
            # print(df[df["In_time"]<"16:46:00"])
            df.to_csv('car_counter.csv', index=False)
            # df=df.loc[df['In_time']]
            # df=df[(df.index.hour>13)]
            # print(df)
            # ydf.loc[df['In_time'].dt.time > time(17,00)]
            break

    cv2.destroyAllWindows()


main()
