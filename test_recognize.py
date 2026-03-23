import cv2
from CORE.face_engine import load_db_features, detect_and_recognize

db_features = load_db_features()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    results = detect_and_recognize(frame, db_features)

    for r in results:

        x1, y1, x2, y2 = r["bbox"]

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,
                    f'{r["name"]} ({r["score"]:.2f})',
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1)==27:
        break