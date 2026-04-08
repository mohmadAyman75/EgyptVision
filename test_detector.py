import cv2
from detector import detect_objects

cap = cv2.VideoCapture(0)

print("شغال — اضغط Ctrl+C للخروج")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    objects = detect_objects(frame)

    if objects:
        print("\n--- Frame ---")
        for obj in objects:
            status = "🔴 بيقرب!" if obj.approaching else ""
            print(f"[ID {obj.track_id}] {obj.label} | {obj.direction} | {obj.distance} {status}")
    # رسم الـ boxes على الشاشة للتأكد
    from ultralytics import YOLO
    model_vis = YOLO("yolov8l.pt")
    results = model_vis.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Egypt Vision AI - Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()