try:
    import cv2
    print("✅ OpenCV (cv2) is working.")
except ImportError as e:
    print("❌ OpenCV missing:", e)

try:
    import insightface
    print("✅ InsightFace is working.")
except ImportError as e:
    print("❌ InsightFace missing:", e)

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO is working.")
except ImportError as e:
    print("❌ Ultralytics missing:", e)
