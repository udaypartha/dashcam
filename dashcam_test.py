import cv2

# Open camera (CSI camera)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()

if ret:
    cv2.imshow("Dashcam", frame)  # Opens a window on RealVNC desktop
    cv2.waitKey(0)                # Wait until you press any key
    cv2.destroyAllWindows()       # Close the window
else:
    print("❌ Failed to read frame")

cap.release()
