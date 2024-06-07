
cv2.polylines(frame, [np.array(zone, np.int32)], isClosed=True, color=(255, 0, 0, 10), thickness=2)

# Define the dropoff zone coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
zone = [(440, 290), (490, 300), (410, 380), (330, 360)]

# For video file
#cap = cv2.VideoCapture('static/video/c2.mp4')

# For IP camera
cap = cv2.VideoCapture('http://100.84.151.191:4747/video')