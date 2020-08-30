import cv2
import numpy as np
import time
from djitellopy import Tello

#set points (center of the frame coordinates in pixels)
rifX = 960/2
rifY = 720/2

Kp_X = 0.1
Ki_X = 0.0
Kd_X = 0

Kp_Y = 0.2

Tc = 0.05


integral_X = 0
derivative_X = 0
error_X = 0
previous_error_X = 0

centroX_pre = rifX
centroY_pre = rifY


net = cv2.dnn.readNetFromCaffe("/home/vittorio/MobileNetSSD_deploy.prototxt.txt", "/home/vittorio/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


drone = Tello()  # declaring drone object
time.sleep(3.0)
print("Connecting...")
drone.connect()
time.sleep(2.0)
print("BATTERY: ")
print(drone.get_battery())

print("Loading...")
time.sleep(1.0)
drone.streamon()  # start camera streaming

print("Takeoff...")
drone.takeoff()

time.sleep(5.0)

while True:
	start = time.time()
	frame = drone.get_frame_read().frame

	cv2.circle(frame, (int(rifX), int(rifY)), 1, (0,0,255), 10)

	h,w,channels = frame.shape

	blob = cv2.dnn.blobFromImage(frame,
		0.007843, (180, 180), (0,0,0),True, crop=False)

	net.setInput(blob)
	detections = net.forward()


	for i in np.arange(0, detections.shape[2]):

		idx = int(detections[0, 0, i, 1])
		confidence = detections[0, 0, i, 2]

		if CLASSES[idx] == "person" and confidence > 0.5:
		
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				colors[idx], 2)
			#draw the center of the person detected
			centroX = (startX + endX)/2
			centroY = (2*startY + endY)/3


			centroX_pre = centroX
			centroY_pre = centroY


			cv2.circle(frame, (int(centroX), int(centroY)), 1, (0,0,255), 10)

			error_X = -(rifX - centroX)
			error_Y = rifY - centroY

			cv2.line(frame, (int(rifX),int(rifY)), (int(centroX),int(centroY)), (0,255,255),5 )


			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

			integral_X = integral_X + error_X*Tc
			derivative_X = (error_X - previous_error_X)/Tc
			uX = Kp_X*error_X + Ki_X*integral_X + Kd_X*derivative_X
			previous_error_X = error_X

			uY = Kp_Y*error_Y

			print(error_X)
			drone.send_rc_control(0,0,uY,round(uX))
			#break when a person is recognized

			break	


		else: #if nobody is recognized take centerX and centerY of the previous frame
			centroX = centroX_pre
			centroY = centroY_pre
			cv2.circle(frame, (int(centroX), int(centroY)), 1, (0,0,255), 10)

			error_X = -(rifX - centroX)
			error_Y = rifY - centroY

			cv2.line(frame, (int(rifX),int(rifY)), (int(centroX),int(centroY)), (0,255,255),5 )

			integral_X = integral_X + error_X*Tc
			derivative_X = (error_X - previous_error_X)/Tc
			uX = Kp_X*error_X + Ki_X*integral_X + Kd_X*derivative_X
			previous_error_X = error_X

			uY = Kp_Y*error_Y
			
			print(error_X)
			drone.send_rc_control(0,0,uY,round(uX))

			continue

	
	cv2.imshow("Frame", frame)

	end = time.time()
	elapsed= end-start
	if Tc - elapsed > 0:
		time.sleep(Tc - elapsed)
	end_ = time.time()
	elapsed_ = end_ - start
	fps = 1/elapsed_
	print("FPS: ",fps)


	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cv2.destroyAllWindows()
print(drone.get_battery())
drone.streamoff()
drone.land()
print("Landing...")
print("BATTERY: ")
print(drone.get_battery()) 
