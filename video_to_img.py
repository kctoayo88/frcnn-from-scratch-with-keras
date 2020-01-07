import os
import cv2

input_video_file = "C:\\Users\\kctoa\\Desktop\\VScode\\frcnn-from-scratch-with-keras\\test_video.mp4"
img_path = "C:\\Users\\kctoa\\Desktop\\VScode\\frcnn-from-scratch-with-keras\\input"
frame_rate = 30.0

print("Converting video to images..")
cam = cv2.VideoCapture(input_video_file)
counter = 0
while True:
	flag, frame = cam.read()
	if flag:
		print('Converting: ', counter)
		cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'), frame)
		counter = counter + 1
	else:
		break
	if cv2.waitKey(1) == 27:
		break
		# press esc to quit
cv2.destroyAllWindows()