import cv2
import os

output_path = 'C:\\Users\\kctoa\\Desktop\\VScode\\frcnn-from-scratch-with-keras\\output'
video_path = 'C:\\Users\\kctoa\\Desktop\\VScode\\frcnn-from-scratch-with-keras\\test_output.mp4'
frame_rate = 25

print('Save to video...')

files = os.listdir(output_path)
files.sort(key=lambda x: int(x.split('.')[0]))

img0 = cv2.imread(os.path.join(output_path, files[0]))
height , width , layers =  img0.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

for path in files:
    full_path = os.path.join(output_path, path)
    print('Saving: ', full_path)
    img = cv2.imread(full_path)
    videowriter.write(img)

videowriter.release()
cv2.destroyAllWindows()