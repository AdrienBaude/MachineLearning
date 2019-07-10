
# Imports --------------------------------------------------------------------------------------------------------------

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Model --------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(train_generator.class_indices), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights("./data/model.h5")

# Selective Search --------------------------------------------------------------------------------------------------------------
 
cv2.setUseOptimized(True);
cv2.setNumThreads(4);
im = cv2.imread("./data/image.png")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
rects = ss.process()
numShowRects = 2000

boxes = []
indexes = []

while True:
	for i, rect in enumerate(rects):
		if (i < numShowRects):
			x, y, w, h = rect
			imgRect = im[y:y+h,x:x+w]
			imgRect = cv2.resize(imgRect, (32, 32)) 
			scores = model.predict([[imgRect]])[0]
			if np.sum(scores) == 1:
				boxes.append([x, y, x+w, y+h])
				indexes.append(np.argmax(scores))

		else:
			break
	break
 
# Suppression --------------------------------------------------------------------------------------------------------------

def non_max_suppression_fast(boxes, indexes):
	
	boxes = np.asarray(boxes)
	indexes = np.asarray(indexes)
	
	if len(boxes) == 0:
		return []
 
	pick = []
 
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		overlap = (w * h) / area[idxs[:last]]
 
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > 0.5)[0])))
			
	return boxes[pick], indexes[pick]
	
# Evaluation --------------------------------------------------------------------------------------------------------------

imOut = im.copy()
boxes, indexes = non_max_suppression_fast(boxes, indexes)
for box, idx in zip(boxes, indexes):
	if idx == 0:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2, cv2.LINE_AA)
	if idx == 1:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2, cv2.LINE_AA)
	if idx == 2:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2, cv2.LINE_AA)
	if idx == 3:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2, cv2.LINE_AA)
	if idx == 4:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2, cv2.LINE_AA)
	if idx == 5:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2, cv2.LINE_AA)
	if idx == 6:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (255, 127, 0), 2, cv2.LINE_AA)
	if idx == 7:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (127, 255, 0), 2, cv2.LINE_AA)
	if idx == 8:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (127, 0, 255), 2, cv2.LINE_AA)
	if idx == 9:
		cv2.rectangle(imOut, (box[0], box[1]), (box[2], box[3]), (255, 0, 127), 2, cv2.LINE_AA)

while True:
	cv2.imshow("Output", imOut)
	k = cv2.waitKey(0) & 0xFF
	break
cv2.destroyAllWindows()

	
	
