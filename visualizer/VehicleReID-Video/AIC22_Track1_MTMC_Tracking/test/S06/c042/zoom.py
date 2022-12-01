import cv2

cap = cv2.VideoCapture('vdo.avi')
while True:
	ret, frame = cap.read()
	if not ret:
		break

	height,width,_ = frame.shape

	frame1 = frame[:height//2, :width//2]
	frame2 = frame[:height//2, width//2:]
	frame3 = frame[height//2:, :width//2]
	frame4 = frame[height//2:, width//2:]

	frame1 = cv2.resize(frame1, (frame1.shape[1]*4, frame1.shape[0]*4))
	frame2 = cv2.resize(frame2, (frame2.shape[1]*4, frame2.shape[0]*4))
	frame3 = cv2.resize(frame3, (frame3.shape[1]*4, frame3.shape[0]*4))
	frame4 = cv2.resize(frame4, (frame4.shape[1]*4, frame4.shape[0]*4))

	cv2.imshow('frame1', frame1)
	cv2.imshow('frame2', frame2)
	cv2.imshow('frame3', frame3)
	cv2.imshow('frame4', frame4)

	_key_ = cv2.waitKey(1)
	if _key_ == ord('q'):
		break

cv2.destroyAllWindows()
