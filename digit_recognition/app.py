import cv2
import numpy as np
from keras.models import load_model

model = load_model('models/mnistCNN.h5')

def get_numbers(y_pred):
    for number, per in enumerate(y_pred[0]):
        if per != 0:
            final_number = str(int(number))
            per = round((per * 100), 2)
            return final_number, per

def predict_image(image_path):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gau = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret, thresh = cv2.threshold(img_gau, 80, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    edged = cv2.Canny(dilation, 50, 250)
    contours, hierachy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_str = ''
    per = ''
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) > 2500:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
        new_img = thresh[y:y + h, x:x + w]
        new_img2 = cv2.resize(new_img, (28, 28))
        im2arr = np.array(new_img2)
        im2arr = im2arr.reshape(1,28,28,1)
        y_pred = model.predict(im2arr)
        print(y_pred)
        num,per = get_numbers(y_pred)
        num_str = '['+str(num) +']'

    y_p = str('Predicted Value is '+str(num_str))
    print(y_p)

predict_image("img.jpg")
