import logging

import PIL
import cv2
import numpy as np
import pytesseract
import time
from PIL import Image
# input video file
from numpy.distutils.fcompiler import none
logging.basicConfig(level=logging.INFO)

cap = cv2.VideoCapture('c.mp4')
# get frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_length)

count = 0
def detectTeamInfo(image, null=None):

    # convert to grey scale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # linear contrast stretching
    minmax_img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(minmax_img, (9, 9), 0)
    # canny = cv2.Canny(blurred, 120, 255, 1)
    # use sobel-x operation
    sobel_img_x = cv2.Sobel(blurred, cv2.CV_8U, 1, 0, ksize=3)
    # threshold
    retval, threshold = cv2.threshold(sobel_img_x, 200, 255, cv2.THRESH_BINARY)
    # Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(11, 2), anchor=(-1, -1))
    # kernel = np.ones((1, 1), np.uint8)
    img_dilate = cv2.morphologyEx(threshold, cv2.MORPH_DILATE, kernel, anchor=(-1, -1), iterations=2,
                                  borderType=cv2.BORDER_REFLECT, borderValue=255)
    # cv2.imshow("dialte", img_dilate)
    cv2.imshow("frame", image)
    y = 85
    x = 80
    h = 600
    w = 1000
    crop_img = image[y:y + h, x:x + w]
    cv2.imshow("cropped", crop_img)
    # cap.set(1, 600);  # Where frame_no is the frame you want
    # ret, crop_img = cap.read()  # Read the frame
    # cv2.imshow('window_name', crop_img)
    # test_image = cv2.imwrite('test.jpg', crop_img)
    frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    if 0 <= frame_pos <= 3000:
        area1 =  detectText(crop_img)
        if(area1 == 1):
            print("Team LINE-UP")

    elif 75000 <= frame_pos <= 75600:
        area2 = detectText(crop_img)
        if (area2 == 1):
            print("1st Half Summary")

    elif 151000 <= frame_pos <= 151500:
        area3 = detectText(crop_img)
        if (area3 == 1):
            print("Match Summary")

    else:
        print("False")

    # detectText(crop_img)



    height = np.size(img_dilate, 0)
    width = np.size(img_dilate, 1)

    # print(height)
    # print(width)
    # Find Contours
    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # GeaomatricalConstraints
    # list = []
    # for contour in contours:
    #     brect = cv2.boundingRect(contour)  # brect = (x,y,w,h)
    #     ar = brect[2] / brect[3]
    #
    #     # if ar >= 2.7 and brect[2] >= 40 and 17 <= brect[3] <= 60:
    #     if ar >= 0.2 and brect[2] < 320 and   brect[3] < 720:  # 2->w 3->h
    #         list.append(brect)
    #
    # for r in list:
    #     # draw region of interest
    #     infobox = cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (250, 0, 0), 2)
    #     # cv2.imwrite('{}.png'.format(r), scoreboard_image)
    #
    #     cv2.imshow("infobox", infobox)


def detectText(image, null=None):
    count = 0
    flag = 0
    # ret, teams_info_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    #
    # inv = cv2.subtract(255, teams_info_image)
    #
    # cv2.imwrite('teams_info_ocr_ready.png', inv)
    # teams_info_text = pytesseract.image_to_string(Image.open('teams_info_ocr_ready.png'))
    # log = logging.info('Teams line up OCR text: {}'.format(teams_info_text))
    # # print('Teams line up OCR text:', teams_info_text)
    #
    # if teams_info_text != null:
    #     return True
    # else:
    #
    #     return False
    # image = cv2.imread('image1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        inv = cv2.subtract(255, thresh)

        # cv2.imwrite('teams_info_ocr_ready.png', inv)
        # teams_info_text = pytesseract.image_to_string(Image.open('teams_info_ocr_ready.png'))
        # log = logging.info('Teams line up OCR text: {}'.format(teams_info_text))
        # num_lines = sum(1 for line in open('logging\_init.py'))
        # if num_lines is none:
        #     num_lines = str(num_lines)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 3)

            text = pytesseract.image_to_string(image)
            # num_lines = sum(1 for line in open(text))
            if text != null:
                print(text)
                for i in text:
                    count += 1
                if count > 150:
                    print(count)
                    flag = 1

            # ROI = image[y:y+h, x:x+w]
            # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            # ROI_number += 1
            # return textArea

    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', image)
    return flag
    cv2.waitKey()


if not cap.isOpened():
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED!')
# count=0
while cap.isOpened():

    RET, FRAME = cap.read()

    if RET:
        # time.sleep(1 / fps)  # to run according to frame rate otherwise it go on highSpeed
        # convert BGR to GrayScale
        detectTeamInfo(FRAME)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()




# Load image, grayscale, Gaussian blur, adaptive threshold






