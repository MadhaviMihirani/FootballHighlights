from PIL import Image
import pytesseract
import cv2
import numpy as np
import logging
import time

# input video file
from numpy.distutils.fcompiler import none

cap = cv2.VideoCapture('a.mp4')
# get frame rate
fps = cap.get(cv2.CAP_PROP_FPS)


def localize_scoreboard_image(image, count, null=None):
    # Read a snapshot image from the video and convert to gray
    # snapshot_image = cv2.imread(self.export_image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    canny_points = cv2.Canny(grayscale_image, 98, 240, 1)

    # Crop upper corner of Canny image
    canny_upper_left_corner = canny_points[2:95, 0:500]

    # cv2.imshow('canny_upper_left_corner', canny_upper_left_corner)
    # cv2.waitKey(0)

    # Localize the scoreboard edges on Canny image
    idx_lst = []

    for i in range(canny_upper_left_corner.shape[0]):
        pxl_cnt = 0
        for j in range(canny_upper_left_corner.shape[1]):
            if pxl_cnt > 100:
                idx_lst.append(i)

            if canny_upper_left_corner[i, j] == 255:
                pxl_cnt += 1
    upper_row = min(idx_lst)
    lower_row = max(idx_lst)

    # Export the localized scoreboard
    scoreboard_image = grayscale_image[upper_row:lower_row, 0:500]

    # if scoreboard_image.any() != null:
    #     print("scorebox is found")
    # else:
    #     print("scorebox is not  found")


def split_scoreboard_image(image, scoreboard_image):
    time_image = np.array(scoreboard_image)[:, 130: 193]
    print(time_image)
    # cv2.imshow('scoreboard_table_left', time_image)
    if scoreboard_image in globals():
        cv2.imwrite('time_table.png', time_image)

    else:
        print("Image is empty")

    teams_goals_image = scoreboard_image[:, 193:]
    cv2.imwrite('teams_goals_table.png', teams_goals_image)
    # cv2.imshow('scoreboard_table_right', teams_goals_image)
    # cv2.imshow('scoreboard_table_left',self.time_image)
    # cv2.imshow('scoreboard_table_right',self.teams_goals_image)
    # cv2.waitKey(0)

    return time_image, teams_goals_image


def enlarge_scoreboard_images(enlarge_ratio, time_image, teams_goals_image):
    time_image = cv2.resize(time_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)
    teams_goals_image = cv2.resize(teams_goals_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)

    cv2.imshow('time_image_enlarged', time_image)
    cv2.imshow('teams_goals_enlarged', teams_goals_image)
    # cv2.waitKey(0)

    return time_image, teams_goals_image


def _get_time_from_image(time_image):
    ret, threshed_img = cv2.threshold(time_image, 200, 255, cv2.THRESH_BINARY_INV)
    # nonzero_pxls = np.count_nonzero(threshed_img)
    # pxls_limit = np.size(threshed_img) / 4
    #
    # if nonzero_pxls < pxls_limit:
    #     time_image = cv2.GaussianBlur(time_image, (3, 3), 0)
    #
    # ret, time_image = cv2.threshold(time_image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    #
    # if nonzero_pxls < pxls_limit:
    #     time_image = cv2.erode(time_image, kernel, iterations=1)
    # else:
    #     time_image = cv2.dilate(time_image, kernel, iterations=1)
    cv2.imshow('Gray image', time_image)
    # cv2.imshow('Original image1',image)
    # cv2.imshow('Original image',gray)
    # closing
    # inv = cv2.subtract(255, teams_time_image)

    cv2.imwrite('teams_time_ocr_ready.png', time_image)
    teams_time_text = pytesseract.image_to_string(Image.open('teams_time_ocr_ready.png'))
    logging.info('Teams and time OCR text: {}'.format(teams_time_text))
    # cv2.imshow('teams_time_OCR_read',  teams_time_text)
    # print('Teams and time OCR text:',teams_time_text)

    if teams_time_text is not none:
        return True
    return False


def _get_teams_goals_from_image(teams_goals_image):
    # Applying Thresholding for Teams goals OCR preprocess
    ret, teams_goals_image = cv2.threshold(teams_goals_image, 180, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('teams_goals_ocr_ready.png', teams_goals_image)
    teams_goals_text = pytesseract.image_to_string(Image.open('teams_goals_ocr_ready.png'))
    logging.info('Teams and goals OCR text: {}'.format(teams_goals_text))
    # cv2.imshow('teams_goals_OCR_read', teams_goals_image)
    print('Teams and goals OCR text: {}'.format(teams_goals_text))

    cv2.imshow('teams_goals_OCR_read', teams_goals_image)
    # cv2.waitKey(0)

    return teams_goals_text


def get_scoreboard_texts(time_text, teams_goals_text, time_image, teams_goals_image):
    # Read text values using Tesseract OCR
    time_text_exists = _get_time_from_image(time_image)
    teams_goals_text_exists = _get_teams_goals_from_image(teams_goals_image)

    scoreboard_texts = []
    # Use values on successful read
    if time_text_exists and teams_goals_text_exists:
        scoreboard_texts.append(time_text)
        scoreboard_texts.append(teams_goals_text)
        scoreboard_texts = np.array(scoreboard_texts)

    return scoreboard_texts


count = 0
if not cap.isOpened():
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED!')

while cap.isOpened():
    RET, FRAME = cap.read()

    if RET:
        # time.sleep(1 / fps)  # to run according to frame rate otherwise it go on highSpeed
        # convert BGR to GrayScale
        scoreboard_image = localize_scoreboard_image(FRAME, count)
        time_image, teams_goals_image = split_scoreboard_image(FRAME, scoreboard_image)
        # detectText(FRAME)
        time_image, teams_goals_image = enlarge_scoreboard_images(2, time_image, teams_goals_image)
        teams_time_text = _get_time_from_image(time_image)
        teams_goals_text = _get_teams_goals_from_image(teams_goals_image)

        scoreboard_text_values = get_scoreboard_texts(teams_time_text, teams_goals_text, time_image, teams_goals_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    count = count + 1
cap.release()

"""
       Returns an array of strings including OCR read time, teams and goals texts.
       :return: numpy array 'scoreboard_texts'
                scoreboard_texts[0] : time text value
                scoreboard_texts[1] : teams and goals text value

       """
