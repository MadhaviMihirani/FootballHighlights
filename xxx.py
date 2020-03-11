
from PIL import Image
from matplotlib import pyplot as plt

import pytesseract
import cv2
import tkinter as tk
import numpy as np
import logging
import time
import re
import threading


# input video file
cap = cv2.VideoCapture('g.mp4')
# get frame rate
fps = cap.get(cv2.CAP_PROP_FPS)



# def localize_scoreboard_image(image,count):
#     """
#     Finds the scoreboard table in the upper corner
#     using Canny edge detection, sets scoreboard_image
#     and exports the picture as 'scoreboard_table.png'
#
#     :return: True when scoreboard is found
#              False when scoreboard is not found
#     """
#
#     # Read a snapshot image from the video and convert to gray
#     # snapshot_image = cv2.imread(self.export_image_path)
#     grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Canny edge detection
#     canny_points = cv2.Canny(grayscale_image, 98, 240, 1)
#
#     # Crop upper corner of Canny image
#     canny_upper_left_corner = canny_points[2:95, 0:500]
#
#     # DEBUG
#     #cv2.imshow('canny_upper_left_corner', canny_upper_left_corner)
#     # cv2.waitKey(0)
#
#     # Localize the scoreboard edges on Canny image
#     idx_lst = []
#     # try:
#     for i in range(canny_upper_left_corner.shape[0]):
#         pxl_cnt = 0
#         for j in range(canny_upper_left_corner.shape[1]):
#             if pxl_cnt > 100:
#                 idx_lst.append(i)
#
#             if canny_upper_left_corner[i, j] == 255:
#                 pxl_cnt += 1
#     upper_row = min(idx_lst)
#     lower_row = max(idx_lst)
#     mad = len(idx_lst)
#     if (mad < 400):
#         print("no video")
#
#
#
#     else :
#     # mad = len(idx_lst)
#
#     # if (pxl_cnt == 0):
#
#
#     # Export the localized scoreboard
#         scoreboard_image = grayscale_image[upper_row:lower_row, 0:500]
#
#             #cv2.imwrite('test'+count+'.jpg', scoreboard_image)
#             # DEBUG
#             #cv2.imshow('scoreboard_table', scoreboard_image)
#
#         # if (scoreboard_image.size != 0):
#         return scoreboard_image
#
#
#     # except Exception as e:
#     #     if len(idx_lst) < 400:
#     #         logging.info(e)
#     #         logging.info("No scoreboard found!")
#     #         return False

def split_scoreboard_image(image,scoreboard_image):
        """
        Splits the scoeboard image into two parts, sets 'time_image' and 'teams_goals_image'
        and exports as 'time_table.png' and 'teams_goals_table.png'
        Left image represents the time.
        Right image represents the teams and goals.

        :return: -
        """
        # if (scoreboard_image.size != 0):
        # print(str(scoreboard_image))
        time_image = np.array(scoreboard_image)[:, 130:193]
        # print(time_image)
        #cv2.imshow('scoreboard_table_left', time_image)
        cv2.imwrite('time_table.png', time_image)

        teams_goals_image = scoreboard_image[:, 193:]
        cv2.imwrite('teams_goals_table.png', teams_goals_image)
        #cv2.imshow('scoreboard_table_right', teams_goals_image)
        ## DEBUG
        # cv2.imshow('scoreboard_table_left',self.time_image)
        # cv2.imshow('scoreboard_table_right',self.teams_goals_image)
        # cv2.waitKey(0)
        ##
        return time_image,teams_goals_image
        # return False

def enlarge_scoreboard_images( enlarge_ratio,time_image,teams_goals_image):
        """
        Enlarges 'time_table.png' and 'teams_goals_table.png'

        :param enlarge_ratio: Defines the enlarging size (e.g 2-3x)
        :return: -
        """
        if (scoreboard_image.size != 0):
             time_image = cv2.resize(time_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)
             teams_goals_image = cv2.resize(teams_goals_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)

             ## DEBUG
        # cv2.imshow('time_image_enlarged',self.time_image)
        # cv2.imshow('teams_goals_enlarged',self.teams_goals_image)
        # cv2.waitKey(0)
        ##
        return time_image,teams_goals_image

def _get_time_from_image(time_image):
        """
        Preprocesses time_image transformations for OCR.
        Exports 'time_ocr_ready.png' after the manipulations.
        Reads match time from 'time_ocr_ready.png' using Tesseract.
        Applies result to time_text.

        :return: True: string is found
                 False: string is not found
        """
        blurred = cv2.GaussianBlur(time_image, (3, 3), 0)
        ret, teams_time_image = cv2.threshold(time_image,200, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('Gray image', time_image)
        # cv2.imshow('Original image1',image)
        # cv2.imshow('Original image',gray)
        # closing
        inv = cv2.subtract(255, teams_time_image)


        cv2.imwrite('teams_time_ocr_ready.png', inv)
        teams_time_text = pytesseract.image_to_string(Image.open('teams_time_ocr_ready.png'))
        logging.info('Teams and time OCR text: {}'.format(teams_time_text))
        cv2.imshow('teams_time_OCR_read', inv)
        # print('Teams and time OCR text:',teams_time_text)

        return teams_time_text

def _get_teams_goals_from_image(teams_goals_image):
        """
        Preprocesses teams_goals_image with transformations for OCR.
        Exports 'teams_goals_ocr_ready.png' after the manipulations.
        Reads teams and goals information from 'teams_goals_ocr_ready.png' using Tesseract.
        Applies result to teams_goals_text.

        :return: True: string is found
                 False: string is not found

        """
        # HISTOGRAM
        # plt.hist(self.teams_goals_image.ravel(), 256, [0, 256])
        # plt.title("Teams goals OCR Image Histogram")
        # plt.show()
        #cv2.imshow('fin', teams_goals_image)
        # Applying Thresholding for Teams goals OCR preprocess
        blurred = cv2.GaussianBlur(teams_goals_image, (9, 9), 0)
        ret, teams_goals_image = cv2.threshold(teams_goals_image, 180, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite('teams_goals_ocr_ready.png',teams_goals_image)
        teams_goals_text = pytesseract.image_to_string(Image.open('teams_goals_ocr_ready.png'))
        logging.info('Teams and goals OCR text: {}'.format(teams_goals_text))
        #cv2.imshow('teams_goals_OCR_read', teams_goals_image)
        print('Teams and goals OCR text: {}'.format(teams_goals_text))
        ## DEBUG
        cv2.imshow('teams_goals_OCR_read',teams_goals_image)
        # cv2.waitKey(0)
        ##

        #if teams_goals_text is not None:
            #return True
        #return False
        return teams_goals_text
def get_scoreboard_texts(time_text,teams_goals_text,time_image,teams_goals_image):
        """
        Returns an array of strings including OCR read time, teams and goals texts.
        :return: numpy array 'scoreboard_texts'
                 scoreboard_texts[0] : time text value
                 scoreboard_texts[1] : teams and goals text value

        """

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




count=0
if not cap.isOpened():
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED!')

while cap.isOpened():
    RET, FRAME = cap.read()

    if RET:
        # time.sleep(1 / fps)  # to run according to frame rate otherwise it go on highSpeed
        # convert BGR to GrayScale
        # scoreboard_image=localize_scoreboard_image(FRAME,count)

        grayscale_image = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        canny_points = cv2.Canny(grayscale_image, 98, 240, 1)

        # Crop upper corner of Canny image
        canny_upper_left_corner = canny_points[2:95, 0:500]

        # DEBUG
        # cv2.imshow('canny_upper_left_corner', canny_upper_left_corner)
        # cv2.waitKey(0)

        # Localize the scoreboard edges on Canny image
        idx_lst = []
        # try:
        for i in range(canny_upper_left_corner.shape[0]):
            pxl_cnt = 0
            for j in range(canny_upper_left_corner.shape[1]):
                if pxl_cnt > 100:
                    idx_lst.append(i)

                if canny_upper_left_corner[i, j] == 255:
                    pxl_cnt += 1
        upper_row = min(idx_lst)
        lower_row = max(idx_lst)
        mad = len(idx_lst)
        if (mad < 200):
            print("no video")
            continue



        else:
            # mad = len(idx_lst)

            # if (pxl_cnt == 0):

            # Export the localized scoreboard
            scoreboard_image = grayscale_image[upper_row:lower_row, 0:500]

            # cv2.imwrite('test'+count+'.jpg', scoreboard_image)
            # DEBUG
            # cv2.imshow('scoreboard_table', scoreboard_image)

            # if (scoreboard_image.size != 0):
            # return scoreboard_image

            time_image,teams_goals_image=split_scoreboard_image(FRAME,scoreboard_image)
            # detectText(FRAME)
            time_image,teams_goals_image=enlarge_scoreboard_images(2, time_image, teams_goals_image)
            time_text=_get_time_from_image(time_image)
            teams_goals_text=_get_teams_goals_from_image(teams_goals_image)

            scoreboard_text_values=get_scoreboard_texts(time_text,teams_goals_text,time_image,teams_goals_image)
            # match_time_temp=cleanse_match_time(scoreboard_text_values)
            #print(match_time_temp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break
    count=count+1
cap.release()
