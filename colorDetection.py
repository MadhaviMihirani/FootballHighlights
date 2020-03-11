import cv2
import numpy as np

# input video file
cap = cv2.VideoCapture('e.mp4')
# get frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# method for text detection
def detectLowerBar(image):
    global ROI
    # convert to grey scale
    img = cv2.cvtColor(~image[0:1280, 0:800], cv2.COLOR_BGR2GRAY)
    # linear contrast stretching
    contrast_img = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX)
    # canny edge detection
    canny = cv2.Canny(contrast_img, 118, 240, 1)
    kernel = np.ones((1, 1), np.uint8)

    img_dilate = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel, anchor=(-1, -1), iterations=2,
                                  borderType=cv2.BORDER_REFLECT, borderValue=255)
    cv2.imshow("dialte", img_dilate)
    # Find Contours
    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # GeaomatricalConstraints
    list = []
    for contour in contours:
        brect = cv2.boundingRect(contour)  # brect = (x,y,w,h)
        ar = brect[2] / brect[3]

        if ar > 2.7 and brect[2] > 270 and 50 <= brect[3] <= 300:  # 2->w 3->h
            list.append(brect)
    for r in list:
        # draw region of interest
        ROI = cv2.rectangle(image, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (250, 0, 0), 2)
        # cv2.imwrite('{}.png'.format(r), ROI)
    # cv2.imshow('frame', image[500:1280,50:1000])
    cv2.imshow('frame2', ROI)
    # cv2.imshow('frame2', img_dilate)
    return ROI


def split_info_image(ROI):

        color_card_image = np.array(ROI)[600:1280, 350:450]
        # cv2.imwrite('teams_goals_table.png', color_card_image)
        # cv2.resizeWindow(color_card_image)
        cv2.imshow('color_card_image', color_card_image)
        ## DEBUG
        # cv2.imshow('scoreboard_table_left',self.time_image)
        # cv2.imshow('scoreboard_table_right',self.teams_goals_image)
        # cv2.waitKey(0)
        #
        return color_card_image


# method for color detection
def detectColor(color_card_image):

    # converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)
    hsv = cv2.cvtColor(color_card_image, cv2.COLOR_BGR2HSV)

    # defining the range of Yellow color
    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    # definig the range of red color
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # finding the range yellow colour in the image
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    red = cv2.inRange(hsv, red_lower, red_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")
    yellow = cv2.dilate(yellow, kernal)
    res1 = cv2.bitwise_and(color_card_image, color_card_image, mask=yellow)
    # print(res1)
    red = cv2.dilate(red, kernal)
    res2 = cv2.bitwise_and(color_card_image, color_card_image, mask=red)

    # Tracking Colour (Yellow)
    contours, hierarchy = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(color_card_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # cv2.imshow("Color Tracking", img)
            if img.any():

                print("Yellow")

        # Tracking Colour (red)
    contours, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 600:
            x, y, w, h = cv2.boundingRect(contour)
            img2 = cv2.rectangle(color_card_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # cv2.imshow("Color Tracking red", img2)
            if img2.any():

                print("Red")
    res = res1
    # cv2.imshow('frame', image[0:100,0:500])
    # # cv2.imshow('frame2', img_dilate)
    # Display results
    # img = cv2.flip(img, 1)
    cv2.imshow("Yellow", res)
    cv2.imshow("Red", res2)

    # cv2.imshow("Color Tracking", img)


if not cap.isOpened():
    print('ERROR FILE NOT FOUND OR WRONG CODEC USED!')

while cap.isOpened():
    RET, FRAME = cap.read()

    if RET:
        # time.sleep(1 / fps)  # to run according to frame rate otherwise it go on highSpeed
        # convert BGR to GrayScale
        # detectColor(FRAME)

        ROI = detectLowerBar(FRAME)
        color_card_image = split_info_image(ROI)
        detectColor(color_card_image)

        if cv2.waitKey(10) & 0xFF == 27:
            break
    else:
        break

cap.release()
