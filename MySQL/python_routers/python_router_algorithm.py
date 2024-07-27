import os
import cv2
import time
import base64
import imutils
import pytesseract
import numpy as np
from PIL import Image

from fastapi import Request, APIRouter
from fastapi.responses import JSONResponse

# Initialising the "router_algorithm" Router
router_algorithm = APIRouter(prefix="/algorithm")

# Vehicle Detection (router_algorithm)
@router_algorithm.get("/vehicle-detection")
async def router_algorithm_vehicledetection(request: Request, file_type: str = None, file_number: int = None):
    # Checking if Query Parameters are Present
    for parameter in [file_type, file_number]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "file_type" Query Parameter
    if (file_type not in ["images", "videos"]):
        # Returning the Error
        return JSONResponse({"Error": "The 'file_type' query parameter must be either 'images' or 'videos'.", "Status Code": 400}, status_code=400)
    else:
        # Checking if "file_type" is "images" or "videos"
        if (file_type == "images"):
            # Checking the Value of the "file_number" Query Parameter
            if (int(file_number) not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
                # Returning the Error
                return JSONResponse({"Error": "The 'file_number' query parameter must be a number from 1 to 10.", "Status Code": 400}, status_code=400)
        elif (file_type == "videos"):
            # Checking the Value of the "file_number" Query Parameter
            if (int(file_number) not in [1]):
                # Returning the Error
                return JSONResponse({"Error": "The 'file_number' query parameter must be 1.", "Status Code": 400}, status_code=400)

    # Setting the File Extension of the File
    if (file_type == "images"): file_extension = ".jpg"
    elif (file_type == "videos"): file_extension = ".mp4"

    # Fetching the Sample Media File Path
    file_path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/sample_media/vehicle_detection/{0}/{1}".format(file_type, str(file_number) + file_extension)

    # Checking the Value of the "file_type" Query Parameter
    if (file_type == "images"):
        # Converting the Image to Base64
        with open(file_path, "rb") as image_base64:
            image_data_base64 = base64.b64encode(image_base64.read())

        # Opening the Image
        image = Image.open(file_path)
        image = image.resize((450, 250))
        image_arr = np.array(image)

        # Converting Image to Greyscale
        grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        Image.fromarray(grey)

        # Blurring the Image
        blur = cv2.GaussianBlur(grey, (5,5), 0)
        Image.fromarray(blur)

        # Dilating the Image
        dilated = cv2.dilate(blur, np.ones((3,3)))
        Image.fromarray(dilated)

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 
        Image.fromarray(closing)

        # Identifying the Cars
        cars = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/models/haarcascade_car.xml").detectMultiScale(closing, 1.1, 1)

        # Counting the Cars
        count = 0

        for (x, y, w, h) in cars:
            # Incrementing "count"
            count += 1

            # Drawing a Rectange Around Each Car
            cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)

        # Displaying the Image
        cv2.imshow("Vehicle Detection (Image)", image_arr)
        cv2.waitKey(0)

        # Returning the Message
        return JSONResponse({"Message": "Successfully detected vehicles in the image.", "Number of Vehicles": count, "Image Data Base64": str(image_data_base64), "Status Code": 200}, status_code=200)
    elif (file_type == "videos"):
        # Opening the Video
        video = cv2.VideoCapture(file_path)

        # Opening the Video and Processing
        while video.isOpened():
            time.sleep(.05)

            # Reading the First Frame
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Identifying the Cars
            cars = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/models/haarcascade_car.xml").detectMultiScale(gray, 1.4, 2)

            for (x,y,w,h) in cars:
                # Drawing a Rectange Around Each Car
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

                # Displaying Each Frame
                cv2.imshow("Vehicle Detection (Video)", frame)

            # Clicking "q" Closes the Video
            if cv2.waitKey(1) == ord("q"):
                break

        # Stopping OpenCV
        video.release()
        cv2.destroyAllWindows()

        # Returning the Message
        return JSONResponse({"Message": "Successfully detected vehicles in the video.", "Status Code": 200}, status_code=200)

# License Plate Detection (router_algorithm)
@router_algorithm.get("/license-plate-detection")
async def router_algorithm_licenseplatedetection(request: Request, file_type: str = None, file_number: int = None, tesseract_path: str = None):
    # Checking if Query Parameters are Present
    for parameter in [file_type, file_number, tesseract_path]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "file_type" Query Parameter
    if (file_type not in ["images"]):
        # Returning the Error
        return JSONResponse({"Error": "The 'file_type' query parameter must be 'images'.", "Status Code": 400}, status_code=400)
    else:
        # Checking if "file_type" is "images"
        if (file_type == "images"):
            # Checking the Value of the "file_number" Query Parameter
            if (int(file_number) not in [1, 2, 3, 4, 5, 6, 7, 8]):
                # Returning the Error
                return JSONResponse({"Error": "The 'file_number' query parameter must be a number from 1 to 8.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "tesseract_path" Query Parameter
    if (tesseract_path == None):
        # Returning the Error
        return JSONResponse({"Error": "The 'tesseract_path' query parameter must point to a valid 'tesseract.exe' file.", "Status Code": 400}, status_code=400)
    else:
        # Checking if the Value of the "tesseract_path" Query Parameter Exists
        if (not os.path.exists(tesseract_path)):
            # Returning the Error
            return JSONResponse({"Error": "The 'tesseract_path' query parameter must point to a valid 'tesseract.exe' file. If Tesseract-OCR is not functioning properly or not installed on your computer, get it from https://github.com/UB-Mannheim/tesseract/wiki.", "Status Code": 400}, status_code=400)

    # Setting the File Extension of the File
    if (file_type == "images"): file_extension = ".jpg"
    elif (file_type == "videos"): file_extension = ".mp4"

    # Fetching the Sample Media File Path
    file_path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/sample_media/license_plate/{0}/{1}".format(file_type, str(file_number) + file_extension)

    # Connecting to Tesseract-OCR
    try:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    except:
        # Returning the Error
        return JSONResponse({"Error": "The 'tesseract_path' query parameter must point to a valid 'tesseract.exe' file. If Tesseract-OCR is not functioning properly or not installed on your computer, get it from https://github.com/UB-Mannheim/tesseract/wiki.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "file_type" Query Parameter
    if (file_type == "images"):
        # Converting the Image to Base64
        with open(file_path, "rb") as image_base64:
            image_data_base64 = base64.b64encode(image_base64.read())

        # Reading the Image
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (600,400))

        # Converting the Image to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 13, 15, 15)

        # Finding & Counting the Contours
        edged = cv2.Canny(gray, 30, 200)
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCount = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                screenCount = approx
                break

        # Drawing the Contours
        if (screenCount is not None):
            cv2.drawContours(img, [screenCount], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCount], 0,255, -1,)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        # Fetching the License Plate Number
        try:
            text = pytesseract.image_to_string(Cropped, config="--psm 11")
        except:
            # Returning the Error
            return JSONResponse({"Error": "The 'tesseract_path' query parameter must point to a valid 'tesseract.exe' file. If Tesseract-OCR is not functioning properly or not installed on your computer, get it from https://github.com/UB-Mannheim/tesseract/wiki.", "Status Code": 400}, status_code=400)

        # Stopping OpenCV
        cv2.destroyAllWindows()

        # Fetching the License Plate Number
        if ((text[-1] == " ") or (text[-1] == "\n")):
            text = text.replace(text[-1], "")

        # Reading the Image (Display)
        img_display = cv2.imread(file_path)

        # Converting the Image to Grayscale (Display)
        gray_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        # Finding the Plates (Display)
        plates_display = cv2.CascadeClassifier(os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/models/haarcascade_russian_plate_number.xml").detectMultiScale(gray_display, 1.2, 5)

        # Displaying Each License Plate (Display)
        for (x,y,w,h) in plates_display:
            cv2.rectangle(img_display, (x,y), (x+w, y+h), (0,255,0), 2)
            gray_plates = gray_display[y:y+h, x:x+w]
            color_plates = img_display[y:y+h, x:x+w]

            cv2.imshow("Vehicle (Image)", img_display)
            cv2.imshow("License Plate (Image)", gray_plates)
            cv2.waitKey(0)

        # Returning the Message
        return JSONResponse({"Message": "Successfully detected the license plate number in the image.", "License Plate Number": text, "Image Data Base64": str(image_data_base64), "Status Code": 200}, status_code=200)

# People Detection (router_algorithm)
@router_algorithm.get("/people-detection")
async def router_algorithm_peopledetection(request: Request, file_type: str = None, file_number: int = None):
    # Checking if Query Parameters are Present
    for parameter in [file_type, file_number]:
        if (parameter in [None, "", " "]):
            # Returning the Error
            return JSONResponse({"Error": "The query parameters are missing or not present. Please try again.", "Status Code": 400}, status_code=400)

    # Checking the Value of the "file_type" Query Parameter
    if (file_type not in ["images", "videos"]):
        # Returning the Error
        return JSONResponse({"Error": "The 'file_type' query parameter must be either 'images' or 'videos'.", "Status Code": 400}, status_code=400)
    else:
        # Checking if "file_type" is "images" or "videos"
        if (file_type == "images"):
            # Checking the Value of the "file_number" Query Parameter
            if (int(file_number) not in [1]):
                # Returning the Error
                return JSONResponse({"Error": "The 'file_number' query parameter must be 1.", "Status Code": 400}, status_code=400)
        elif (file_type == "videos"):
            # Checking the Value of the "file_number" Query Parameter
            if (int(file_number) not in [1, 2, 3, 4, 5]):
                # Returning the Error
                return JSONResponse({"Error": "The 'file_number' query parameter must be a number from 1 to 5.", "Status Code": 400}, status_code=400)

    # Setting the File Extension of the File
    if (file_type == "images"): file_extension = ".jpg"
    elif (file_type == "videos"): file_extension = ".mp4"

    # Fetching the Sample Media File Path
    file_path = os.path.dirname(os.path.realpath(__file__)).replace(os.sep, "/") + "/sample_media/people_detection/{0}/{1}".format(file_type, str(file_number) + file_extension)

    # Initializing the HOG Descriptor
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Nested Function 1 - Detect
    def detect(frame):
        # Setting the Box Coordinates and Styles
        bounding_box_cordinates, weights =  detector.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)

        # Detecting the People
        person = 1

        for x,y,w,h in bounding_box_cordinates:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Person {person}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            person += 1

        # Placing Text on the Output
        cv2.putText(frame, f"Total People: {person-1}", (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)

        # Displaying the Output
        cv2.imshow("People Detection (Video)", frame)

        # Returning the Frame
        return frame

    # Nested Function 2 - Detect By Video
    def detect_by_video(file_path):
        # Initializing the Video and Capturing the Frames
        video = cv2.VideoCapture(file_path)
        check, frame = video.read()

        # Checking if the Video is Null
        if (check == False):
            raise Exception("There is something wrong with this video. Try again with another one.")

        # Opening the Video and Sending Video Data to the "detect()" Nested Function
        while video.isOpened():
            check, frame = video.read()

            if (check):
                frame = imutils.resize(frame, width=min(800, frame.shape[1]))
                frame = detect(frame)

                # Clicking "q" Closes the Video
                if (cv2.waitKey(1) == ord("q")):
                    break
            else:
                raise Exception("There is something wrong with this video. Try again with another one.")
                break

        # Stopping OpenCV
        video.release()
        cv2.destroyAllWindows()

    # Checking the Value of the "file_type" Query Parameter
    if (file_type == "images"):
        # Converting the Image to Base64
        with open(file_path, "rb") as image_base64:
            image_data_base64 = base64.b64encode(image_base64.read())

        # Reading the Image
        image = cv2.imread(file_path)

        # Detecting the Humans
        (humans, _) = detector.detectMultiScale(image, winStride=(10, 10), padding=(32, 32), scale=1.1)

        for (x, y, w, h) in humans:
            pad_w, pad_h = int(0.15 * w), int(0.01 * h)
            cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

        # Displaying the Image
        cv2.imshow("People Detection (Image)", image)
        cv2.waitKey(0)

        # Returning the Message
        return JSONResponse({"Message": "Successfully detected people in the image.", "Number of People": len(humans), "Image Data Base64": str(image_data_base64), "Status Code": 200}, status_code=200)
    elif (file_type == "videos"):
        # Sending the Video to the "detect_by_video()" Nested Function
        detect_by_video(file_path)

        # Returning the Message
        return JSONResponse({"Message": "Successfully detected people in the video.", "Status Code": 200}, status_code=200)