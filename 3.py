import cv2


def detect_number_plate(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce high frequency noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply Canny edge detection
    edged = cv2.Canny(blur, 10, 250)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and filter out the ones that are not number plate-like
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        # Check if the aspect ratio is within the specified range and if the contour has enough area
        if aspect_ratio > 2.5 and aspect_ratio < 6.0 and cv2.contourArea(contour) > 500:
            return image[y:y + h, x:x + w]
    return None


# Load the image and detect the number plate
image = cv2.imread("3.jpg")
number_plate = detect_number_plate(image)

if number_plate is not None:
    cv2.imshow("Number Plate", number_plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No number plate was detected.")
