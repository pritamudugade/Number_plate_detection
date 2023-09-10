import cv2
import numpy as np
import pytesseract

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the license plate cascade classifier
cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Dictionary to map state codes to their names
states={"AN":"Andaman and Nicobar",
    "AP":"Andhra Pradesh","AR":"Arunachal Pradesh",
    "AS":"Assam","BR":"Bihar","CH":"Chandigarh",
    "DN":"Dadra and Nagar Haveli","DD":"Daman and Diu",
    "DL":"Delhi","GA":"Goa","GJ":"Gujarat",
    "HR":"Haryana","HP":"Himachal Pradesh",
    "JK":"Jammu and Kashmir","KA":"Karnataka","KL":"Kerala",
    "LD":"Lakshadweep","MP":"Madhya Pradesh","MH":"Maharashtra","MN":"Manipur",
    "ML":"Meghalaya","MZ":"Mizoram","NL":"Nagaland","OD":"Odissa",
    "PY":"Pondicherry","PN":"Punjab","RJ":"Rajasthan","SK":"Sikkim","TN":"TamilNadu",
    "TR":"Tripura","UP":"Uttar Pradesh", "WB":"West Bengal","CG":"Chhattisgarh",
    "TS":"Telangana","JH":"Jharkhand","UK":"Uttarakhand"}

def extract_num_from_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the frame
    plates = cascade.detectMultiScale(gray, 1.1, 4)

    # Process each detected license plate
    for (x, y, w, h) in plates:
        wT, hT, cT = frame.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = frame[y + a:y + h - a, x + b:x + w - b, :]

        # Enhance the license plate for OCR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        # Perform OCR on the license plate
        plate_text = pytesseract.image_to_string(plate)
        plate_text = ''.join(e for e in plate_text if e.isalnum())
        
        # Extract the state code
        state_code = plate_text[:2]
        
        # Get the state name from the dictionary
        state_name = states.get(state_code, "Unknown")

        # Draw the bounding box and recognized text on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(frame, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(frame, f"{state_name}: {plate_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

# Create a VideoCapture object to read from a video file
cap = cv2.VideoCapture("test1.mp4")

# Get the video frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output_video6.mp4", fourcc, 20.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = extract_num_from_frame(frame)
    
    # Display the processed frame
    cv2.imshow("License Plate Detection", processed_frame)

    # Write the frame to the output video
    out.write(processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
