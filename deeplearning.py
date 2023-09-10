import numpy as np
import cv2
import pytesseract
from PIL import Image

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO Model
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
net = cv2.dnn.readNetFromONNX('./static/models/best2.onnx')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get Detections
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    detections = detections.reshape((-1, 6))
    return input_image, detections

# Non-Maximum Suppression
def non_maximum_suppression(input_image, detections):
    boxes = []
    confidences = []

    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])

                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    index = np.array(index)
    index = index.flatten()

    return boxes_np, confidences_np, index

# Extract Text using OCR
def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Use Tesseract OCR to extract text
    text = pytesseract.image_to_string(Image.fromarray(gray))

    return text.strip()

# Drawing Bounding Boxes and Text
def draw_boxes(image, boxes_np, confidences_np, index):
    text_list = []
    for ind in index:
        x, y, w, h = boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf * 100)
        license_text = extract_text(image, boxes_np[ind])
        licence = license_text

        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 0, 255), -1)
        cv2.rectangle(image, (x, y+h), (x+w, y+h+30), (0, 0, 0), -1)

        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(image, licence, (x, y+h+27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        text_list.append(license_text)

    return image, text_list

# YOLO Predictions
def yolo_predictions(img, net):
    input_image, detections = get_detections(img, net)
    boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)
    result_img, text = draw_boxes(img, boxes_np, confidences_np, index)
    return result_img, text

# Video Object Detection
# Video Object Detection
def video_object_detection(input_path, output_path):
    lst = []
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Unable to read video')
            break

        results, text_list = yolo_predictions(frame, net)
        lst.append(text_list)
        out.write(results)

    # Extracting the best output with the maximum confidence level
    a = list()
    large = 0
    best = 0
    for i in lst:
        if len(i) > 0:
            a.append(i)

    for j in a:
        for k in j:
            for m in k:
                try:
                    confidence_str = m[1]
                    confidence = float(confidence_str)
                    if confidence > large:
                        large = confidence
                        best = m[0]
                except IndexError:
                    print("IndexError: string index out of range in:", m)

    print("Best Recognized Text:", best)

    cap.release()
    out.release()

def extract_text(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Apply thresholding or other preprocessing as needed
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Use Tesseract OCR to extract text with custom configuration
    custom_config = r'--oem 3 --psm 7'
    text = pytesseract.image_to_string(thresholded, config=custom_config)

    return text.strip()


# Example usage for video object detection
if __name__ == "__main__":
    input_video_path = 'Traffic.mp4'
    output_video_path = 'output_video6.mp4'
    recognized_text = video_object_detection(input_video_path, output_video_path)
