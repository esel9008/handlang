import threading
from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

cameras = {
    'webcam1': cv2.VideoCapture(0),
    'webcam2': cv2.VideoCapture(0),
}

def gen_frames(camera_id):
    cap = cameras.get(camera_id)
    if cap is None:
        return

    order_detected = False
    possible_detected = False
    food_detected = False

    def is_left_palm_up(hand_landmarks):
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        return wrist.y < middle_finger_tip.y

    def is_right_thumb_near_left_palm(right_hand_landmarks, left_hand_landmarks):
        wrist = left_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        middle_finger_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        left_palm_center = (
            (wrist.x + middle_finger_tip.x) / 2,
            (wrist.y + middle_finger_tip.y) / 2,
            (wrist.z + middle_finger_tip.z) / 2
        )
        right_thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        distance = ((right_thumb_tip.x - left_palm_center[0]) ** 2 +
                   (right_thumb_tip.y - left_palm_center[1]) ** 2 +
                   (right_thumb_tip.z - left_palm_center[2]) ** 2) ** 0.5
        return distance < 0.1

    def is_order_gesture(results):
        if len(results.multi_hand_landmarks) == 2:
            hand_landmarks1 = results.multi_hand_landmarks[0]
            hand_landmarks2 = results.multi_hand_landmarks[1]
            if hand_landmarks1.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks2.landmark[mp_hands.HandLandmark.WRIST].x:
                left_hand_landmarks = hand_landmarks1
                right_hand_landmarks = hand_landmarks2
            else:
                left_hand_landmarks = hand_landmarks2
                right_hand_landmarks = hand_landmarks1
            if is_left_palm_up(left_hand_landmarks) and is_right_thumb_near_left_palm(right_hand_landmarks, left_hand_landmarks):
                return True
        return False

    def is_possible_gesture(hand_landmarks):
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        hand_open_angle = abs(index_finger_tip.y - middle_finger_tip.y) > 0.02
        hand_angled = wrist.y > min(index_finger_tip.y, middle_finger_tip.y, ring_finger_tip.y, pinky_tip.y)
        palm_facing = wrist.z < index_finger_tip.z and wrist.z < middle_finger_tip.z

        return hand_open_angle and hand_angled and palm_facing

    def is_food_gesture(results):
        if len(results.multi_hand_landmarks) == 1:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            hand_open = (index_tip.y < wrist.y and middle_tip.y < wrist.y and
                         ring_tip.y < wrist.y and pinky_tip.y < wrist.y)
            palm_up = wrist.y < thumb_tip.y

            return hand_open and palm_up
        return False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(image_rgb)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if camera_id == 'webcam1':
                    if is_order_gesture(results_hands):
                        order_detected = True

                    if order_detected and is_possible_gesture(hand_landmarks):
                        possible_detected = True

                elif camera_id == 'webcam2':
                    if is_food_gesture(results_hands):
                        food_gesture_detected = True

            if camera_id == 'webcam1':
                if order_detected:
                    cv2.putText(image, 'ORDER', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                if possible_detected:
                    cv2.putText(image, 'POSSIBLE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                if order_detected and possible_detected:
                    cv2.putText(image, 'SUCCESS', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
            
            elif camera_id == 'webcam2':
                if food_gesture_detected:
                    cv2.putText(image, 'FOOD', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            print("Failed to encode image")
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(gen_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/order')
def order():
    return render_template('order.html')

@app.route('/food')
def food():
    return render_template('food.html')

def run_flask():
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
