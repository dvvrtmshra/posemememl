import cv2
import mediapipe as mp
import math

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face.FaceMesh(max_num_faces=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Load meme images
thinking_img = cv2.imread("thinking.png")
pointing_img = cv2.imread("pointing.png")

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)

    show_thinking = False
    show_pointing = False

    # ----- FACE DETECTION (chin) -----
    if face_result.multi_face_landmarks:
        face = face_result.multi_face_landmarks[0]
        chin = face.landmark[152]   # chin landmark
        chin_point = (int(chin.x * w), int(chin.y * h))

    # ----- HAND DETECTION -----
    if hand_result.multi_hand_landmarks:
        hand = hand_result.multi_hand_landmarks[0]

        index_tip = hand.landmark[8]
        index_mcp = hand.landmark[5]
        middle_tip = hand.landmark[12]

        index_tip_point = (int(index_tip.x * w), int(index_tip.y * h))
        index_mcp_point = (int(index_mcp.x * w), int(index_mcp.y * h))
        middle_tip_point = (int(middle_tip.x * w), int(middle_tip.y * h))

        # 🤔 THINKING POSE
        if face_result.multi_face_landmarks:
            d = distance(index_tip_point, chin_point)
            if d < 40:
                show_thinking = True

        # ☝️ POINTING POSE
        if index_tip.y < index_mcp.y and index_tip.y < middle_tip.y:
            show_pointing = True

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ----- DISPLAY MEMES -----
    if show_thinking:
        frame[10:210, 10:210] = cv2.resize(thinking_img, (200, 200))

    if show_pointing:
        frame[10:210, w-210:w-10] = cv2.resize(pointing_img, (200, 200))

    # ✅ SHOW FRAME INSIDE LOOP
    cv2.imshow("Pose Meme ML", frame)

    # ✅ KEY CHECK INSIDE LOOP
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break



cap.release()
cv2.destroyAllWindows()
