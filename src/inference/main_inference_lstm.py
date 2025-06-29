import socket
import cv2
import mediapipe as mp
import math
import numpy as np
import os
import time
import torch

# ──────────────────────────────────────────────────────────────
# JetBot 통신 설정
# ──────────────────────────────────────────────────────────────
JETBOT_IP = "192.168.0.15"
UDP_PORT = 5555
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
previous_flag = -1

# ──────────────────────────────────────────────────────────────
# 눈, 입, 동공 관련 지표 계산 함수
# ──────────────────────────────────────────────────────────────
def calc_distance(p1, p2):
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5

def eye_aspect_ratio(landmarks, eye):
    N1 = calc_distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = calc_distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = calc_distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = calc_distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def eye_feature(landmarks):
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2

def mouth_feature(landmarks):
    N1 = calc_distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = calc_distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = calc_distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = calc_distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)

def pupil_circularity(landmarks, eye):
    perimeter = (calc_distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) +
                 calc_distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) +
                 calc_distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) +
                 calc_distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) +
                 calc_distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) +
                 calc_distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) +
                 calc_distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) +
                 calc_distance(landmarks[eye[1][1]], landmarks[eye[0][0]]))
    area = math.pi * ((calc_distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)

def pupil_feature(landmarks):
    return (pupil_circularity(landmarks, left_eye) + pupil_circularity(landmarks, right_eye)) / 2

# ──────────────────────────────────────────────────────────────
# 얼굴 특징 추출
# ──────────────────────────────────────────────────────────────
def run_face_detection(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)
        landmarks[:, 0] *= image.shape[1]
        landmarks[:, 1] *= image.shape[0]

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)

        ear = eye_feature(landmarks)
        mar = mouth_feature(landmarks)
        puc = pupil_feature(landmarks)
        moe = mar / ear
    else:
        ear, mar, puc, moe = (-1000, -1000, -1000, -1000)

    return ear, mar, puc, moe, image

# ──────────────────────────────────────────────────────────────
# 중립 얼굴 기준값 측정
# ──────────────────────────────────────────────────────────────
def calibrate_reference(frames=25):
    ears, mars, pucs, moes = [], [], [], []
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        ear, mar, puc, moe, image = run_face_detection(image)
        if ear != -1000:
            ears.append(ear)
            mars.append(mar)
            pucs.append(puc)
            moes.append(moe)

        cv2.putText(image, "Calibrating...", (int(0.02 * image.shape[1]), int(0.14 * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.imshow('FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q") or len(ears) >= frames:
            break

    cv2.destroyAllWindows()
    cap.release()
    ears, mars, pucs, moes = map(np.array, (ears, mars, pucs, moes))
    return [ears.mean(), ears.std()], [mars.mean(), mars.std()], [pucs.mean(), pucs.std()], [moes.mean(), moes.std()]

# ──────────────────────────────────────────────────────────────
# LSTM 졸음 분류
# ──────────────────────────────────────────────────────────────
def classify_state(input_data):
    segments = [input_data[:5], input_data[3:8], input_data[6:11], input_data[9:14], input_data[12:17], input_data[15:]]
    model_input = torch.FloatTensor(np.array(segments))
    preds = torch.sigmoid(model(model_input)).gt(0.75).int().data.numpy()
    return int(preds.sum() >= 5)

# ──────────────────────────────────────────────────────────────
# 실시간 상태 추론
# ──────────────────────────────────────────────────────────────
def real_time_inference(ears_ref, mars_ref, pucs_ref, moes_ref):
    global previous_flag
    ear_avg = mar_avg = puc_avg = moe_avg = 0
    decay = 0.9
    input_buffer = []
    frame_counter = 0
    drowsy_start = None
    danger_mode = False
    danger_limit = 5.0

    cap = cv2.VideoCapture(0)
    label = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        ear, mar, puc, moe, image = run_face_detection(image)
        if ear != -1000:
            ear = (ear - ears_ref[0]) / ears_ref[1]
            mar = (mar - mars_ref[0]) / mars_ref[1]
            puc = (puc - pucs_ref[0]) / pucs_ref[1]
            moe = (moe - moes_ref[0]) / moes_ref[1]

            ear_avg = ear_avg * decay + (1 - decay) * ear
            mar_avg = mar_avg * decay + (1 - decay) * mar
            puc_avg = puc_avg * decay + (1 - decay) * puc
            moe_avg = moe_avg * decay + (1 - decay) * moe
        else:
            ear_avg = mar_avg = puc_avg = moe_avg = -1000

        if len(input_buffer) == 20:
            input_buffer.pop(0)
        input_buffer.append([ear_avg, mar_avg, puc_avg, moe_avg])

        frame_counter += 1
        if frame_counter >= 15 and len(input_buffer) == 20:
            frame_counter = 0
            label = classify_state(input_buffer)
            if label == 1:
                if drowsy_start is None:
                    drowsy_start = time.time()
                elif time.time() - drowsy_start >= danger_limit:
                    danger_mode = True
            else:
                drowsy_start = None
                danger_mode = False

        flag = 2 if danger_mode else label
        if flag != previous_flag:
            udp_socket.sendto(str(flag).encode(), (JETBOT_IP, UDP_PORT))
            previous_flag = flag

        cv2.putText(image, f"EAR: {ear_avg:.2f} MAR: {mar_avg:.2f} PUC: {puc_avg:.2f} MOE: {moe_avg:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if danger_mode:
            cv2.putText(image, "DANGER", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        elif label is not None:
            status_text = "Drowsy" if label else "Alert"
            status_color = (0, 255, 255) if label else (0, 255, 0)
            cv2.putText(image, status_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 2)

        cv2.imshow('FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ──────────────────────────────────────────────────────────────
# 사전 정의 인덱스
# ──────────────────────────────────────────────────────────────
right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# LSTM 로드
model_path = os.path.join('models', 'lstm.pth')
model = torch.jit.load(model_path)
model.eval()

# ──────────────────────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────────────────────
print("[INFO] Neutral Face Calibration 시작")
time.sleep(1)
ears_norm, mars_norm, pucs_norm, moes_norm = calibrate_reference()

print("[INFO] 실시간 졸음 감지 시작")
time.sleep(1)
real_time_inference(ears_norm, mars_norm, pucs_norm, moes_norm)

face_mesh.close()
