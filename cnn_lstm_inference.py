import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from collections import deque
from pytorchcv.model_provider import get_model
import mediapipe as mp

class MobileNet_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        full_model = get_model('mobilenet_w1', pretrained=True)
        layers = list(full_model.features.children())
        if isinstance(layers[-1], (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            layers = layers[:-1]
        self.backbone = nn.Sequential(*layers)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=64, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64,128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.5),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64,1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, T, 3, 145,145]
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)             
        f = self.backbone(x)               
        f = self.gap(f).view(B, T, 1024)     
        out, _ = self.lstm(f)                
        out = out[:, -1, :]                  
        return self.classifier(out)    
    
    
         
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNet_LSTM().to(device)
model.load_state_dict(torch.load('mobileNet_finetuned_model.pth', map_location=device))
model.eval()


mp_fd = mp.solutions.face_detection
face_detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((145,145)),
    T.ToTensor(),
])


SEQ_LEN = 5               
THRESHOLD_SEQ = 3         

buffer = deque(maxlen=SEQ_LEN)
drowsy_seq_count = 0



cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)

    face = None
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(bbox.xmin * w);  y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w);  y2 = y1 + int(bbox.height * h)
        face = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    status = "ALERT"
    color = (0,255,0)

    if face is not None and face.size != 0:
        img_t = transform(face)
        buffer.append(img_t)

        if len(buffer) == SEQ_LEN:
            seq = torch.stack(list(buffer), dim=0).unsqueeze(0).to(device)  # [1,SEQ_LEN,3,145,145]
            with torch.no_grad():
                prob = model(seq).item()

            is_drowsy = prob > 0.5

            if is_drowsy:
                drowsy_seq_count += 1
            else:
                drowsy_seq_count = 0

            if drowsy_seq_count >= THRESHOLD_SEQ:
                status = "DROWSY"
                color = (0,0,255)

    if results.detections and face is not None:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{status} ({drowsy_seq_count})",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
