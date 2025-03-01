from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace

# Carregar o modelo YOLOv8
model = YOLO("yolov8n.pt")

# Iniciar a captura de vídeo (0 para webcam ou um caminho de vídeo)
video_path = 0
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Fazer a detecção de objetos no frame
    results = model(frame)

    # Percorrer as detecções
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
            conf = box.conf[0].item()  # Confiança da detecção
            cls = int(box.cls[0])  # Classe detectada

            # Verificar se a classe é "pessoa" (coco 0)
            if cls == 0:
                # Recortar a região do rosto (assumindo que o rosto está dentro da bbox)
                face = frame[y1:y2, x1:x2]

                # Evita erros caso a detecção seja inválida
                if face.shape[0] > 0 and face.shape[1] > 0:
                    try:
                        # Realizar análise de gênero
                        analysis = DeepFace.analyze(face, actions=["gender"], enforce_detection=False)

                        # Verifica se o gênero identificado é feminino
                        gender = analysis[0]["dominant_gender"]
                        if gender == "Woman":
                            label = f"Mulher ({conf:.2f})"
                            color = (0, 0, 255)  # Vermelho

                            # Desenha a bounding box e a label no frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    except Exception as e:
                        print("Erro na análise de gênero:", e)

    # Mostrar o frame
    cv2.imshow("YOLOv8 - Detecção de Mulheres", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()