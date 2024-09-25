import numpy as np
import cv2

# definem os limites inferior e superior das cores no espaço de cor HSV
# HSV = HSV (HSV = O HSV, possui H (hue) que consiste na cor mesmo, o S (saturation) é a saturação de cor, muito relacionado a intensidade e o V (value) é o brilho da cor. Com as variações dessas características chegamos a diferentes níveis de cores e com intensidades e brilhos distintos.)
# Utilizamos HSV porque é um jeito mais intuitivo de descrever a cor

lower = {
  'red': (0,100,100),
  'black': (0, 0, 0)
}

upper = {
  'red': (180, 255, 255),
  'black': (180, 255, 45)
}

# definir cores padrão para o círculo à volta do objeto 
# BGR
colors = {
  'red': (0, 0, 255),
  'black': (0, 0, 0)
}

# Tamanho do video
video_width = 640
video_height = 360

# Inicialização da variavel do video
video_read = cv2.VideoCapture('video_redblack.mp4')

# Inicialização da variavel aonde vai salvar o video com a identificação das cores dos objetos
video_write = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (video_width, video_height))

while True:
    (sucesso, frame) = video_read.read()
    if not sucesso:
        break
    
    # Ajustar o tamanho do frame
    frame = cv2.resize(frame, (video_width, video_height))
    
	# Converter as cores do frame para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
	# Percore o array de cores para poder identificar na imagem
    for key, value in upper.items():
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        # Suavização pela Gaussiana para gerar menos borrão na imagem e reduzir o ruido
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
		# morfológica para remover mais ruido
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # encontrar os contornos da mascara
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        center = None
        if len(cnts) > 0:
            # encontra o maior contorno da mascara e utiliza para calcular o círculo
            threshold = 0.5
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
			
			# Centro da mascara
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if radius > threshold:
                # desenha o contorno do objeto e coloca o titulo da cor do objeto
                cv2.circle(frame, (int(x), int(y)),int(radius), colors[key], 2)
                cv2.putText(frame, key + " object", (int(x-radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[key], 2)

	# Salvar o frame em outro video
    video_write.write(frame) 
    # Exibir o frame
    cv2.imshow('video', frame)
    
    # se apertar Q para o loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video_read.release()
video_write.release()
cv2.destroyAllWindows()