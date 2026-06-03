# Relatório — Video Face Privacy Blur

## Case principal
Detecção de rostos em vídeo e aplicação de desfoque localizado para privacidade.

## Vídeo processado
- Entrada: `data/corte_video_base/corte_video_base.mp4`
- Saída: `results/video/privacy_video.mp4`
- Resolução: 1920x1080
- FPS: 60.00
- Frames processados: 644

## Requisitos demonstrados
- Detecção facial em **644** frame(s).
- Média de **1.88** região(ões) faciais borradas por frame.
- ROI, máscara e blend aplicados às regiões faciais.
- Otsu como threshold principal e Adaptive Threshold como comparação.
- Sobel como borda principal e Canny como comparação.
- Detector: **yolo-face+roi** (com ROI).
- YOLO-Face encontrou **1** rosto(s) no frame-chave.

## Observação ética
O projeto não reconhece identidade; ele apenas reduz informação visual identificável.