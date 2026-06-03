# Video Face Privacy Blur

Projeto final de **Processamento Digital de Imagens** focado em:

> detectar rostos em um vídeo e aplicar desfoque para privacidade.

O vídeo é tratado como uma sequência de frames. Para cada frame, o sistema detecta regiões faciais com Haar Cascade, cria uma máscara, aplica blur/pixelização apenas nessas regiões e salva um novo vídeo anonimizado.

## Objetivo principal

Anonimizar rostos em um vídeo real curto, demonstrando uma aplicação prática de privacidade em mídia audiovisual.

## Requisitos demonstrados

- **Detecção de rostos e privacidade:** Haar Cascade + blur localizado.
- **Threshold & blur:** pré-processamento no frame-chave com Otsu e suavização.
- **Bordas e cantos:** Canny, Sobel e Harris no frame-chave.
- **ROI, máscara e blend/paste:** máscara facial e composição do blur nas regiões detectadas.
- **YOLO:** detecção da classe `person` no frame-chave.
- **Ambiente real:** vídeo de entrevista em boa qualidade dentro de `data/corte_video_base/`.

## Como executar

Com a `.venv` ativada:

```powershell
python -m src.main
```

Ou diretamente pelo Python da venv:

```powershell
.\.venv\Scripts\python.exe -m src.main
```

Teste rápido com poucos frames:

```powershell
python -m src.main --max-frames 120 --skip-yolo
```

Por padrão, a detecção facial roda em todo frame. Se quiser acelerar a execução aceitando menor precisão temporal, use:

```powershell
python -m src.main --detection-stride 5
```

## Entradas

- `data/corte_video_base/corte_video_base.mp4` — vídeo base do projeto.

## Saídas

- `results/privacy_video.mp4` — vídeo final com rostos anonimizados.
- `results/final_panel.png` — painel com frame original, detecção, privacidade, threshold, Canny e YOLO.
- `results/final_result.png` — frame-chave anonimizado.
- `results/face_mask.png` — máscara facial do frame-chave.
- `results/faces_detected_frame.png` — frame-chave com caixas de rosto.
- `results/canny_frame.png` — bordas Canny do frame-chave.
- `results/harris_frame.png` — cantos Harris do frame-chave.
- `results/report.md` — resumo técnico da execução.
- `results/summary.json` — métricas e metadados.

## Observação

O vídeo final gerado pelo OpenCV não preserva áudio. Isso é aceitável para o objetivo de PDI, pois o processamento visual é o foco do projeto.
