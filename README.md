# Video Face Privacy Blur

Projeto final de Processamento Digital de Imagens focado em um caso realista e de escopo pequeno: detectar rostos em um vídeo curto e aplicar desfoque localizado para privacidade.

O pipeline usa uma entrevista de aproximadamente 10,73 segundos como base, processa o vídeo frame a frame, detecta as regiões faciais e gera imagens de apoio para explicar pré-processamento, ROI, máscara, blend, YOLO-Face e comparação com Haar Cascade.

## Objetivo Do Projeto

O objetivo principal é detectar rostos e aplicar desfoque para privacidade. Além disso, o projeto também demonstra:

- identificação de pessoas/rostos em ambiente externo ou de entrevista;
- aplicação de ROI, máscara e blend/paste sobre regiões específicas da imagem;
- uso de técnicas clássicas de Processamento Digital de Imagens;
- comparação entre Haar Cascade e YOLO-Face;
- geração de imagens intermediárias para apresentação técnica.

## Estrutura Do Repositório

```text
data/
  corte_video_base/
    corte_video_base.mp4
models/
  haarcascades/
    haarcascade_*.xml
  yolov8n-face-lindevs.onnx
presentation/
  Interview Face Privacy Presentation.pdf
results/
  frames/
  panels/
  preprocessing/
  video/
src/
  classical.py
  config.py
  face_privacy.py
  main.py
  output_artifacts.py
  roi_tracker.py
  utils.py
  video_privacy.py
  visualization.py
  yolo_face_onnx.py
requirements.txt
run_project.bat
```

## Pré-Requisitos

- Python 3.10 ou superior.
- `pip` atualizado.
- Vídeo base em `data/corte_video_base/corte_video_base.mp4`.
- Modelo YOLO-Face em `models/yolov8n-face-lindevs.onnx`.

Caso o modelo ONNX não esteja presente, o código tenta baixá-lo automaticamente a partir da URL configurada em `src/config.py`.

Se o download automático falhar, faça a instalação manual:

1. Acesse a página de releases do repositório: <https://github.com/Lima-Developer/pid_final_project/releases/latest>.
2. Abra o dropdown **Assets** da última release disponível.
3. Baixe o arquivo `yolov8n-face-lindevs.onnx`.
4. Coloque o arquivo baixado dentro da pasta `models/`.
5. Confira se o caminho final ficou `models/yolov8n-face-lindevs.onnx`.

## Como Rodar Do Zero

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd pid_final_project
```

### 2. Criar e ativar o ambiente virtual

No Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Rodar o projeto completo

```bash
python -m src.main
```

No Windows, também é possível usar:

```powershell
.\run_project.bat
```

### 5. Rodar um teste rápido

Este comando processa apenas uma parte do vídeo, útil para conferir se o ambiente está funcionando:

```bash
python -m src.main --max-frames 60 --output results/video/validation_privacy_video.mp4
```

## Opções Úteis

- `--input`: define o caminho do vídeo de entrada.
- `--output`: define o caminho do vídeo anonimizado.
- `--max-frames`: limita a quantidade de frames processados.
- `--sample-second`: define o segundo usado para gerar as imagens explicativas.
- `--detection-stride`: executa detecção facial a cada N frames.
- `--skip-yolo-face`: roda sem YOLO-Face, usando apenas o fluxo clássico/fallback.
- `--skip-roi`: roda sem rastreamento por ROI.
- `--yolo-conf`: ajusta a confiança mínima do YOLO-Face.

## Pipeline Atual

1. O vídeo é aberto com OpenCV e lido frame a frame.
2. Cada frame passa pelo detector YOLO-Face carregado via OpenCV DNN.
3. As caixas detectadas são expandidas em escala 1,20 para cobrir melhor a região facial.
4. O ROI tracker mantém uma região de busca coerente para cada pessoa ao longo do vídeo.
5. Quando o YOLO-Face falha em algum frame, o Haar Cascade atua como fallback dentro da ROI ou no frame completo.
6. As caixas faciais são convertidas em uma máscara binária.
7. A máscara recebe suavização nas bordas para evitar um recorte artificial.
8. O rosto é anonimizado com blur forte e leve pixelização.
9. O resultado final é composto por blend entre frame original, máscara e região borrada.
10. Um frame-chave também é usado para gerar imagens didáticas de pré-processamento e comparação.

## Pré-Processamento Clássico

O pré-processamento clássico não é o responsável direto pelo blur final do vídeo. Ele é usado para demonstrar os conceitos pedidos no projeto e comparar leituras tradicionais de imagem.

- Escala de cinza: reduz o frame para uma matriz de intensidade luminosa.
- Gaussian Blur: suaviza ruídos locais antes de threshold e bordas.
- Otsu Threshold: escolhido como threshold principal por separar melhor regiões claras e escuras no frame-chave.
- Adaptive Threshold: mantido como comparação, pois responde melhor a variações locais de iluminação.
- Sobel: escolhido como método principal de bordas por preservar contornos amplos e estruturais.
- Canny: mantido como comparação, pois gera bordas mais finas e seletivas.

## Outputs Esperados

Após rodar `python -m src.main`, os principais arquivos esperados são:

### Vídeo

- `results/video/privacy_video.mp4`: vídeo final com desfoque facial.

### Frames

- `results/frames/final_result.png`: frame-chave final anonimizado.
- `results/frames/face_mask.png`: máscara usada para delimitar a região facial.
- `results/frames/faces_detected_frame.png`: frame com as caixas de detecção desenhadas.

### Painéis

- `results/panels/final_panel.png`: visão geral do pipeline em um único painel.
- `results/panels/roi_mask_blend_2x2.png`: ROI, máscara, máscara sobreposta e blend final.
- `results/panels/yolo_face_panel.png`: comparação entre frame original e detecção YOLO-Face.
- `results/panels/haar_vs_yolo_face_comparison.png`: comparação direta entre Haar Cascade e YOLO-Face.

### Pré-Processamento

- `results/preprocessing/01_original_frame.png`: frame original usado como base.
- `results/preprocessing/02_grayscale.png`: conversão para escala de cinza.
- `results/preprocessing/03_gaussian_blur.png`: suavização por Gaussian Blur.
- `results/preprocessing/04_threshold_otsu.png`: threshold por Otsu.
- `results/preprocessing/05_threshold_adaptive_comparison.png`: threshold adaptativo para comparação.
- `results/preprocessing/06_sobel_magnitude.png`: magnitude de bordas por Sobel.
- `results/preprocessing/07_canny_comparison.png`: bordas por Canny para comparação.
- `results/preprocessing/preprocessing_panel.png`: painel com os principais passos clássicos.

### Relatórios

- `results/report.md`: relatório textual resumido da execução.
- `results/summary.json`: metadados do processamento e lista de outputs.

## Organização Dos Outputs

O repositório foi limpo para evitar imagens duplicadas sem função clara. Os resultados finais ficam separados por finalidade:

- `results/video/`: vídeos gerados.
- `results/frames/`: imagens individuais do frame-chave.
- `results/panels/`: composições visuais usadas nos slides.
- `results/preprocessing/`: etapas clássicas de processamento de imagem.

## Limitações Conhecidas

- O processamento é feito em CPU usando OpenCV DNN pelo pacote `opencv-python-headless`.
- O pacote padrão do OpenCV instalado via `pip` não usa CUDA neste projeto; usar GPU exigiria outra instalação/build do OpenCV com backend CUDA.
- Em um Ryzen 7 5700X, com 8 núcleos, 16 threads e clock base de 3,4 GHz, o vídeo de 10,73 segundos com 644 frames leva aproximadamente 8 minutos para ser processado por completo.
- O vídeo final gerado pelo OpenCV não preserva a trilha de áudio original.
- O projeto detecta e anonimiza faces; ele não faz reconhecimento de identidade.
- Rosto em perfil, oclusões, movimento e mudanças bruscas de iluminação podem causar falhas pontuais de detecção.
- O Haar Cascade é usado como fallback e comparação didática, mas tende a ser menos robusto que o YOLO-Face.

## Observação Ética

O projeto reduz informação visual identificável em vídeo, mas não deve ser tratado como garantia absoluta de anonimização. Em aplicações reais, seria necessário validar o resultado com critérios de privacidade, qualidade e segurança mais rigorosos.
