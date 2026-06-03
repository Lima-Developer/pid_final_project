import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Diretório do projeto
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Caminhos dos arquivos
MODEL_PATH = MODELS_DIR / "yolov8n-face-lindevs.pt"
ONNX_PATH = MODELS_DIR / "yolov8n-face-lindevs.onnx"

# URL de fallback (só usa se o .pt não existir)
MODEL_URL = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face-lindevs.pt"


def download_with_progress(url: str, destination: Path):
    """Download com barra de progresso simples."""
    print(f"Baixando {url}...")
    print(f"Destino: {destination}")
    
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        print(f"\r  Progresso: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_hook)
        print()
        return True
    except Exception as e:
        print(f"\n❌ Erro no download: {e}")
        return False


def find_exported_onnx() -> Path | None:
    """Procura o arquivo ONNX exportado pelo ultralytics."""
    # O ultralytics salva no diretório atual ou em runs/
    candidates = [
        Path("yolov8n-face-lindevs.onnx"),
        Path("yolov8n-face-lindevs.onnx"),
    ]
    
    # Procura recursivamente
    for pattern in ["*.onnx", "**/*.onnx"]:
        for candidate in Path(".").glob(pattern):
            if "yolov8n-face" in candidate.name.lower():
                return candidate
    
    return None


def main():
    print("=" * 60)
    print("Exportação YOLOv8n-Face → ONNX (OpenCV-compatible)")
    print("=" * 60)

    # 1. Verifica se o .pt existe
    if not MODEL_PATH.exists():
        print(f"\n❌ Modelo .pt não encontrado em: {MODEL_PATH}")
        print("Tentando download automático...")
        
        success = download_with_progress(MODEL_URL, MODEL_PATH)
        if not success:
            print("\nFalha no download.")
            print("\nBaixe manualmente:")
            print("1. Acesse: https://github.com/lindevs/yolov8-face/releases/tag/v1.0.0")
            print("2. Baixe 'yolov8n-face-lindevs.pt'")
            print(f"3. Coloque em: {MODEL_PATH}")
            return
    else:
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"\n✅ Modelo .pt encontrado: {MODEL_PATH}")
        print(f"   Tamanho: {size_mb:.1f} MB")

    # 2. Verifica se ONNX já existe e está válido
    if ONNX_PATH.exists() and ONNX_PATH.stat().st_size > 5_000_000:
        print(f"\n✅ ONNX já existe: {ONNX_PATH}")
        print(f"   Tamanho: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        test_only = input("\nDeseja testar o ONNX existente? (s/n): ").lower().strip() == 's'
        if not test_only:
            print("Saindo. Use o ONNX existente.")
            return
    else:
        test_only = False

    # 3. Exporta para ONNX (se necessário)
    if not test_only:
        print("\n" + "=" * 60)
        print("Exportando para ONNX")
        print("=" * 60)
        print("Parâmetros: dynamic=False | opset=12 | simplify=True")
        print("Objetivo: Compatibilidade total com OpenCV DNN")
        print()
        
        print("Carregando modelo PyTorch...")
        model = YOLO(str(MODEL_PATH))
        
        print("Iniciando exportação ONNX...")
        try:
            model.export(
                format="onnx",
                dynamic=False,      # ⬅️ ESSENCIAL: shapes estáticos (evita erro Concat)
                opset=12,           # ⬅️ ESSENCIAL: compatível com OpenCV 4.x
                simplify=True,      # ⬅️ ESSENCIAL: remove ops problemáticas
                imgsz=640,
            )
        except Exception as e:
            print(f"\n❌ Erro na exportação: {e}")
            return
        
        # 4. Encontra e move o ONNX gerado
        print("\nProcurando arquivo ONNX gerado...")
        exported = find_exported_onnx()
        
        if exported is None:
            print("❌ Arquivo ONNX não encontrado após exportação")
            print("   Verifique o diretório atual por arquivos .onnx")
            return
        
        print(f"   Encontrado: {exported}")
        
        # Move para models/
        if exported.resolve() != ONNX_PATH.resolve():
            shutil.move(str(exported), str(ONNX_PATH))
            print(f"   Movido para: {ONNX_PATH}")
        
        print(f"\n✅ ONNX exportado: {ONNX_PATH}")
        print(f"   Tamanho: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

    # 5. Teste de carregamento no OpenCV DNN
    print("\n" + "=" * 60)
    print("Testando carregamento no OpenCV DNN")
    print("=" * 60)
    
    try:
        print(f"Carregando: {ONNX_PATH}")
        net = cv2.dnn.readNetFromONNX(str(ONNX_PATH))
        print("✅ Modelo carregado com sucesso no OpenCV DNN!")
        
        # Teste de inferência
        print("\nTestando inferência com imagem dummy...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(dummy, 1/255.0, (640, 640), swapRB=True, crop=False)
        
        net.setInput(blob)
        out = net.forward()
        
        print(f"   Shape da entrada: {blob.shape}")
        print(f"   Shape da saída: {out.shape}")
        print(f"   Tipo da saída: {out.dtype}")
        
        # Verifica formato da saída
        if out.ndim == 3:
            print(f"   Formato: (batch, features, anchors) = {out.shape}")
            print("   ✅ Formato compatível com YOLO pós-processamento!")
        else:
            print(f"   ⚠️ Formato inesperado: {out.ndim} dimensões")
        
        print("\n" + "=" * 60)
        print("🎉 TUDO PRONTO!")
        print("=" * 60)
        print(f"\nModelo ONNX salvo em: {ONNX_PATH}")
        print("\nAtualize seu config.py:")
        print(f'  YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-face-lindevs.onnx"')
        
    except Exception as e:
        print(f"\n❌ Falha ao carregar no OpenCV: {e}")
        print("\nPossíveis causas:")
        print("1. ONNX exportado com dynamic shapes → use dynamic=False")
        print("2. Opset muito alto → use opset=12")
        print("3. Operações não suportadas → use simplify=True")
        print("\nSolução alternativa: use ONNX Runtime em vez de OpenCV DNN")


if __name__ == "__main__":
    main()