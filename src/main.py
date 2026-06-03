from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .classical import detect_edges_and_corners, preprocess_image
from .config import DATA_DIR, RESULTS_DIR
from .utils import ensure_dir, save_json, write_text
from .video_privacy import process_video
from .visualization import build_panel
from .yolo_face_onnx import YOLOFaceOnnxDetector, draw_yolo_face_detections, ensure_yolo_face_model


CASE_DIR = DATA_DIR / "corte_video_base"
CASE_VIDEO_PATH = CASE_DIR / "corte_video_base.mp4"
DEFAULT_OUTPUT_VIDEO = RESULTS_DIR / "privacy_video.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anonimização facial em vídeo com YOLO-Face e Haar")
    parser.add_argument("--input", default=str(CASE_VIDEO_PATH), help="Caminho do vídeo de entrada")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_VIDEO), help="Caminho do vídeo anonimizado")
    parser.add_argument("--sample-second", type=float, default=None, help="Segundo usado no painel final")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita frames processados para teste rápido")
    parser.add_argument("--detection-stride", type=int, default=1, help="Executa detecção facial a cada N frames")
    parser.add_argument("--skip-yolo-face", action="store_true", help="Não usa YOLO-Face (apenas Haar Cascade)")
    parser.add_argument("--skip-roi", action="store_true", help="Não usa ROI tracking")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="Threshold de confiança do YOLO-Face")
    return parser.parse_args()


def build_report(summary: dict, yolo_count: int | None) -> str:
    detector = summary.get("detector_used", "desconhecido")
    roi_status = "com ROI" if "roi" in detector else "sem ROI"
    
    yolo_line = (
        f"- YOLO-Face encontrou **{yolo_count}** rosto(s) no frame-chave."
        if yolo_count is not None
        else "- YOLO-Face foi pulado nesta execução."
    )
    
    return "\n".join(
        [
            "# Relatório — Video Face Privacy Blur (YOLO-Face + ROI)",
            "",
            "## Case principal",
            "Detecção facial híbrida: YOLO-Face (primário) + Haar Cascade em ROIs (fallback).",
            "",
            "## Vídeo processado",
            f"- Entrada: `{summary['input_video']}`",
            f"- Saída: `{summary['output_video']}`",
            f"- Resolução: {summary['width']}x{summary['height']}",
            f"- FPS: {summary['fps']:.2f}",
            f"- Frames processados: {summary['processed_frames']}",
            "",
            "## Requisitos demonstrados",
            f"- Detecção facial: rosto(s) em **{summary['frames_with_faces']}** frame(s).",
            f"- Privacidade: média de **{summary['average_faces_per_frame']:.2f}** faces borradas/frame.",
            "- Blur gaussiano extremo (kernel 299) + pixelação agressiva.",
            f"- Detector: **{detector}** ({roi_status}).",
            yolo_line,
            "",
            "## Observação ética",
            "O projeto não reconhece identidade. Apenas reduz informação identificável.",
        ]
    )


def main() -> None:
    args = parse_args()
    results_dir = ensure_dir(RESULTS_DIR)
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Inicializa YOLO-Face
    yolo_detector = None
    if not args.skip_yolo_face:
        try:
            model_path = ensure_yolo_face_model(skip_downloads=False)
            yolo_detector = YOLOFaceOnnxDetector(
                model_path,
                confidence_threshold=args.yolo_conf,
            )
            print(f"YOLO-Face carregado: {model_path}")
        except Exception as e:
            print(f"YOLO-Face indisponível: {e}")

    # Processa vídeo
    video_summary = process_video(
        input_path=input_path,
        output_path=output_path,
        yolo_detector=yolo_detector,
        use_roi=not args.skip_roi,
        sample_second=args.sample_second,
        max_frames=args.max_frames,
        detection_stride=args.detection_stride,
    )
    
    sample = video_summary.sample_result
    if sample is None:
        raise RuntimeError("Nenhum frame-chave foi gerado.")

    preprocess = preprocess_image(sample.original_frame)
    edges = detect_edges_and_corners(sample.original_frame, preprocess.gray)

    # YOLO-Face no frame de amostra
    yolo_drawn = sample.original_frame.copy()
    yolo_count: int | None = None
    if yolo_detector is not None:
        try:
            yolo_detections = yolo_detector.detect(sample.original_frame)
            yolo_count = len(yolo_detections)
            yolo_drawn = draw_yolo_face_detections(sample.original_frame, yolo_detections)
        except Exception:
            pass

    # ROI visualization
    roi_image = sample.original_frame.copy()
    if video_summary.roi_tracker is not None:
        roi_image = video_summary.roi_tracker.draw_rois(roi_image)

    panel_items = [
        ("1. Frame original", sample.original_frame),
        ("2. Rostos detectados", sample.face_overlay),
        ("3. Privacidade final", sample.anonymized_frame),
        ("4. Threshold Otsu", preprocess.threshold_otsu),
        ("5. Bordas Canny", edges.canny),
        ("6. YOLO-Face + ROIs", yolo_drawn),
    ]
    build_panel(panel_items, results_dir / "final_panel.png", cell_width=480, cell_height=270)

    cv2.imwrite(str(results_dir / "final_result.png"), sample.anonymized_frame)
    cv2.imwrite(str(results_dir / "face_mask.png"), sample.face_mask)
    cv2.imwrite(str(results_dir / "faces_detected_frame.png"), sample.face_overlay)
    cv2.imwrite(str(results_dir / "canny_frame.png"), edges.canny)

    average_faces = (
        video_summary.total_face_detections / video_summary.processed_frames
        if video_summary.processed_frames
        else 0.0
    )
    
    detector_used = []
    if yolo_detector:
        detector_used.append("yolo-face")
    if video_summary.roi_tracker:
        detector_used.append("roi")
    if not detector_used:
        detector_used.append("haar-only")
    
    summary_data = {
        "case": "Detectar rostos em vídeo com YOLO-Face + ROI + Haar e aplicar desfoque",
        "input_video": str(input_path),
        "output_video": str(output_path),
        "width": video_summary.metadata.width,
        "height": video_summary.metadata.height,
        "fps": video_summary.metadata.fps,
        "duration_seconds": video_summary.metadata.duration_seconds,
        "processed_frames": video_summary.processed_frames,
        "frames_with_faces": video_summary.frames_with_faces,
        "total_face_detections": video_summary.total_face_detections,
        "average_faces_per_frame": average_faces,
        "max_faces_in_frame": video_summary.max_faces_in_frame,
        "sample_frame_index": sample.frame_index,
        "sample_face_boxes": sample.face_boxes,
        "yolo_face_count_on_sample": yolo_count,
        "detector_used": "+".join(detector_used),
        "outputs": [
            "results/privacy_video.mp4",
            "results/final_panel.png",
            "results/final_result.png",
            "results/face_mask.png",
            "results/faces_detected_frame.png",
            "results/canny_frame.png",
            "results/report.md",
        ],
    }

    write_text(results_dir / "report.md", build_report(summary_data, yolo_count))
    save_json(summary_data, results_dir / "summary.json")

    print("\n" + "="*50)
    print("Projeto executado com sucesso!")
    print(f"Detector: {'+'.join(detector_used)}")
    print(f"Frames processados: {video_summary.processed_frames}")
    print(f"Frames com rosto: {video_summary.frames_with_faces}")
    print(f"Vídeo final: {output_path}")
    print(f"Painel final: {results_dir / 'final_panel.png'}")
    print("="*50)


if __name__ == "__main__":
    main()