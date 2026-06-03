from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .classical import detect_edges, preprocess_image
from .config import CASE_VIDEO_PATH, RESULTS_DIR, ROOT_DIR
from .output_artifacts import (
    save_haar_vs_yolo_comparison,
    save_preprocessing_outputs,
    save_roi_mask_blend_panel,
    save_yolo_face_panel,
)
from .utils import ensure_dir, save_json, write_text
from .video_privacy import process_video
from .visualization import build_panel
from .yolo_face_onnx import YOLOFaceOnnxDetector, draw_yolo_face_detections, ensure_yolo_face_model


DEFAULT_OUTPUT_VIDEO = RESULTS_DIR / "video" / "privacy_video.mp4"


def parse_args() -> argparse.Namespace:
    """Lê os parâmetros de execução do pipeline."""

    parser = argparse.ArgumentParser(description="Anonimização facial em vídeo com YOLO-Face, ROI e Haar Cascade")
    parser.add_argument("--input", default=str(CASE_VIDEO_PATH), help="Caminho do vídeo de entrada")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_VIDEO), help="Caminho do vídeo anonimizado")
    parser.add_argument("--sample-second", type=float, default=None, help="Segundo usado como frame-chave dos painéis")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita frames processados para testes rápidos")
    parser.add_argument("--detection-stride", type=int, default=1, help="Executa detecção facial a cada N frames")
    parser.add_argument("--skip-yolo-face", action="store_true", help="Executa sem YOLO-Face")
    parser.add_argument("--skip-roi", action="store_true", help="Executa sem ROI tracking")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="Confiança mínima do YOLO-Face")
    return parser.parse_args()


def build_report(summary: dict, yolo_count: int | None) -> str:
    """Monta o relatório resumido da execução."""

    detector = summary.get("detector_used", "desconhecido")
    roi_status = "com ROI" if "roi" in detector else "sem ROI"
    yolo_line = (
        f"- YOLO-Face encontrou **{yolo_count}** rosto(s) no frame-chave."
        if yolo_count is not None
        else "- YOLO-Face foi pulado nesta execução."
    )

    return "\n".join(
        [
            "# Relatório — Video Face Privacy Blur",
            "",
            "## Case principal",
            "Detecção de rostos em vídeo e aplicação de desfoque localizado para privacidade.",
            "",
            "## Vídeo processado",
            f"- Entrada: `{summary['input_video']}`",
            f"- Saída: `{summary['output_video']}`",
            f"- Resolução: {summary['width']}x{summary['height']}",
            f"- FPS: {summary['fps']:.2f}",
            f"- Frames processados: {summary['processed_frames']}",
            "",
            "## Requisitos demonstrados",
            f"- Detecção facial em **{summary['frames_with_faces']}** frame(s).",
            f"- Média de **{summary['average_faces_per_frame']:.2f}** região(ões) faciais borradas por frame.",
            "- ROI, máscara e blend aplicados às regiões faciais.",
            "- Otsu como threshold principal e Adaptive Threshold como comparação.",
            "- Sobel como borda principal e Canny como comparação.",
            f"- Detector: **{detector}** ({roi_status}).",
            yolo_line,
            "",
            "## Observação ética",
            "O projeto não reconhece identidade; ele apenas reduz informação visual identificável.",
        ]
    )


def load_yolo_detector(args: argparse.Namespace) -> YOLOFaceOnnxDetector | None:
    """Carrega o detector YOLO-Face quando a execução permite seu uso."""

    if args.skip_yolo_face:
        return None

    try:
        model_path = ensure_yolo_face_model(skip_downloads=False)
        detector = YOLOFaceOnnxDetector(model_path, confidence_threshold=args.yolo_conf)
        print(f"YOLO-Face carregado: {model_path}")
        return detector
    except Exception as error:
        print(f"YOLO-Face indisponível: {error}")
        return None


def display_path(path: Path) -> str:
    """Retorna caminho relativo ao projeto quando possível."""

    try:
        return str(path.resolve().relative_to(ROOT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def main() -> None:
    """Executa o pipeline completo e salva todos os artefatos esperados."""

    args = parse_args()
    results_dir = ensure_dir(RESULTS_DIR)
    ensure_dir(results_dir / "video")
    frames_dir = ensure_dir(results_dir / "frames")
    panels_dir = ensure_dir(results_dir / "panels")
    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    yolo_detector = load_yolo_detector(args)

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
    edges = detect_edges(sample.original_frame, preprocess.gray)
    yolo_count = save_yolo_face_panel(panels_dir, sample.original_frame, yolo_detector)
    yolo_panel_frame = sample.original_frame.copy()
    if yolo_detector is not None:
        yolo_panel_frame = draw_yolo_face_detections(sample.original_frame, yolo_detector.detect(sample.original_frame))

    build_panel(
        [
            ("1. Frame original", sample.original_frame),
            ("2. Rostos detectados", sample.face_overlay),
            ("3. Privacidade final", sample.anonymized_frame),
            ("4. Threshold Otsu", preprocess.primary_threshold),
            ("5. Bordas Sobel", edges.primary_edges),
            ("6. YOLO-Face", yolo_panel_frame),
        ],
        panels_dir / "final_panel.png",
        cell_width=480,
        cell_height=270,
        columns=3,
    )

    cv2.imwrite(str(frames_dir / "final_result.png"), sample.anonymized_frame)
    cv2.imwrite(str(frames_dir / "face_mask.png"), sample.face_mask)
    cv2.imwrite(str(frames_dir / "faces_detected_frame.png"), sample.face_overlay)

    save_preprocessing_outputs(results_dir, sample.original_frame, preprocess, edges)
    save_roi_mask_blend_panel(panels_dir, sample.original_frame, sample.face_boxes, sample.face_mask, sample.anonymized_frame)
    save_haar_vs_yolo_comparison(panels_dir, sample.original_frame, yolo_detector)

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
        "case": "Detectar rostos em vídeo e aplicar desfoque para privacidade",
        "input_video": display_path(input_path),
        "output_video": display_path(output_path),
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
            display_path(output_path),
            "results/frames/final_result.png",
            "results/frames/face_mask.png",
            "results/frames/faces_detected_frame.png",
            "results/panels/final_panel.png",
            "results/panels/roi_mask_blend_2x2.png",
            "results/panels/yolo_face_panel.png",
            "results/panels/haar_vs_yolo_face_comparison.png",
            "results/preprocessing/01_original_frame.png",
            "results/preprocessing/02_grayscale.png",
            "results/preprocessing/03_gaussian_blur.png",
            "results/preprocessing/04_threshold_otsu.png",
            "results/preprocessing/05_threshold_adaptive_comparison.png",
            "results/preprocessing/06_sobel_magnitude.png",
            "results/preprocessing/07_canny_comparison.png",
            "results/preprocessing/preprocessing_panel.png",
            "results/report.md",
            "results/summary.json",
        ],
    }

    write_text(results_dir / "report.md", build_report(summary_data, yolo_count))
    save_json(summary_data, results_dir / "summary.json")

    print("\n" + "=" * 50)
    print("Projeto executado com sucesso!")
    print(f"Detector: {'+'.join(detector_used)}")
    print(f"Frames processados: {video_summary.processed_frames}")
    print(f"Frames com rosto: {video_summary.frames_with_faces}")
    print(f"Vídeo final: {output_path}")
    print(f"Painel final: {panels_dir / 'final_panel.png'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
