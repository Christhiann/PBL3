

import os
import tempfile
import argparse
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image as PILImage
import pytesseract

# --------------------------
# Configurações / Thresholds
# --------------------------
THRESH_SS = 0.92
THRESH_BLOCK_RATIO = 3.0
THRESH_HIST_CHI2 = 0.25

# Ajuste caminho do tesseract se necessário
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --------------------------
# Funções utilitárias
# --------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_and_preprocess(path, size=(1024, 1024)):
    """Lê imagem, redimensiona e devolve BGR e gray."""
    try:
        pil_img = PILImage.open(path).convert('RGB')
        pil_img = pil_img.resize(size)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
    except Exception as e:
        raise ValueError(f"Falha ao ler {path}: {e}")

def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def edge_map(gray):
    return cv2.Canny(gray, 100, 200)

def histogram_chi2(img_a, img_b, bins=32):
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0,1], None, [bins, bins], [0,180,0,256]).flatten()
    hist_b = cv2.calcHist([hsv_b], [0,1], None, [bins, bins], [0,180,0,256]).flatten()
    hist_a /= (hist_a.sum() + 1e-9)
    hist_b /= (hist_b.sum() + 1e-9)
    eps = 1e-9
    chi2 = 0.5 * np.sum(((hist_a - hist_b)**2) / (hist_a + hist_b + eps))
    return float(chi2)

def block_variances(gray, block=64):
    h, w = gray.shape
    vars_list = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            blk = gray[y:y+block, x:x+block]
            if blk.size == 0:
                continue
            vars_list.append(float(np.var(blk)))
    return np.array(vars_list)

def compute_ssim_and_diff(img_a_gray, img_b_gray):
    score, diff = ssim(img_a_gray, img_b_gray, full=True)
    diff_n = (diff - diff.min()) / (diff.max() - diff.min() + 1e-9)
    return float(score), diff_n

# --------------------------
# Funções de OCR e SSIM local
# --------------------------
def ocr_text_and_boxes(path):
    pil = PILImage.open(path).convert("RGB")
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, lang='por+eng')
    text = " ".join([w for w in data['text'] if w.strip() != ""])
    boxes = []
    for i in range(len(data['level'])):
        w = data['text'][i].strip()
        if not w:
            continue
        x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({"text": w, "left": x, "top": y, "w": w_box, "h": h_box})
    return text, boxes

def compare_ocr_texts(path_a, path_b):
    txt_a, boxes_a = ocr_text_and_boxes(path_a)
    txt_b, boxes_b = ocr_text_and_boxes(path_b)
    sa = " ".join(txt_a.lower().split())
    sb = " ".join(txt_b.lower().split())
    set_a = set(sa.split())
    set_b = set(sb.split())
    added_words = set_b - set_a
    removed_words = set_a - set_b
    return {
        "text_a": sa,
        "text_b": sb,
        "added_words": list(added_words)[:20],
        "removed_words": list(removed_words)[:20],
        "boxes_a": boxes_a,
        "boxes_b": boxes_b
    }

def local_ssim_map(gray_a, gray_b, win_size=64, step=32):
    h, w = gray_a.shape
    score_map = np.zeros((h, w), dtype=float)
    counts = np.zeros((h, w), dtype=float)
    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            wa = gray_a[y:y+win_size, x:x+win_size]
            wb = gray_b[y:y+win_size, x:x+win_size]
            try:
                s, _ = ssim(wa, wb, full=True)
            except Exception:
                s = 1.0
            score_map[y:y+win_size, x:x+win_size] += s
            counts[y:y+win_size, x:x+win_size] += 1
    counts[counts == 0] = 1
    score_map /= counts
    return score_map

def diff_regions_from_ssim_map(score_map, threshold=0.95, min_area=200):
    mask = (score_map < threshold).astype('uint8') * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        boxes.append((x,y,w,h))
    return boxes, mask

# --------------------------
# Função principal de análise
# --------------------------
def analyze_image(path_img, ref_images, tmpdir):
    img, gray = read_and_preprocess(path_img)
    metrics = {}

    metrics['lap_var'] = laplacian_variance(gray)
    edges = edge_map(gray)
    edges_path = os.path.join(tmpdir, "edges.png")
    cv2.imwrite(edges_path, edges)

    vars_arr = block_variances(gray, block=64)
    metrics['block_mean'] = float(vars_arr.mean())
    metrics['block_median'] = float(np.median(vars_arr))
    metrics['block_max'] = float(vars_arr.max())
    metrics['block_std'] = float(vars_arr.std())
    metrics['block_ratio'] = float(metrics['block_max'] / (metrics['block_median'] + 1e-9))

    hist_path = os.path.join(tmpdir, "hist.png")
    plt.figure(figsize=(6,3))
    plt.title("Histograma (cinza)")
    plt.hist(gray.flatten(), bins=256)
    plt.tight_layout()
    plt.savefig(hist_path)
    plt.close()

    best_ssim = -1.0
    best_ref = None
    best_diff_map = None
    best_chi2 = None
    ocr_added_words = []

    for ref in ref_images:
        try:
            ref_img, ref_gray = read_and_preprocess(ref)
        except Exception:
            continue

        score, diff_map = compute_ssim_and_diff(gray, ref_gray)
        chi2 = histogram_chi2(img, ref_img)

        if score > best_ssim:
            best_ssim = score
            best_ref = ref
            best_diff_map = diff_map
            best_chi2 = chi2

            # SSIM local
            score_map = local_ssim_map(gray, ref_gray, win_size=64, step=32)
            boxes_diff, diff_mask = diff_regions_from_ssim_map(score_map, threshold=0.96, min_area=250)

            vis_mask_path = os.path.join(tmpdir, "local_diff_mask.png")
            cv2.imwrite(vis_mask_path, diff_mask)

            vis_boxes = img.copy()
            for (x,y,w,h) in boxes_diff:
                cv2.rectangle(vis_boxes, (x,y), (x+w, y+h), (0,0,255), 2)
            vis_boxes_path = os.path.join(tmpdir, "local_diff_boxes.png")
            cv2.imwrite(vis_boxes_path, cv2.cvtColor(vis_boxes, cv2.COLOR_BGR2RGB))

            # OCR
            ocr_compare = compare_ocr_texts(path_img, ref)
            ocr_added_words = ocr_compare['added_words']

    metrics['best_ssim'] = float(best_ssim)
    metrics['best_ref'] = best_ref
    metrics['best_chi2'] = float(best_chi2) if best_chi2 is not None else None

    if best_diff_map is not None:
        cmap_path = os.path.join(tmpdir, "ssim_diff.png")
        plt.figure(figsize=(4,4))
        plt.imshow(best_diff_map, cmap='viridis')
        plt.title(f"Mapa de diferença SSIM (score={best_ssim:.4f})")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(cmap_path)
        plt.close()
    else:
        cmap_path = None

    orig_path = os.path.join(tmpdir, "orig.png")
    cv2.imwrite(orig_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Heatmap blocos
    h, w = gray.shape
    block = 64
    heatmap = np.zeros((h, w), dtype=float)
    idx = 0
    vars_arr = block_variances(gray, block)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if idx >= len(vars_arr): break
            heatmap[y:y+block, x:x+block] = vars_arr[idx]
            idx += 1
    heatmap_n = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9) if heatmap.sum() > 0 else heatmap
    heatmap_path = os.path.join(tmpdir, "block_heatmap.png")
    plt.figure(figsize=(4,4))
    plt.imshow(heatmap_n, cmap='magma')
    plt.title("Heatmap var. por bloco")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    # Decisão
    reasons = []
    suspect = False
    if metrics['best_ssim'] < THRESH_SS:
        suspect = True
        reasons.append(f"SSIM baixo com referência ({metrics['best_ssim']:.3f} < {THRESH_SS})")
    if metrics['block_ratio'] > THRESH_BLOCK_RATIO:
        suspect = True
        reasons.append(f"Discrepância alta entre blocos (block_ratio={metrics['block_ratio']:.2f} > {THRESH_BLOCK_RATIO})")
    if metrics['best_chi2'] is not None and metrics['best_chi2'] > THRESH_HIST_CHI2:
        suspect = True
        reasons.append(f"Diferença de histograma (chi2={metrics['best_chi2']:.3f} > {THRESH_HIST_CHI2})")
    if metrics['lap_var'] < 50:
        reasons.append(f"Imagem bastante borrada (lap_var={metrics['lap_var']:.1f})")
    if len(ocr_added_words) > 0:
        suspect = True
        reasons.append(f"Palavras adicionadas detectadas: {', '.join(ocr_added_words)}")

    metrics['suspect'] = suspect
    metrics['reasons'] = reasons

    files = {
        'orig': orig_path,
        'edges': edges_path,
        'hist': hist_path,
        'ssim_diff': cmap_path,
        'block_heatmap': heatmap_path,
        'local_diff_mask': vis_mask_path if best_ref else None,
        'local_diff_boxes': vis_boxes_path if best_ref else None
    }
    return metrics, files

# --------------------------
# PDF
# --------------------------
def generate_pdf_report(metrics, files, output_pdf_path, image_name):
    doc = SimpleDocTemplate(output_pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Relatório de verificação - {image_name}", styles['Title']))
    story.append(Spacer(1,12))

    summary_lines = [
        f"Arquivo analisado: {image_name}",
        f"SSIM melhor com referência: {metrics.get('best_ssim', None):.4f}",
        f"Histograma chi2 (melhor ref): {metrics.get('best_chi2', None):.4f}",
        f"Var. Laplaciana (nitidez): {metrics.get('lap_var', None):.2f}",
        f"Block mean: {metrics.get('block_mean', None):.2f}",
        f"Block median: {metrics.get('block_median', None):.2f}",
        f"Block max: {metrics.get('block_max', None):.2f}",
        f"Block ratio: {metrics.get('block_ratio', None):.2f}",
        f"Decisão: {'POSSÍVEL ADULTERAÇÃO' if metrics['suspect'] else 'Provável autêntica'}"
    ]
    for line in summary_lines:
        story.append(Paragraph(line, styles['Normal']))
        story.append(Spacer(1,6))

    if metrics['reasons']:
        story.append(Spacer(1,8))
        story.append(Paragraph("<b>Motivos / Indícios:</b>", styles['Heading3']))
        for r in metrics['reasons']:
            story.append(Paragraph(f"- {r}", styles['Normal']))
            story.append(Spacer(1,4))

    story.append(Spacer(1,10))

    def add_image(path, caption=None):
        if path and os.path.exists(path):
            pil_img = PILImage.open(path)
            w, h = pil_img.size
            target_width = 400
            target_height = int(h * target_width / w)
            rlimg = RLImage(path, width=target_width, height=target_height)
            story.append(rlimg)
            if caption:
                story.append(Paragraph(caption, styles['Italic']))
            story.append(Spacer(1,8))

    add_image(files.get('orig'), "Imagem original (redimensionada)")
    add_image(files.get('edges'), "Mapa de bordas (Canny)")
    add_image(files.get('hist'), "Histograma (tons de cinza)")
    add_image(files.get('block_heatmap'), "Heatmap de variância por bloco")
    add_image(files.get('ssim_diff'), "Mapa de diferença SSIM (melhor referência)")
    add_image(files.get('local_diff_mask'), "Mapa de diferença SSIM local")
    add_image(files.get('local_diff_boxes'), "Diferenças destacadas com caixas (SSIM + OCR)")

    doc.build(story)

# --------------------------
# Main CLI
# --------------------------
def main(args):
    input_folder = args.input_folder
    ref_folder = args.ref_folder
    output_folder = args.output_folder

    ensure_dir(output_folder)

    ref_images = sorted(glob(os.path.join(ref_folder, "*.*")))
    if len(ref_images) == 0:
        print("AVISO: Nenhuma referência encontrada. Coloque prints autênticos em --ref_folder")
    else:
        print(f"{len(ref_images)} imagens de referência carregadas.")

    to_check = sorted(glob(os.path.join(input_folder, "*.*")))
    if len(to_check) == 0:
        print("Nada para analisar em", input_folder)
        return

    for img_path in to_check:
        print("Analisando:", img_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                metrics, files = analyze_image(img_path, ref_images, tmpdir)
            except Exception as e:
                print("Erro analisando", img_path, e)
                continue
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_pdf = os.path.join(output_folder, f"{base}_report.pdf")
            generate_pdf_report(metrics, files, out_pdf, os.path.basename(img_path))
            print("Relatório gerado:", out_pdf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificação de screenshots sem IA")
    parser.add_argument("--input_folder", help="Pasta com screenshots a verificar")
    parser.add_argument("--ref_folder", help="Pasta com prints de referência autênticos")
    parser.add_argument("--output_folder", help="Pasta para salvar relatórios PDF")
    args = parser.parse_args()

    if not args.input_folder:
        args.input_folder = r"C:\Users\USER\Desktop\PBL 3\to_check"
    if not args.ref_folder:
        args.ref_folder = r"C:\Users\USER\Desktop\PBL 3\references"
    if not args.output_folder:
        args.output_folder = r"C:\Users\USER\Desktop\PBL 3\reports"

    main(args)
