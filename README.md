# Verificação de Screenshots 🖼️🔍

Analisa capturas de tela para detectar possíveis manipulações **sem usar machine learning**, gerando relatórios PDF com visualizações, diferenças locais e análise de texto (OCR).

---

## Funcionalidades

- Comparação com imagens de referência usando **SSIM** e histogramas.  
- Análise de variância por blocos para detectar alterações locais.  
- Mapa de diferença **localizado** com janelas móveis.  
- OCR para identificar palavras adicionadas ou removidas.  
- Relatórios PDF com:  
  - Imagem original  
  - Heatmap de blocos  
  - Mapa de bordas (Canny)  
  - Mapa de diferença SSIM global e local  
  - Caixas em regiões suspeitas  
  - Palavras adicionadas pelo OCR  

---

## Pré-requisitos

- **Python 3.10+**  
- **Tesseract OCR** instalado:  
  - Windows: [Download Tesseract](https://github.com/tesseract-ocr/tesseract/wiki)  
  - Linux: `sudo apt install tesseract-ocr`  
  - Mac: `brew install tesseract`  
- Bibliotecas Python:

```bash
pip install opencv-python matplotlib numpy scikit-image reportlab Pillow pytesseract
