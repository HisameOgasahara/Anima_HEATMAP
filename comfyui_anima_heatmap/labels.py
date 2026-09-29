"""ComfyUI 이미지 배치에 선택 이름을 표시한다."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


def label_images(images, labels):
    width = images[0].shape[1]
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), max(12, width // 40))
    padding = max(4, width // 80)
    lines = []
    for label in labels:
        wrapped, line = [], ""
        for char in label:
            if line and font.getlength(line + char) > width - padding * 2:
                wrapped.append(line)
                line = ""
            line += char
        lines.append(wrapped + [line])
    line_height = sum(font.getmetrics())
    header = max(map(len, lines)) * line_height + padding * 2
    result = []
    for pixels, text_lines in zip(images, lines):
        # Preserve the map pixels; put the title in a separate strip.
        title = Image.new("RGB", (width, header), "#202020")
        draw = ImageDraw.Draw(title)
        for index, line in enumerate(text_lines):
            draw.text((padding, padding + index * line_height), line, font=font, fill="white")
        result.append(np.concatenate((np.asarray(title).astype(np.float32) / 255, pixels), axis=0))
    return np.stack(result)
