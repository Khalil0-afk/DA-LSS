import argparse
import html
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--font-dir", default="C:/Windows/Fonts")
    parser.add_argument("--work", default=str(root / "runs/supplement-layout"))
    parser.add_argument("--output", default=str(root / "supplementary/Supplementary_Methods.pdf"))
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    font_dir = Path(args.font_dir)
    fonts = {"regular": font_dir / "times.ttf", "italic": font_dir / "timesi.ttf",
             "bold": font_dir / "timesbd.ttf", "bold_italic": font_dir / "timesbi.ttf"}
    for path in fonts.values():
        if not path.exists():
            raise FileNotFoundError(f"Times New Roman font file required: {path}")
    text = (root / "supplementary/SUPPLEMENTARY_METHODS.md").read_text(encoding="utf-8")
    equations = re.findall(r"\$\$\s*(.*?)\s*\$\$", text, flags=re.S)
    if args.layout:
        import matplotlib
        from matplotlib import font_manager
        from matplotlib.mathtext import MathTextParser
        for path in fonts.values():
            font_manager.fontManager.addfont(str(path))
        matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
            "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
            "mathtext.cal": "Times New Roman:italic", "mathtext.fallback": "stix"})
        engine = MathTextParser("path")
        layouts, font_paths, fallback = [], {}, []
        for index, equation in enumerate(equations):
            display = " ".join(equation.split()).replace("\\bigl", "").replace("\\bigr", "")
            display = display.replace("\\frac", "\\dfrac")
            for name in ("L", "P", "T"):
                display = display.replace("\\mathcal " + name, "\\mathcal{" + name + "}")
            parsed = engine.parse("$" + display + "$", dpi=72, prop=font_manager.FontProperties(size=13))
            glyphs = []
            for font, size, code, x, y in parsed.glyphs:
                name = font.postscript_name
                if font.get_char_index(code) == 0:
                    raise ValueError(f"Missing equation glyph U+{code:04X}")
                font_paths[name] = str(font.fname)
                glyphs.append([name, float(size), int(code), float(x), float(y)])
                if not name.startswith("TimesNewRoman"):
                    fallback.append({"equation": index + 1, "symbol": chr(code), "font": name})
            layouts.append({"width": float(parsed.width), "height": float(parsed.height),
                            "depth": float(parsed.depth), "glyphs": glyphs,
                            "rules": [list(map(float, rule)) for rule in parsed.rects]})
        (work / "layout.json").write_text(json.dumps({"layouts": layouts, "font_paths": font_paths,
            "fallback_symbols": fallback}, indent=2), encoding="utf-8")
        print(f"Laid out {len(layouts)} equations using Times New Roman")
        return
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable

    aliases = {"regular": "TNR", "italic": "TNRI", "bold": "TNRB", "bold_italic": "TNRBI"}
    for style, path in fonts.items():
        pdfmetrics.registerFont(TTFont(aliases[style], str(path)))
    pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNRB", italic="TNRI", boldItalic="TNRBI")
    from reportlab import rl_config
    rl_config.canvas_basefontname = "TNR"
    layout = json.loads((work / "layout.json").read_text(encoding="utf-8"))
    math_names = {"TimesNewRomanPSMT": "TNR", "TimesNewRomanPS-ItalicMT": "TNRI",
                  "TimesNewRomanPS-BoldMT": "TNRB", "TimesNewRomanPS-BoldItalicMT": "TNRBI"}
    for index, (name, path) in enumerate(layout["font_paths"].items()):
        if name not in math_names:
            alias = f"Math{index}"
            pdfmetrics.registerFont(TTFont(alias, path))
            math_names[name] = alias

    class Equation(Flowable):
        def __init__(self, record, index):
            super().__init__()
            self.record, self.index = record, index

        def wrap(self, width, height):
            self.width = width
            self.scale = min(1.0, (width - 45) / self.record["width"])
            self.height = self.record["height"] * self.scale + 16
            return self.width, self.height

        def draw(self):
            canvas = self.canv
            canvas.saveState()
            canvas.translate((self.width - self.record["width"] * self.scale) / 2,
                             8 + self.record["depth"] * self.scale)
            canvas.scale(self.scale, self.scale)
            canvas.setFillColor(colors.black)
            for name, size, code, x, y in self.record["glyphs"]:
                canvas.setFont(math_names[name], size)
                canvas.drawString(x, y, chr(code))
            for x, y, width, height in self.record["rules"]:
                canvas.rect(x, y, width, height, fill=1, stroke=0)
            canvas.restoreState()
            canvas.setFont("TNR", 10)
            canvas.drawRightString(self.width, self.height / 2 - 3, f"(S{self.index})")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=(8.27 * inch, 11.69 * inch),
        rightMargin=54, leftMargin=54, topMargin=48, bottomMargin=45,
        title="DA-LSS Supplementary Methods and Reproducibility Protocol", author="DA-LSS contributors")
    body = ParagraphStyle("Body", fontName="TNR", fontSize=11, leading=14.7,
                          alignment=TA_JUSTIFY, spaceAfter=7, allowWidows=0, allowOrphans=0)
    heading = ParagraphStyle("Section", fontName="TNRB", fontSize=12.5, leading=15,
                             spaceBefore=11, spaceAfter=6, keepWithNext=True)
    title = ParagraphStyle("Title", fontName="TNRB", fontSize=17, leading=21,
                           alignment=TA_CENTER, spaceAfter=10)
    story, index = [], 0
    for part in re.split(r"(\$\$.*?\$\$)", text, flags=re.S):
        if part.startswith("$$"):
            story.append(Equation(layout["layouts"][index], index + 1))
            index += 1
        else:
            for block in re.split(r"\n\s*\n", part.strip()):
                line = " ".join(block.split())
                if not line:
                    continue
                if line.startswith("# "):
                    story.append(Paragraph(html.escape(line[2:]), title))
                elif line.startswith("## "):
                    story.append(Paragraph(html.escape(line[3:]), heading))
                else:
                    story.append(Paragraph(html.escape(line).replace("`", ""), body))
    for index in range(len(story) - 1):
        if isinstance(story[index], Paragraph) and isinstance(story[index + 1], Equation) and len(story[index].text) < 300:
            story[index].keepWithNext = True
    def footer(canvas, document):
        canvas.setFont("TNR", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(54, 25, "DA-LSS | Supplementary Methods | v0.3.0")
        canvas.drawRightString(document.pagesize[0] - 54, 25, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    audit = {"body_font": "Times New Roman", "equation_letter_fonts": ["Times New Roman", "Times New Roman Italic"],
             "equation_count": len(equations), "equations_are_vector_text": True, "equation_images_used": False,
             "font_files_distributed": False, "math_font_postscript_names": sorted(layout["font_paths"]),
             "times_compatible_symbol_fallbacks": layout["fallback_symbols"]}
    (root / "supplementary/font_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
