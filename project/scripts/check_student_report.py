"""Inspect the exported report and render review images into project scratch space."""
from __future__ import annotations
import json
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "scratch" / "report-runtime"))
import pymupdf as fitz
from PIL import Image, ImageDraw
from docx import Document
from docx.oxml.ns import qn


def main():
    pdf = fitz.open(ROOT.parent / "BaoCao_MovieLens25M.pdf")
    out = ROOT / "artifacts" / "scratch" / "report-review"
    out.mkdir(parents=True, exist_ok=True)
    all_text = []
    stats = []
    issues = []
    for i, page in enumerate(pdf):
        text = page.get_text()
        all_text.append(f"\n=== PDF PAGE {i+1} ===\n{text}")
        titles = [line for line in text.splitlines() if re.match(r"^(CHƯƠNG|MỞ ĐẦU|[1-5]\.\d\.|Bảng \d|Hình \d|TÀI LIỆU|PHỤ LỤC)", line)]
        stats.append({"page": i + 1, "chars": len(text), "titles": titles})
        if i >= 2 and len(text.strip()) < 20:
            issues.append({"page": i+1, "issue": "nearly_empty"})
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if "\ufffd" in span["text"]:
                        issues.append({"page":i+1,"issue":"replacement_character","text":span["text"]})
        pix = page.get_pixmap(matrix=fitz.Matrix(.45,.45), alpha=False)
        pix.save(out / f"thumb_{i+1:02d}.png")
    sheet = Image.new("RGB", (6*285, ((len(pdf)+5)//6)*420), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for i in range(len(pdf)):
        image = Image.open(out / f"thumb_{i+1:02d}.png")
        x, y = (i%6)*285+8, (i//6)*420+22
        sheet.paste(image,(x,y))
        draw.text((x,y-18),f"PDF page {i+1}",fill="black")
    sheet.save(out/"contact_sheet.png")
    (out/"text.txt").write_text("\n".join(all_text),encoding="utf-8")
    result={"pages":len(pdf), "page_titles":stats,"issues":issues}
    (out/"layout_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    doc=Document(ROOT.parent/"BaoCao_MovieLens25M.docx")
    template=Document(ROOT.parent/"MauBaoCao.docx")
    result["cover_text_matches"]= [p.text for p in doc.paragraphs[:48]] == [p.text for p in template.paragraphs[:48]]
    result["template_unchanged_sha256"] = __import__("hashlib").sha256((ROOT.parent/"MauBaoCao.docx").read_bytes()).hexdigest()
    result["table_widths_cm"] = [round(sum(col.width.cm for col in t.columns),2) for t in doc.tables]
    result["toc_fields"] = [n.text for n in doc._element.iter(qn("w:instrText")) if n.text and "TOC " in n.text]
    result["sparse_pages"] = [r for r in stats if r["page"] > 8 and r["chars"] < 250]
    result["footer_page_fields"] = [len([n for n in section.footer._element.iter(qn("w:instrText")) if n.text and "PAGE" in n.text]) for section in doc.sections]
    result["reference_count"] = sum(p.style.name == "Report Reference" for p in doc.paragraphs)
    # Final package audit after Word has materialized the automatic indexes.
    result["styles"] = {}
    defaults = doc.styles.element.find(qn("w:docDefaults")).find(qn("w:rPrDefault")).find(qn("w:rPr"))
    default_font = defaults.find(qn("w:rFonts")).get(qn("w:ascii"))
    default_size_pt = int(defaults.find(qn("w:sz")).get(qn("w:val"))) / 2
    def inherited(style, path):
        while style is not None:
            obj = style
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if obj is not None:
                return obj
            style = style.base_style
        return None
    for name in ["Report Body", "Report Chapter", "Report Heading 1", "Report Heading 2", "Report Table Text"]:
        style = doc.styles[name]
        size = inherited(style,"font.size")
        result["styles"][name] = {"font": inherited(style,"font.name") or default_font, "size_pt": size.pt if size is not None else default_size_pt,
                                   "line_spacing": inherited(style,"paragraph_format.line_spacing"),
                                   "color": str(inherited(style,"font.color.rgb"))}
    result["margins_cm"] = [{"left":round(s.left_margin.cm,3),"right":round(s.right_margin.cm,3),
                              "top":round(s.top_margin.cm,3),"bottom":round(s.bottom_margin.cm,3)} for s in doc.sections]
    result["table_headers_repeat"] = all(t.rows[0]._tr.trPr.find(qn("w:tblHeader")) is not None for t in doc.tables)
    body_texts = [p.text for p in doc.paragraphs if p.style.name.startswith("Report")]
    low = body_texts.index("4.5. NHÓM GIẢM CHIỀU DỮ LIỆU")
    high = body_texts.index("4.6. NHÓM HỌC TĂNG CƯỜNG")
    result["reduction_results_blank"] = all(not text or text.startswith("4.5.") for text in body_texts[low+1:high])
    result["result_sources_unchanged"] = True
    evidence_path = ROOT.parent/"BaoCao_MovieLens25M_kiemtra.json"
    evidence=json.loads(evidence_path.read_text(encoding="utf-8"))
    for source in evidence["sources"].values():
        actual=hashlib.sha256((ROOT/"artifacts"/"experiment-lab"/source["run_id"]/"result.json").read_bytes()).hexdigest()
        result["result_sources_unchanged"] &= actual == source["result_sha256"]
    assert result["cover_text_matches"] and result["reduction_results_blank"] and result["result_sources_unchanged"]
    assert result["table_headers_repeat"] and result["footer_page_fields"] == [0,1,1]
    assert not result["issues"]
    expected_sizes = [13,18,14,13,12]
    assert all(value["font"] == "Times New Roman" and value["size_pt"] == size for value,size in zip(result["styles"].values(),expected_sizes))
    evidence["final_render_validation"] = {key:value for key,value in result.items() if key not in ["page_titles","toc_fields"]}
    evidence["final_files"] = {path.name: {"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in [ROOT.parent/"BaoCao_MovieLens25M.docx",ROOT.parent/"BaoCao_MovieLens25M.pdf"]}
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"layout_check.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "page_titles"},ensure_ascii=False,indent=2))
    for number in sys.argv[1:]:
        page=pdf[int(number)-1]
        page.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(out/f"page_{int(number):02d}.png")


if __name__ == "__main__":
    main()
