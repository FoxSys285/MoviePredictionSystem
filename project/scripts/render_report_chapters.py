"""Render the requested report chapters into a copy of the supplied DOCX template."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "project" / "docs" / "report-chapters-1-4.md"
TEMPLATE = REPO.parent / "Nhom_05.docx"
OUTPUT = REPO / "Nhom_05_Bao_Cao_Tong_Quan.docx"


def set_run_font(run, size: int = 13, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    fonts = run._r.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")


def set_style_font(style, size: int, bold: bool = False) -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")


def add_outline_level(style, level: int) -> None:
    ppr = style.element.get_or_add_pPr()
    outline = ppr.find(qn("w:outlineLvl"))
    if outline is None:
        outline = OxmlElement("w:outlineLvl")
        ppr.append(outline)
    outline.set(qn("w:val"), str(level))


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    set_style_font(normal, 13)
    normal.paragraph_format.line_spacing = 1.3
    normal.paragraph_format.space_before = Pt(6)
    normal.paragraph_format.space_after = Pt(6)

    specs = {
        "Report Body": (13, False),
        "Report Chapter": (18, True),
        "Report Heading 1": (14, True),
        "Report Heading 2": (13, True),
        "Report Figure Caption": (13, False),
        "Report Table Caption": (13, False),
        "Report Table Text": (12, False),
        "Report Equation": (13, False),
        "Report Reference": (13, False),
        "Report Field": (13, False),
    }
    for name, (size, bold) in specs.items():
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = normal
        set_style_font(style, size, bold)
        fmt = style.paragraph_format
        fmt.line_spacing = 1.3
        fmt.space_before = Pt(6)
        fmt.space_after = Pt(6)
        fmt.first_line_indent = Cm(1) if name == "Report Body" else Cm(0)
        fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    chapter = doc.styles["Report Chapter"].paragraph_format
    chapter.alignment = WD_ALIGN_PARAGRAPH.CENTER
    chapter.keep_with_next = True
    chapter.page_break_before = True
    add_outline_level(doc.styles["Report Chapter"], 0)

    h1 = doc.styles["Report Heading 1"].paragraph_format
    h1.keep_with_next = True
    add_outline_level(doc.styles["Report Heading 1"], 1)

    h2 = doc.styles["Report Heading 2"].paragraph_format
    h2.keep_with_next = True
    add_outline_level(doc.styles["Report Heading 2"], 2)

    for name in ("Report Figure Caption", "Report Table Caption"):
        caption = doc.styles[name].paragraph_format
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.keep_with_next = True
        caption.first_line_indent = Cm(0)

    table_text = doc.styles["Report Table Text"].paragraph_format
    table_text.space_before = Pt(0)
    table_text.space_after = Pt(0)
    table_text.first_line_indent = Cm(0)

    equation = doc.styles["Report Equation"].paragraph_format
    equation.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.first_line_indent = Cm(0)
    equation.keep_together = True

    reference = doc.styles["Report Reference"].paragraph_format
    reference.first_line_indent = Cm(-0.8)
    reference.left_indent = Cm(0.8)

    field = doc.styles["Report Field"].paragraph_format
    field.alignment = WD_ALIGN_PARAGRAPH.LEFT
    field.first_line_indent = Cm(0)
    field.space_before = Pt(0)
    field.space_after = Pt(0)

    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3.5)
        section.right_margin = Cm(2.5)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.footer_distance = Cm(1.25)
        section.header_distance = Cm(1.25)

    update = doc.settings.element.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        doc.settings.element.append(update)
    update.set(qn("w:val"), "true")


def set_page_number(section, number_format: str | None) -> None:
    section.footer.is_linked_to_previous = False
    footer = section.footer
    for child in list(footer._element):
        if child.tag != qn("w:sectPr"):
            footer._element.remove(child)
    paragraph = footer.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)

    if number_format is None:
        return

    for child in list(section._sectPr):
        if child.tag == qn("w:pgNumType"):
            section._sectPr.remove(child)
    page_type = OxmlElement("w:pgNumType")
    page_type.set(qn("w:fmt"), number_format)
    page_type.set(qn("w:start"), "1")
    section._sectPr.append(page_type)

    run = paragraph.add_run()
    set_run_font(run, 12)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t")
    result.text = "1" if number_format == "decimal" else "i"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, instruction, separate, result, end):
        run._r.append(node)


def find_paragraph(doc: Document, text: str, exact: bool = True):
    for paragraph in doc.paragraphs:
        value = paragraph.text.strip()
        if (value == text) if exact else value.startswith(text):
            return paragraph
    raise ValueError(f"Could not find paragraph: {text}")


def body_child_index(doc: Document, element) -> int:
    return list(doc._element.body).index(element)


def remove_between(doc: Document, start_paragraph, end_paragraph) -> None:
    body = doc._element.body
    children = list(body)
    left = children.index(start_paragraph._p) + 1
    right = children.index(end_paragraph._p)
    for child in children[left:right]:
        ppr = child.find(qn("w:pPr")) if child.tag == qn("w:p") else None
        if ppr is not None and ppr.find(qn("w:sectPr")) is not None:
            continue
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def remove_from_paragraph_to_end(doc: Document, start_paragraph) -> None:
    body = doc._element.body
    children = list(body)
    start = children.index(start_paragraph._p)
    for child in children[start:]:
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def add_field_after(doc: Document, anchor, instruction: str, cached: str) -> None:
    paragraph = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    pstyle = OxmlElement("w:pStyle")
    pstyle.set(qn("w:val"), doc.styles["Report Field"].style_id)
    ppr.append(pstyle)
    paragraph.append(ppr)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")
    rpr.append(fonts)
    run.append(rpr)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    code = OxmlElement("w:instrText")
    code.set(qn("xml:space"), "preserve")
    code.text = f" {instruction} "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = cached
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, code, separate, text, end):
        run.append(node)
    paragraph.append(run)
    anchor._p.addnext(paragraph)


def replace_frontmatter_fields(doc: Document) -> None:
    toc = find_paragraph(doc, "MỤC LỤC")
    acronyms = find_paragraph(doc, "DANH MỤC CÁC KÝ HIỆU VÀ CHỮ VIẾT TẮT")
    remove_between(doc, toc, acronyms)
    add_field_after(doc, toc, 'TOC \\o "1-3" \\h \\z \\u', "Cập nhật trường mục lục trong Word")

    figures = find_paragraph(doc, "DANH MỤC HÌNH")
    tables = find_paragraph(doc, "DANH MỤC BẢNG")
    intro = find_paragraph(doc, "CHƯƠNG 1: TỔNG QUAN")
    remove_between(doc, figures, tables)
    add_field_after(doc, figures, 'TOC \\h \\z \\t "Report Figure Caption,1"', "Cập nhật danh mục hình trong Word")
    remove_between(doc, tables, intro)
    add_field_after(doc, tables, 'TOC \\h \\z \\t "Report Table Caption,1"', "Cập nhật danh mục bảng trong Word")


def add_acronym_table(doc: Document) -> None:
    values = [
        ("Ký hiệu", "Tiếng Anh", "Diễn giải"),
        ("CF", "Collaborative Filtering", "Lọc cộng tác"),
        ("NCF", "Neural Collaborative Filtering", "Lọc cộng tác bằng mạng nơ-ron"),
        ("SVD-CF", "Singular Value Decomposition Collaborative Filtering", "Phân rã nhân tố tiềm ẩn để dự đoán rating"),
        ("TruncatedSVD", "Truncated Singular Value Decomposition", "Giảm chiều không center dữ liệu"),
        ("PCA", "Principal Component Analysis", "Phân tích thành phần chính"),
        ("RMSE", "Root Mean Squared Error", "Căn sai số bình phương trung bình"),
        ("MAE", "Mean Absolute Error", "Sai số tuyệt đối trung bình"),
        ("AUC-ROC", "Area Under the ROC Curve", "Diện tích dưới đường cong ROC"),
        ("DBI", "Davies–Bouldin Index", "Chỉ số Davies–Bouldin"),
        ("CV", "Cross-validation", "Đánh giá chéo"),
        ("MLP", "Multilayer Perceptron", "Mạng truyền thẳng nhiều lớp"),
        ("RL", "Reinforcement Learning", "Học tăng cường"),
        ("DQN", "Deep Q-Network", "Mạng nơ-ron xấp xỉ giá trị Q"),
        ("DBSCAN", "Density-Based Spatial Clustering of Applications with Noise", "Phân cụm dựa trên mật độ"),
        ("TPE", "Tree-structured Parzen Estimator", "Bộ lấy mẫu tối ưu siêu tham số"),
    ]
    anchor = next(
        paragraph for paragraph in doc.paragraphs
        if paragraph.text.strip().startswith("Viết tắt")
        and "Tiếng Anh" in paragraph.text
        and "Tiếng Việt" in paragraph.text
    )
    anchor.text = "Danh mục ký hiệu và chữ viết tắt dùng trong báo cáo"
    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    widths = [2.7, 5.4, 6.9]
    for column, width in zip(table.columns, widths):
        column.width = Cm(width)
    for row_index, data in enumerate(values):
        cells = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for cell, text, width in zip(cells, data, widths):
            cell.width = Cm(width)
            cell.text = text
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.style = doc.styles["Report Table Text"]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_index == 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    set_run_font(run, 12, bold=row_index == 0)
        if row_index == 0:
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "EDEDED")
            for cell in cells:
                cell._tc.get_or_add_tcPr().append(shading)
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(header)
    body = doc._element.body
    body.remove(table._tbl)
    anchor._p.addnext(table._tbl)


def strip_markdown(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append([strip_markdown(cell) for cell in cells])
    return rows


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    columns = len(rows[0])
    table = doc.add_table(rows=1, cols=columns)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    available_width = 15.0
    widths = [available_width / columns] * columns
    for col, width in zip(table.columns, widths):
        col.width = Cm(width)
    for row_index, values in enumerate(rows):
        cells = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for index, cell in enumerate(cells):
            value = values[index] if index < len(values) else ""
            cell.width = Cm(widths[index])
            cell.text = value
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.style = doc.styles["Report Table Text"]
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_index == 0 or index > 0 else WD_ALIGN_PARAGRAPH.LEFT
                for run in paragraph.runs:
                    set_run_font(run, 12, bold=row_index == 0)
        if row_index == 0:
            header = OxmlElement("w:tblHeader")
            header.set(qn("w:val"), "true")
            table.rows[0]._tr.get_or_add_trPr().append(header)
    for row in table.rows:
        cant_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(cant_split)
    doc.add_paragraph("", style="Report Table Text")


def add_body_paragraph(doc: Document, text: str) -> None:
    text = strip_markdown(text)
    paragraph = doc.add_paragraph(style="Report Body")
    paragraph.add_run(text)

    if text.startswith("[CHÈN") or text.startswith("Vị trí chèn"):
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.first_line_indent = Cm(0)
        paragraph.paragraph_format.keep_together = True
        for run in paragraph.runs:
            run.italic = True
            run.font.color.rgb = RGBColor(100, 100, 100)
    elif re.match(r"^\[\d+\]", text):
        paragraph.style = doc.styles["Report Reference"]


def render_markdown(doc: Document, markdown: str) -> None:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].strip()
        if not raw:
            index += 1
            continue

        if raw.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            add_table(doc, parse_table(table_lines))
            continue

        if raw.startswith("# "):
            paragraph = doc.add_paragraph(style="Report Chapter")
            run = paragraph.add_run(strip_markdown(raw[2:]).upper())
            set_run_font(run, 18, bold=True)
            index += 1
            continue

        if raw.startswith("## "):
            paragraph = doc.add_paragraph(style="Report Heading 1")
            run = paragraph.add_run(strip_markdown(raw[3:]).upper())
            set_run_font(run, 14, bold=True)
            index += 1
            continue

        if raw.startswith("### "):
            paragraph = doc.add_paragraph(style="Report Heading 2")
            run = paragraph.add_run(strip_markdown(raw[4:]))
            set_run_font(run, 13, bold=True)
            index += 1
            continue

        if raw.startswith("**Bảng ") or raw.startswith("**Hình "):
            caption_style = "Report Table Caption" if raw.startswith("**Bảng ") else "Report Figure Caption"
            paragraph = doc.add_paragraph(style=caption_style)
            run = paragraph.add_run(strip_markdown(raw))
            set_run_font(run, 13)
            index += 1
            continue

        if raw.startswith("**") and raw.endswith("**"):
            paragraph = doc.add_paragraph(style="Report Equation")
            run = paragraph.add_run(strip_markdown(raw))
            set_run_font(run, 13, bold=True)
            index += 1
            continue

        add_body_paragraph(doc, raw)
        index += 1


def build() -> Path:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)

    doc = Document(TEMPLATE)
    configure_styles(doc)
    replace_frontmatter_fields(doc)
    add_acronym_table(doc)

    chapter_one = find_paragraph(doc, "CHƯƠNG 1: TỔNG QUAN")
    remove_from_paragraph_to_end(doc, chapter_one)
    render_markdown(doc, SOURCE.read_text(encoding="utf-8"))

    if len(doc.sections) >= 3:
        set_page_number(doc.sections[0], None)
        set_page_number(doc.sections[1], "lowerRoman")
        set_page_number(doc.sections[2], "decimal")

    doc.core_properties.title = "Báo cáo đồ án MovieLens 25M — Chương 1 đến Chương 4"
    doc.core_properties.subject = "Cơ sở lý thuyết, dữ liệu và thiết kế thực nghiệm, thực nghiệm và so sánh"
    doc.core_properties.keywords = "MovieLens 25M, rating prediction, NCF, recommender systems"
    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
