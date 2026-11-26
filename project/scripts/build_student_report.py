"""Build the MovieLens course report from the supplied DOCX template and saved runs."""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
TEMPLATE = REPO / "MauBaoCao.docx"
OUTPUT = REPO / "BaoCao_MovieLens25M.docx"
RUN_ROOT = PROJECT / "artifacts" / "experiment-lab"
RUN_IDS = {
    "rating": "20261126T015401-4c1f33e3",
    "classification": "20261126T060857-59d1c9fa",
    "clustering": "20261126T063703-2538ed62",
    "association": "20261126T064427-dda71bbb",
    "reinforcement": "20261126T064844-25de2fc0",
}
FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD_PATH = Path("C:/Windows/Fonts/arialbd.ttf")
BLACK = (27, 37, 47)
BLUE = (47, 104, 168)
TEAL = (42, 143, 140)
GOLD = (220, 155, 52)
RED = (181, 82, 75)
GRAY = (105, 117, 128)
PALE = (232, 239, 245)
WHITE = (255, 255, 255)
PAGE_WIDTH_CM = 15.0


def load_run(name: str) -> dict:
    path = RUN_ROOT / RUN_IDS[name] / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def vi(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    value = float(value)
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def vi_int(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    return f"{int(value):,}".replace(",", ".")


def mib(byte_count: int | float | None) -> str:
    return vi(float(byte_count or 0) / (1024 * 1024), 2)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD_PATH if bold and FONT_BOLD_PATH.exists() else FONT_PATH
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                  used_font: ImageFont.ImageFont, fill=BLACK, anchor="mm") -> None:
    draw.text(xy, text, font=used_font, fill=fill, anchor=anchor, align="center")


def chart_grouped(
    path: Path,
    title: str,
    categories: list[str],
    series: list[tuple[str, tuple[int, int, int], list[float]]],
    ymin: float,
    ymax: float,
    ticks: list[float],
    tick_format,
    errors: list[float] | None = None,
) -> None:
    width, height = 1800, 980
    left, right, top, bottom = 165, 70, 130, 190
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw_centered(draw, (width // 2, 58), title, font(38, True))
    plot_right, plot_bottom = width - right, height - bottom

    def y_of(value: float) -> int:
        return round(plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - top))

    for tick in ticks:
        y = y_of(tick)
        draw.line((left, y, plot_right, y), fill=(218, 224, 229), width=2)
        draw.text((left - 20, y), tick_format(tick), font=font(23), fill=GRAY, anchor="rm")
    draw.line((left, top, left, plot_bottom), fill=BLACK, width=3)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=BLACK, width=3)
    group_width = (plot_right - left) / len(categories)
    count = len(series)
    bar_width = min(106, group_width / (count + 1))
    base_y = y_of(ymin)
    for category_index, category in enumerate(categories):
        center = left + group_width * (category_index + 0.5)
        for series_index, (label, color, values) in enumerate(series):
            x = center + (series_index - (count - 1) / 2) * bar_width
            y = y_of(values[category_index])
            draw.rectangle((int(x - bar_width * 0.34), y, int(x + bar_width * 0.34), base_y),
                           fill=color)
            draw.text((int(x), y - 13), tick_format(values[category_index]),
                      font=font(18), fill=BLACK, anchor="ms")
            if errors is not None and series_index == 0:
                err = errors[category_index]
                hi, lo = y_of(values[category_index] + err), y_of(values[category_index] - err)
                draw.line((int(x), hi, int(x), lo), fill=BLACK, width=3)
                draw.line((int(x - 12), hi, int(x + 12), hi), fill=BLACK, width=3)
                draw.line((int(x - 12), lo, int(x + 12), lo), fill=BLACK, width=3)
        draw.multiline_text((center, plot_bottom + 27), category, font=font(22),
                            fill=BLACK, anchor="ma", align="center", spacing=4)
    legend_y = height - 45
    total_legend_width = len(series) * 330
    x = (width - total_legend_width) // 2
    for label, color, _values in series:
        draw.rectangle((x, legend_y - 14, x + 25, legend_y + 11), fill=color)
        draw.text((x + 38, legend_y - 2), label, font=font(22), fill=BLACK, anchor="lm")
        x += 330
    image.save(path, quality=95)


def chart_line(path: Path, title: str, x_values: list[int],
               series: list[tuple[str, tuple[int, int, int], list[float]]],
               ymin: float, ymax: float) -> None:
    width, height = 1800, 980
    left, right, top, bottom = 160, 70, 130, 180
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw_centered(draw, (width // 2, 58), title, font(38, True))
    plot_right, plot_bottom = width - right, height - bottom

    def x_of(index: int) -> int:
        if len(x_values) == 1:
            return (left + plot_right) // 2
        return round(left + index * (plot_right - left) / (len(x_values) - 1))

    def y_of(value: float) -> int:
        return round(plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - top))

    for tick_index in range(6):
        value = ymin + (ymax - ymin) * tick_index / 5
        y = y_of(value)
        draw.line((left, y, plot_right, y), fill=(218, 224, 229), width=2)
        draw.text((left - 18, y), vi(value, 2), font=font(22), fill=GRAY, anchor="rm")
    draw.line((left, top, left, plot_bottom), fill=BLACK, width=3)
    draw.line((left, plot_bottom, plot_right, plot_bottom), fill=BLACK, width=3)
    for index, value in enumerate(x_values):
        x = x_of(index)
        draw.text((x, plot_bottom + 28), str(value), font=font(22), fill=BLACK, anchor="mt")
    for label, color, values in series:
        points = [(x_of(i), y_of(v)) for i, v in enumerate(values)]
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=WHITE, width=2)
    legend_y = height - 43
    total_legend_width = len(series) * 405
    x = (width - total_legend_width) // 2
    for label, color, _values in series:
        draw.line((x, legend_y, x + 34, legend_y), fill=color, width=6)
        draw.text((x + 46, legend_y), label, font=font(22), fill=BLACK, anchor="lm")
        x += 405
    image.save(path, quality=95)


def chart_split(path: Path) -> None:
    width, height = 1800, 650
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw_centered(draw, (width // 2, 64), "Tách dữ liệu theo thời gian trong từng người dùng",
                  font(38, True))
    left, top, bar_width, bar_height = 160, 205, 1480, 130
    widths = [round(bar_width * 0.70), round(bar_width * 0.15), 0]
    widths[2] = bar_width - widths[0] - widths[1]
    colors = [BLUE, TEAL, GOLD]
    labels = ["Train · 70%", "Validation · 15%", "Test · 15%"]
    x = left
    for segment_width, color, label in zip(widths, colors, labels):
        draw.rectangle((x, top, x + segment_width, top + bar_height), fill=color)
        draw_centered(draw, (x + segment_width // 2, top + bar_height // 2),
                      label, font(28, True), WHITE)
        x += segment_width
    draw_centered(draw, (width // 2, 410),
                  "Các timestamp bằng nhau được giữ cùng một tập; mỗi user chỉ dùng lịch sử quá khứ để dự đoán đoạn sau.",
                  font(25), BLACK)
    draw_centered(draw, (width // 2, 480),
                  "Validation chọn cấu hình và epoch. Test được giữ riêng đến bước báo cáo cuối.",
                  font(25), GRAY)
    image.save(path, quality=95)


def chart_architecture(path: Path) -> None:
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw_centered(draw, (width // 2, 65), "Kiến trúc ứng dụng MoviePredictionSystem",
                  font(38, True))
    labels = [
        ("MovieLens CSV", "ratings · movies\ntags · links"),
        ("Data layer", "kiểm tra · làm sạch\nchia thời gian"),
        ("ML services", "fit · predict\nevaluate"),
        ("Artifacts", "mô hình · runs\nCSV · JSON"),
        ("FastAPI", "REST · xác thực\ncông việc nền"),
        ("React + TS", "tìm phim · lịch sử\nTop-N"),
    ]
    colors = [PALE, (221, 237, 235), (226, 235, 247),
              (249, 237, 215), (228, 232, 239), (225, 238, 244)]
    left, right, y, box_h = 70, 1730, 265, 220
    box_w = 250
    gap = (right - left - box_w * len(labels)) / (len(labels) - 1)
    for index, ((title, detail), fill) in enumerate(zip(labels, colors)):
        x = round(left + index * (box_w + gap))
        draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=20,
                               fill=fill, outline=(156, 170, 181), width=3)
        draw_centered(draw, (x + box_w // 2, y + 65), title, font(25, True))
        draw.multiline_text((x + box_w // 2, y + 140), detail, font=font(21),
                            fill=GRAY, anchor="mm", align="center", spacing=7)
        if index < len(labels) - 1:
            start_x = x + box_w + 8
            end_x = round(x + box_w + gap - 8)
            cy = y + box_h // 2
            draw.line((start_x, cy, end_x, cy), fill=BLUE, width=5)
            draw.polygon([(end_x, cy), (end_x - 15, cy - 10), (end_x - 15, cy + 10)], fill=BLUE)
    draw_centered(draw, (width // 2, 585),
                  "Huấn luyện chạy ngoài HTTP request; giao diện gọi API để xem dự đoán, gợi ý và trạng thái thực nghiệm.",
                  font(24), BLACK)
    image.save(path, quality=95)


def create_charts(directory: Path, runs: dict) -> dict[str, Path]:
    paths = {key: directory / f"{key}.png" for key in
             ("split", "architecture", "ratings", "regression", "ncf", "classification", "clustering", "rl")}
    chart_split(paths["split"])
    chart_architecture(paths["architecture"])
    rating_counts = [393068, 776815, 399490, 1640868, 1262797, 4896928,
                     3177318, 6639798, 2200539, 3612474]
    chart_grouped(
        paths["ratings"], "Phân bố rating trong MovieLens 25M",
        ["0,5", "1,0", "1,5", "2,0", "2,5", "3,0", "3,5", "4,0", "4,5", "5,0"],
        [("Số lượt đánh giá", BLUE, [n / 1_000_000 for n in rating_counts])],
        0, 7, [0, 1, 2, 3, 4, 5, 6, 7], lambda x: f"{x:.0f}M",
    )
    rating = runs["rating"]
    order = ["linear", "xgboost", "svd", "ncf"]
    labels = ["Linear\nRegression", "XGBoost", "SVD-CF", "NCF"]
    cv = [rating["cv"][name + "_raw"]["metrics"]["rmse"]["mean"] for name in order]
    std = [rating["cv"][name + "_raw"]["metrics"]["rmse"]["std"] for name in order]
    test = [next(r["rmse"] for r in rating["rows"] if r["algorithm"] == name and r["split"] == "test")
            for name in order]
    chart_grouped(
        paths["regression"], "RMSE dự đoán rating: CV theo thời gian và Test",
        labels, [("CV mean ± std", BLUE, cv), ("Test", TEAL, test)],
        0.75, 0.95, [0.75, 0.80, 0.85, 0.90, 0.95],
        lambda x: vi(x, 2), errors=std,
    )
    ncf_record = next(t for t in rating["tuning"] if t["algorithm"] == "ncf")
    curve = ncf_record["learning_curve"]
    epochs = [entry["epoch"] for entry in curve]
    train_mse = [entry["train_rating_mse"] for entry in curve]
    valid_mse = [entry["validation_rmse"] ** 2 for entry in curve]
    chart_line(
        paths["ncf"], "NCF: Train MSE và Validation MSE theo epoch",
        epochs, [("Train MSE (mẫu 50.000)", BLUE, train_mse),
                 ("Validation MSE = RMSE²", RED, valid_mse)],
        min(min(train_mse), min(valid_mse)) - 0.03,
        max(max(train_mse), max(valid_mse)) + 0.03,
    )
    classification = runs["classification"]
    cls_order = ["logistic", "random_forest"]
    cls_labels = ["Logistic\nRegression", "Random Forest"]
    cls_rows = {
        name: next(r for r in classification["rows"]
                   if r["algorithm"] == name and r["split"] == "test")
        for name in cls_order
    }
    chart_grouped(
        paths["classification"], "Phân loại nhãn Thích: F1 và AUC-ROC trên Test",
        cls_labels,
        [("F1", BLUE, [cls_rows[name]["f1"] for name in cls_order]),
         ("AUC-ROC", GOLD, [cls_rows[name]["auc_roc"] for name in cls_order])],
        0.70, 0.85, [0.70, 0.73, 0.76, 0.79, 0.82, 0.85],
        lambda x: vi(x, 2),
    )
    clustering = runs["clustering"]
    rec_rows = clustering["recommendation"]["rows"]
    rec_algorithms = ["popularity", "kmeans", "dbscan"]
    rec_labels = ["Popularity", "K-Means", "DBSCAN"]
    chart_grouped(
        paths["clustering"], "NDCG@10 trên cùng cohort đánh giá",
        rec_labels,
        [
            ("Validation", BLUE, [next(r["ndcg_at_10"] for r in rec_rows
                                      if r["algorithm"] == a and r["split"] == "validation")
                                  for a in rec_algorithms]),
            ("Test", TEAL, [next(r["ndcg_at_10"] for r in rec_rows
                                 if r["algorithm"] == a and r["split"] == "test")
                            for a in rec_algorithms]),
        ],
        0.052, 0.071, [0.052, 0.056, 0.060, 0.064, 0.068, 0.071],
        lambda x: vi(x, 3),
    )
    rl_rows = {r["algorithm"]: r for r in runs["reinforcement"]["rows"] if r["split"] == "test"}
    rl_algorithms = ["dqn", "random", "popular"]
    chart_grouped(
        paths["rl"], "Reward trung bình trên mỗi action trong Test",
        ["DQN", "Random", "Popularity"],
        [("Reward/action", BLUE, [rl_rows[a]["average_reward"] for a in rl_algorithms])],
        0, 0.04, [0, 0.01, 0.02, 0.03, 0.04],
        lambda x: vi(x, 2),
    )
    return paths


def set_run_font(run, size: int = 13, bold: bool = False, italic: bool = False) -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    fonts = run._r.get_or_add_rPr().get_or_add_rFonts()
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Times New Roman")


def outline_level(style, level: int) -> None:
    ppr = style.element.get_or_add_pPr()
    old = ppr.find(qn("w:outlineLvl"))
    if old is not None:
        ppr.remove(old)
    value = OxmlElement("w:outlineLvl")
    value.set(qn("w:val"), str(level))
    ppr.append(value)


def add_styles(doc: Document) -> None:
    specs = {
        "Report Body": (13, False),
        "Report Chapter": (18, True),
        "Report Heading 1": (14, True),
        "Report Heading 2": (13, True),
        "Report Front": (18, True),
        "Report Figure Caption": (13, False),
        "Report Table Caption": (13, False),
        "Report Table Text": (12, False),
        "Report Equation": (13, False),
        "Report Reference": (13, False),
        "Report Field": (13, False),
    }
    for name, (size, bold) in specs.items():
        style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        style.base_style = doc.styles["Normal"]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for key in ("ascii", "hAnsi", "eastAsia", "cs"):
            fonts.set(qn(f"w:{key}"), "Times New Roman")
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
    outline_level(doc.styles["Report Chapter"], 0)
    for name, level in (("Report Heading 1", 1), ("Report Heading 2", 2)):
        doc.styles[name].paragraph_format.keep_with_next = True
        outline_level(doc.styles[name], level)
    for name in ("Report Front", "Report Figure Caption", "Report Table Caption", "Report Equation"):
        style = doc.styles[name]
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.keep_with_next = name == "Report Table Caption"
    doc.styles["Report Front"].paragraph_format.page_break_before = True
    table = doc.styles["Report Table Text"].paragraph_format
    table.first_line_indent = Cm(0)
    table.space_before = Pt(0)
    table.space_after = Pt(0)
    doc.styles["Report Equation"].paragraph_format.keep_together = True
    reference = doc.styles["Report Reference"].paragraph_format
    reference.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    reference.left_indent = Cm(0.8)
    reference.first_line_indent = Cm(-0.8)
    field = doc.styles["Report Field"].paragraph_format
    field.alignment = WD_ALIGN_PARAGRAPH.LEFT
    field.first_line_indent = Cm(0)
    field.space_before = Pt(0)
    field.space_after = Pt(0)
    update = doc.settings.element.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        doc.settings.element.append(update)
    update.set(qn("w:val"), "true")


def set_page_number(section, number_format: str) -> None:
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.footer_distance = Cm(1.25)
    section.header_distance = Cm(1.25)
    section.footer.is_linked_to_previous = False
    footer = section.footer
    for paragraph in list(footer._element.findall(qn("w:p"))):
        footer._element.remove(paragraph)
    p = footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    set_run_font(run, 12)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    code = OxmlElement("w:instrText")
    code.set(qn("xml:space"), "preserve")
    code.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    result = OxmlElement("w:t")
    result.text = "i" if number_format == "lowerRoman" else "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, code, separate, result, end):
        run._r.append(node)
    for old in list(section._sectPr):
        if old.tag == qn("w:pgNumType"):
            section._sectPr.remove(old)
    page_type = OxmlElement("w:pgNumType")
    page_type.set(qn("w:fmt"), number_format)
    page_type.set(qn("w:start"), "1")
    section._sectPr.append(page_type)


def add_field(paragraph, instruction: str, result: str) -> None:
    run = paragraph.add_run()
    set_run_font(run, 13)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    begin.set(qn("w:dirty"), "true")
    code = OxmlElement("w:instrText")
    code.set(qn("xml:space"), "preserve")
    code.text = f" {instruction} "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    value = OxmlElement("w:t")
    value.text = result
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for node in (begin, code, separate, value, end):
        run._r.append(node)


def add_frontmatter(doc: Document) -> None:
    front = doc.add_paragraph("MỤC LỤC", "Report Front")
    front.paragraph_format.page_break_before = False
    field_p = doc.add_paragraph(style="Report Field")
    add_field(field_p, 'TOC \\o "1-3" \\h \\z \\u',
              "Mục lục tự động sẽ cập nhật khi mở bằng Microsoft Word.")

    title = doc.add_paragraph("DANH MỤC CÁC KÝ HIỆU VÀ CHỮ VIẾT TẮT", "Report Front")
    values = [
        ["Ký hiệu", "Tên tiếng Anh", "Diễn giải"],
        ["AUC-ROC", "Area Under the ROC Curve", "Diện tích dưới đường cong ROC"],
        ["CF", "Collaborative Filtering", "Lọc cộng tác"],
        ["CV", "Cross-validation", "Đánh giá chéo"],
        ["DBI", "Davies–Bouldin Index", "Chỉ số Davies–Bouldin"],
        ["DBSCAN", "Density-Based Spatial Clustering of Applications with Noise", "Phân cụm theo mật độ"],
        ["DQN", "Deep Q-Network", "Mạng xấp xỉ giá trị Q"],
        ["F1", "F1-score", "Trung bình điều hòa giữa Precision và Recall"],
        ["MAE", "Mean Absolute Error", "Sai số tuyệt đối trung bình"],
        ["MLP", "Multilayer Perceptron", "Mạng truyền thẳng nhiều lớp"],
        ["NCF", "Neural Collaborative Filtering", "Lọc cộng tác bằng mạng nơ-ron"],
        ["NDCG", "Normalized Discounted Cumulative Gain", "Độ lợi tích lũy giảm dần chuẩn hóa"],
        ["PCA", "Principal Component Analysis", "Phân tích thành phần chính"],
        ["RMSE", "Root Mean Squared Error", "Căn sai số bình phương trung bình"],
        ["SVD-CF", "Singular Value Decomposition Collaborative Filtering", "Phân rã nhân tố để dự đoán rating"],
    ]
    add_table_rows(doc, values)

    for heading, instruction, fallback in [
        ("DANH MỤC HÌNH", 'TOC \\h \\z \\t "Report Figure Caption,1"',
         "Danh mục hình tự động sẽ cập nhật khi mở bằng Microsoft Word."),
        ("DANH MỤC BẢNG", 'TOC \\h \\z \\t "Report Table Caption,1"',
         "Danh mục bảng tự động sẽ cập nhật khi mở bằng Microsoft Word."),
    ]:
        p = doc.add_paragraph(heading, "Report Front")
        field_p = doc.add_paragraph(style="Report Field")
        add_field(field_p, instruction, fallback)
    # The section after the untouched two-page cover starts with Roman numerals.
    set_page_number(doc.sections[-1], "lowerRoman")


def add_table_rows(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    col_count = len(rows[0])
    table = doc.add_table(rows=1, cols=col_count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    widths = [PAGE_WIDTH_CM / col_count] * col_count
    for column, width in zip(table.columns, widths):
        column.width = Cm(width)
    for row_index, values in enumerate(rows):
        cells = table.rows[0].cells if row_index == 0 else table.add_row().cells
        for index, cell in enumerate(cells):
            text = values[index] if index < len(values) else ""
            cell.width = Cm(widths[index])
            cell.text = str(text)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.style = doc.styles["Report Table Text"]
                paragraph.alignment = (
                    WD_ALIGN_PARAGRAPH.CENTER
                    if row_index == 0 or index > 0
                    else WD_ALIGN_PARAGRAPH.LEFT
                )
                for run in paragraph.runs:
                    set_run_font(run, 12, bold=row_index == 0)
            tcpr = cell._tc.get_or_add_tcPr()
            for shading in list(tcpr.findall(qn("w:shd"))):
                tcpr.remove(shading)
        trpr = table.rows[row_index]._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        trpr.append(cant_split)
        if row_index == 0:
            header = OxmlElement("w:tblHeader")
            header.set(qn("w:val"), "true")
            trpr.append(header)
    spacer = doc.add_paragraph(style="Report Table Text")
    spacer.paragraph_format.line_spacing = 0.7


def result_tables(runs: dict) -> dict[str, list[list[str]]]:
    rating = runs["rating"]
    regression = [["Mô hình", "RMSE CV (mean ± std)", "MAE CV (mean ± std)",
                   "RMSE Validation", "RMSE Test", "MAE Test", "R² Test"]]
    costs = [["Mô hình", "Fit cuối (s)", "Inference (µs/rating)",
              "Model (MiB)", "Peak RSS (MiB)"]]
    model_labels = {
        "linear": "Linear Regression",
        "xgboost": "XGBoost Regressor",
        "svd": "SVD Collaborative Filtering",
        "ncf": "Neural Collaborative Filtering",
    }
    for name, label in model_labels.items():
        metric = rating["cv"][name + "_raw"]["metrics"]
        validation = next(r for r in rating["rows"]
                          if r["algorithm"] == name and r["split"] == "validation")
        test = next(r for r in rating["rows"]
                    if r["algorithm"] == name and r["split"] == "test")
        regression.append([
            label,
            f"{vi(metric['rmse']['mean'], 4)} ± {vi(metric['rmse']['std'], 4)}",
            f"{vi(metric['mae']['mean'], 4)} ± {vi(metric['mae']['std'], 4)}",
            vi(validation["rmse"], 4), vi(test["rmse"], 4),
            vi(test["mae"], 4), vi(test["r2"], 4),
        ])
        costs.append([
            label, vi(test["fit_seconds"], 2), vi(test["inference_us_per_row"], 3),
            mib(test["model_bytes"]), vi(test["peak_rss_mb"], 1),
        ])

    classification = runs["classification"]
    cls = [["Mô hình", "F1 CV (mean ± std)", "Accuracy Test",
            "F1 Test", "AUC-ROC Test"]]
    cls_cost = [["Mô hình", "Ngưỡng", "Precision Test", "Recall Test",
                 "Confusion matrix [[TN,FP],[FN,TP]]", "Fit (s)", "Inference (µs/rating)"]]
    for name, label in (("logistic", "Logistic Regression"),
                        ("random_forest", "Random Forest")):
        metric = classification["cv"][name + "_raw"]["metrics"]
        test = next(r for r in classification["rows"]
                    if r["algorithm"] == name and r["split"] == "test")
        config = classification["best_configs"][name + "_raw"]
        matrix = test["confusion_matrix"]
        cls.append([
            label,
            f"{vi(metric['f1']['mean'], 4)} ± {vi(metric['f1']['std'], 4)}",
            vi(test["accuracy"], 4), vi(test["f1"], 4), vi(test["auc_roc"], 4),
        ])
        cls_cost.append([
            label, vi(config["threshold"], 2), vi(test["precision"], 4),
            vi(test["recall"], 4),
            f"[[{vi_int(matrix[0][0])}, {vi_int(matrix[0][1])}], "
            f"[{vi_int(matrix[1][0])}, {vi_int(matrix[1][1])}]]",
            vi(test["fit_seconds"], 2), vi(test["inference_us_per_row"], 3),
        ])

    clustering = runs["clustering"]
    cluster_table = [["Thuật toán", "Số cụm", "Noise", "Silhouette",
                      "Davies–Bouldin", "Stability ARI"]]
    for name, label in (("kmeans", "K-Means"), ("dbscan", "DBSCAN")):
        row = next(r for r in clustering["rows"] if r["algorithm"] == name)
        cluster_table.append([
            label, vi_int(row["clusters"]), vi(row["noise_fraction"] * 100, 2) + "%",
            vi(row.get("silhouette"), 4) if row.get("silhouette") is not None else "N/A",
            vi(row.get("dbi"), 4) if row.get("dbi") is not None else "N/A",
            vi(row.get("stability_ari"), 4) if row.get("stability_ari") is not None else "N/A",
        ])
    rec_rows = clustering["recommendation"]["rows"]
    ranking = [["Phương pháp", "Recall@10 Validation", "NDCG@10 Validation",
                "Recall@10 Test", "NDCG@10 Test"]]
    for algorithm, label in (("popularity", "Popularity baseline"),
                             ("kmeans", "K-Means"),
                             ("dbscan", "DBSCAN")):
        val = next(r for r in rec_rows if r["algorithm"] == algorithm and r["split"] == "validation")
        test = next(r for r in rec_rows if r["algorithm"] == algorithm and r["split"] == "test")
        ranking.append([
            label, vi(val["recall_at_10"], 5), vi(val["ndcg_at_10"], 5),
            vi(test["recall_at_10"], 5), vi(test["ndcg_at_10"], 5),
        ])

    association = runs["association"]
    assoc_row = association["rows"][0]
    assoc_table = [
        ["Transactions Train", "Rating thích", "Catalog", "Frequent itemsets",
         "Luật", "Support TB", "Confidence TB", "Lift TB"],
        [vi_int(assoc_row["transactions"]), "≥ 4,0 sao",
         vi_int(association["association_data"]["candidate_movies"]),
         vi_int(assoc_row["itemsets"]), vi_int(assoc_row["rule_count"]),
         vi(assoc_row["support"], 4), vi(assoc_row["confidence"], 4),
         vi(assoc_row["lift"], 4)],
    ]
    rules = sorted(association["rules"], key=lambda r: r["lift"], reverse=True)[:5]
    rules_table = [["Vế trái X", "Vế phải Y", "Support", "Confidence", "Lift"]]
    for rule in rules:
        rules_table.append([
            " + ".join(rule["antecedents"]), " + ".join(rule["consequents"]),
            vi(rule["support"], 4), vi(rule["confidence"], 4), vi(rule["lift"], 3),
        ])

    rl = runs["reinforcement"]
    rl_table = [["Policy", "Cumulative reward Test", "Reward/action Test",
                 "Feedback quan sát", "Recall@5", "NDCG@5", "Fit (s)",
                 "Inference (µs/action)"]]
    for name, label in (("dqn", "DQN"), ("random", "Random"),
                        ("popular", "Popularity")):
        row = next(r for r in rl["rows"] if r["algorithm"] == name and r["split"] == "test")
        latency = row["inference_seconds"] / max(1, row["actual_steps"]) * 1_000_000
        rl_table.append([
            label, vi(row["cumulative_reward"], 2), vi(row["average_reward"], 5),
            vi(row["observed_feedback_fraction"] * 100, 2) + "%",
            vi(row["recall_at_horizon"], 4), vi(row["ndcg_at_horizon"], 4),
            vi(row["fit_seconds"], 2) if row["fit_seconds"] else "—",
            vi(latency, 2),
        ])
    summary = [["Mô hình", "RMSE CV (mean ± std)", "RMSE Test",
                "Fit cuối (s)", "Đánh giá triển khai"]]
    notes = {
        "linear": "Nhẹ, nhanh; sai số cao hơn SVD/NCF.",
        "xgboost": "Fit lâu nhất; không dẫn đầu lỗi rating.",
        "svd": "CV thấp nhất; inference nhanh; ứng viên cân bằng.",
        "ncf": "Được Validation chọn; Test gap lớn hơn; cần xác nhận.",
    }
    for name, label in model_labels.items():
        metric = rating["cv"][name + "_raw"]["metrics"]["rmse"]
        test = next(r for r in rating["rows"]
                    if r["algorithm"] == name and r["split"] == "test")
        summary.append([
            label, f"{vi(metric['mean'], 4)} ± {vi(metric['std'], 4)}",
            vi(test["rmse"], 4), vi(test["fit_seconds"], 1), notes[name],
        ])
    return {
        "regression": regression, "rating_cost": costs, "classification": cls,
        "classification_cost": cls_cost, "clustering": cluster_table,
        "ranking": ranking, "association": assoc_table, "rules": rules_table,
        "rl": rl_table, "rating_summary": summary,
    }


def figure(doc: Document, paths: dict[str, Path], key: str, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(paths[key]), width=Cm(PAGE_WIDTH_CM))
    cap = doc.add_paragraph(caption, "Report Figure Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_together = True
    cap.paragraph_format.first_line_indent = Cm(0)


def add_caption(doc: Document, text: str, style: str) -> None:
    p = doc.add_paragraph(text, style)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    style = "Report Heading 1" if level == 1 else "Report Heading 2"
    p = doc.add_paragraph(text, style)
    if level == 1:
        p.paragraph_format.keep_with_next = True
    return p


def add_chapter(doc: Document, text: str, first: bool = False) -> None:
    p = doc.add_paragraph(text.upper(), "Report Chapter")
    p.paragraph_format.page_break_before = not first
    p.paragraph_format.keep_with_next = True


def add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Report Body")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(text)


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    p = doc.add_paragraph(style="Report Body")
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Cm(0.8 if level == 0 else 1.3)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    p.add_run(("– " if level == 0 else "+ ") + text)


def add_equation(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Report Equation")
    p.paragraph_format.keep_together = True
    p.add_run(text)


def add_field_front(doc: Document, heading: str, instruction: str, fallback: str) -> None:
    doc.add_paragraph(heading, "Report Front")
    p = doc.add_paragraph(style="Report Field")
    add_field(p, instruction, fallback)


def add_table(doc: Document, caption: str, rows: list[list[str]]) -> None:
    add_caption(doc, caption, "Report Table Caption")
    add_table_rows(doc, rows)


def build_content(doc: Document, runs: dict, charts: dict[str, Path]) -> None:
    tables = result_tables(runs)
    # The cover pages are preserved byte-for-byte at the paragraph/table level.
    add_frontmatter(doc)
    # A new section restarts Arabic numbering at the first chapter.
    body_section = doc.add_section(WD_SECTION_START.NEW_PAGE)
    set_page_number(body_section, "decimal")
    add_chapter(doc, "CHƯƠNG 1: TỔNG QUAN", first=True)

    add_heading(doc, "1.1. ĐẶT VẤN ĐỀ")
    add_body(doc,
        "Danh mục nội dung số ngày càng lớn trong khi thời gian người xem dành để tìm một bộ phim phù hợp có hạn. "
        "Hệ gợi ý hỗ trợ thu hẹp danh sách lựa chọn bằng cách khai thác lịch sử đánh giá, đặc điểm nội dung và "
        "mẫu hành vi của nhiều người dùng. Bài toán của đồ án là dự đoán rating mà một người dùng có thể gán cho "
        "một bộ phim chưa đánh giá, sau đó dùng dự đoán hoặc sở thích tương tự để tạo danh sách Top-N.")
    add_body(doc,
        "MovieLens 25M cung cấp rating tường minh trên thang 0,5–5,0 cùng userId, movieId và timestamp. "
        "Ma trận tương tác người dùng–phim thưa hơn 99,7%; phần lớn ô không có rating. Ô trống là phản hồi "
        "chưa quan sát, không phải rating 0 hay nhãn “không thích”. Vì vậy, mô hình phải học từ các cặp đã "
        "quan sát và phép đánh giá phải bảo đảm rating ở tương lai không lọt vào đặc trưng được fit ở quá khứ [1], [2].")
    add_body(doc,
        "Một hệ thống thực tế còn đối mặt với người dùng hoặc phim ít lịch sử, thay đổi theo thời gian, "
        "mất cân bằng nhãn khi đổi rating thành thích/không thích, chi phí dự đoán và yêu cầu giải thích "
        "kết quả. Đồ án khảo sát nhiều nhóm thuật toán để chỉ ra nhóm nào trả lời câu hỏi nào; không gộp "
        "RMSE, F1, Silhouette, Lift và reward thành một thứ hạng duy nhất.")

    add_heading(doc, "1.2. LÝ DO CHỌN ĐỀ TÀI")
    add_body(doc,
        "Thứ nhất, MovieLens 25M là bộ dữ liệu công khai có mô tả nguồn, định dạng và điều kiện trích dẫn rõ "
        "ràng. Bộ dữ liệu bao gồm hàng chục triệu rating nhưng vẫn có schema gọn, thuận tiện để tổ chức "
        "một quy trình tái lập từ làm sạch đến đánh giá [1], [2].")
    add_body(doc,
        "Thứ hai, hệ gợi ý là ví dụ gần gũi để đối chiếu các giả định học máy. Hồi quy ước lượng số sao; "
        "phân loại dự đoán ngưỡng yêu thích; phân cụm mô tả các kiểu hồ sơ thể loại; Apriori tìm sự đồng "
        "xuất hiện; PCA/TruncatedSVD nén đặc trưng; còn học tăng cường mô phỏng lựa chọn tuần tự. Mỗi "
        "nhóm đòi hỏi biểu diễn dữ liệu, tiêu chí đánh giá và cách diễn giải khác nhau.")
    add_body(doc,
        "Thứ ba, đồ án cần cân nhắc chất lượng cùng chi phí triển khai. NCF có thể biểu diễn tương tác phi "
        "tuyến bằng embedding và MLP, nhưng cần thời gian huấn luyện và kiểm tra overfitting; mô hình nhân "
        "tố SVD-CF đơn giản hơn nhưng vẫn học được tương tác user–item. Đối chiếu bằng cùng giao thức giúp "
        "tránh giả định rằng mô hình phức tạp hơn luôn tốt hơn [3], [4].")

    add_heading(doc, "1.3. MỤC TIÊU NGHIÊN CỨU")
    add_body(doc,
        "Mục tiêu tổng quát là xây dựng và đánh giá một ứng dụng dự đoán rating phim trên MovieLens 25M, "
        "đồng thời cung cấp các thực nghiệm để phân tích sở thích và minh họa gợi ý cá nhân hóa trên giao diện web.")
    for item in [
        "Rà soát và chuẩn hóa dữ liệu MovieLens, giữ nguyên schema và lưu dấu vết dữ liệu.",
        "Xây dựng tập Train, Validation và Test theo thứ tự thời gian; tránh dùng dữ liệu tương lai khi fit thống kê hoặc mô hình.",
        "Cài đặt các thuật toán theo sáu nhóm nhiệm vụ, dùng baseline và metric phù hợp với đầu ra.",
        "So sánh bốn mô hình dự đoán rating theo RMSE/MAE, độ ổn định qua fold, thời gian huấn luyện, latency và kích thước model.",
        "Tích hợp backend FastAPI với giao diện React/TypeScript để tìm phim, quản lý lịch sử và xem gợi ý Top-N.",
        "Ghi rõ những hạng mục chưa có artifact hoặc chưa được chạy, trong đó phép so sánh XGBoost với XGBoost + PCA/TruncatedSVD được dành lại để bổ sung.",
    ]:
        add_bullet(doc, item)

    add_heading(doc, "1.4. ĐỐI TƯỢNG VÀ QUY MÔ NGHIÊN CỨU")
    add_body(doc,
        "Đối tượng chính là các lượt rating liên kết giữa người dùng và phim, gồm userId, movieId, rating và "
        "timestamp. Metadata genres từ movies.csv dùng để tạo đặc trưng sở thích. Tập dữ liệu có 162.541 user, "
        "62.423 movie metadata và 25.000.095 rating; rating trải từ 0,5 đến 5,0. Các lần đánh giá được lưu "
        "trong giai đoạn 1995–2019, nên kết luận về độ phổ biến chỉ phản ánh giai đoạn lịch sử này [1].")
    add_body(doc,
        "Phạm vi mô hình gồm Linear Regression, XGBoost Regressor, biased matrix factorization thường gọi "
        "Funk SVD, NCF, Logistic Regression, Random Forest, K-Means, DBSCAN, Apriori, PCA, TruncatedSVD "
        "và DQN/Q-Learning. NCF và DQN là nhánh mạng nơ-ron; DQN được ghi riêng với Q-Learning dạng bảng.")
    add_body(doc,
        "Đồ án đánh giá offline trên dữ liệu rating đã quan sát, không thực hiện A/B test hoặc suy luận nhân "
        "khẩu học. Dataset không có tuổi, giới tính, cốt truyện, diễn viên hay poster; báo cáo không suy diễn "
        "những thuộc tính này từ mã người dùng hoặc tên phim.")

    add_heading(doc, "1.5. NỘI DUNG VÀ KỸ THUẬT NGHIÊN CỨU")
    add_body(doc,
        "Nghiên cứu kết hợp khảo cứu bài báo nền tảng và tài liệu chính thức với kiểm tra mã nguồn, cấu hình "
        "và artifact của dự án. Dữ liệu gốc được kiểm tra trước, bản làm sạch được lưu riêng; bước chuẩn bị "
        "tập tạo các file Parquet và manifest để lần chạy có thể truy nguyên.")
    add_body(doc,
        "Các mô hình hồi quy và phân loại dùng đặc trưng theo batch từ rating quá khứ, hồ sơ user–genre và "
        "thống kê chỉ fit trên phần Train. SVD-CF/NCF dùng biểu diễn user–item tiềm ẩn; clustering dùng hồ "
        "sơ thể loại; Apriori dùng transaction phim được thích; Q-learning/DQN được đánh giá trong môi trường "
        "offline với reward được chuẩn hóa từ rating quan sát.")
    add_body(doc,
        "Ứng dụng sử dụng Python 3.12, scikit-learn, XGBoost, PyTorch, FastAPI và React/TypeScript. Huấn "
        "luyện nặng chạy trong worker/CLI thay vì giữ một request giao diện trong lúc fit; API phục vụ dự "
        "đoán, lịch sử và trạng thái thực nghiệm. Các chi tiết thực thi và giới hạn theo từng protocol được "
        "trình bày trong Chương 3.")

    add_chapter(doc, "CHƯƠNG 2: CƠ SỞ LÝ THUYẾT")
    add_heading(doc, "2.1. NHÓM THUẬT TOÁN HỒI QUY")
    add_heading(doc, "2.1.1. Linear Regression", 2)
    add_body(doc,
        "Linear Regression mô hình hóa rating bằng tổ hợp tuyến tính của vector đặc trưng x: r̂ = β₀ + "
        "βᵀx. Ordinary Least Squares chọn hệ số β để giảm tổng bình phương sai số trên các rating Train. "
        "Nếu đặc trưng tương quan hoặc số chiều lớn, Ridge có thể thêm phạt L2; tuy nhiên lần chạy được "
        "báo cáo ở đây dùng lớp LinearRating và cấu hình tuyến tính mặc định, không gọi kết quả ấy là Ridge [6].")
    add_equation(doc, "J(β) = Σ(i∈Train) (rᵢ − β₀ − βᵀxᵢ)²")
    add_body(doc,
        "Ưu điểm là baseline dễ kiểm tra, chi phí suy luận thấp và có thể kết hợp nhiều đặc trưng. Khả năng "
        "học tương tác phi tuyến phụ thuộc cách xây đặc trưng; hệ số tuyến tính không tự mô tả mọi tương "
        "tác user–movie. Trong project, feature builder tạo thống kê từ Train, tỷ lệ và độ phân tán rating, "
        "lịch sử gần đây cùng các tương tác hồ sơ user–genre–movie.")
    add_heading(doc, "2.1.2. XGBoost Regressor", 2)
    add_body(doc,
        "XGBoost xây một tổng cộng dồn các cây hồi quy. Ở mỗi vòng, cây mới hiệu chỉnh sai số của mô hình "
        "hiện tại; hàm mục tiêu kết hợp loss với regularization cho độ phức tạp của cây. Learning rate, "
        "số cây và max depth kiểm soát tốc độ cập nhật, sức biểu diễn và chi phí fit [5].")
    add_equation(doc, "r̂ᵢ = Σₜ fₜ(xᵢ),     Obj = Σᵢ ℓ(rᵢ, r̂ᵢ) + Σₜ Ω(fₜ)")
    add_body(doc,
        "Cây có thể học tương tác và ranh giới phi tuyến mà không yêu cầu chuẩn hóa tuyến tính như Logistic "
        "Regression. Đổi lại, hàng trăm cây có thể tốn thời gian hơn ở cả fit và inference; độ sâu lớn có "
        "thể làm mô hình khớp quá mức. Bài báo XGBoost nhấn mạnh các thủ thuật sparsity-aware, quantile "
        "sketch và tối ưu bộ nhớ, nhưng lợi ích của implementation không thay thế việc đo trên đúng workload [5].")
    add_heading(doc, "2.1.3. SVD Collaborative Filtering", 2)
    add_body(doc,
        "Trong gợi ý rating, tên SVD thường chỉ mô hình nhân tố tiềm ẩn học từ các rating đã quan sát, không "
        "phải phép phân rã chính xác một ma trận user×movie đã điền đầy. Dự đoán biased matrix factorization "
        "có dạng r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀqᵢ; μ là trung bình Train, bᵤ/bᵢ là bias và pᵤ/qᵢ là vector "
        "latent của user/phim [3].")
    add_equation(doc, "J = Σ((u,i,r)∈ΩTrain) (r − μ − bᵤ − bᵢ − pᵤᵀqᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖² + bᵤ² + bᵢ²)")
    add_body(doc,
        "Mã nguồn khởi tạo factor, bias và tối ưu SGD chỉ trên ΩTrain. Rating thiếu không bị thay bằng 0. "
        "Trong lần chạy hiện tại có 32 factors, learning rate 0,005, regularization 0,02 và 15 epoch. "
        "SVD-CF khai thác cộng tác tốt khi có lịch sử; user/item chưa gặp cần bias baseline hoặc fallback "
        "vì latent factor riêng chưa được học.")
    add_heading(doc, "2.1.4. Neural Collaborative Filtering", 2)
    add_body(doc,
        "NCF thay tích vô hướng cố định bằng một mạng có thể học tương tác từ embedding user và item. "
        "Bài báo gốc dùng embedding và MLP để mô hình hóa hàm tương tác [4]. Implementation đồ án điều "
        "chỉnh NCF cho rating tường minh: một bias baseline dự đoán mức nền, còn mạng nhận embedding user, "
        "embedding item và tích từng phần tử của hai embedding để dự đoán phần dư.")
    add_equation(doc, "r̂ᵤᵢ = μ + bᵤ + bᵢ + MLP([pᵤ, qᵢ, pᵤ ⊙ qᵢ])")
    add_body(doc,
        "Mô hình dùng MSE, AdamW, dropout và early stopping theo Validation. Bản chạy có 32 chiều embedding, "
        "hidden size 32, dropout 0,1, learning rate 0,003, batch 2.048 và patience 3. GPU CUDA được dùng "
        "nếu khả dụng; một minibatch được chuyển thiết bị trong quá trình học. NCF linh hoạt hơn nhưng "
        "cần theo dõi đường loss và chi phí so với factorization.")
    add_heading(doc, "2.1.5. So sánh các mô hình hồi quy", 2)
    add_body(doc,
        "Bốn mô hình cùng dự đoán số sao nhưng có inductive bias khác nhau: tuyến tính phụ thuộc feature "
        "được tạo; boosting ghép các cây; SVD-CF học factor thấp chiều trên tương tác; NCF học hàm tương "
        "tác phi tuyến bằng mạng. Do đó, so sánh phải dùng cùng split và cùng mẫu, đồng thời báo cáo RMSE "
        "và MAE. Các mô hình latent cũng không tự giải quyết cold-start tuyệt đối.")

    add_heading(doc, "2.2. NHÓM THUẬT TOÁN PHÂN LOẠI")
    add_heading(doc, "2.2.1. Logistic Regression", 2)
    add_body(doc,
        "Logistic Regression biến tổ hợp tuyến tính z = β₀ + βᵀx thành xác suất bằng sigmoid "
        "p(y=1|x)=1/(1+e⁻ᶻ). Mô hình được học bằng log-loss có regularization; C điều khiển cường độ "
        "phạt theo quy ước của scikit-learn. Với hai lớp, xác suất được đổi thành nhãn bằng ngưỡng được "
        "chọn trên Validation, thay vì mặc định luôn dùng 0,5 [6].")
    add_heading(doc, "2.2.2. Random Forest", 2)
    add_body(doc,
        "Random Forest huấn luyện nhiều cây trên các bootstrap sample và các tập con đặc trưng; dự đoán "
        "phân loại lấy trung bình xác suất hoặc bỏ phiếu. Tính ngẫu nhiên giữa cây giúp giảm tương quan "
        "giữa từng estimator và thường làm mô hình vững hơn một cây đơn [7]. Số cây, độ sâu, kích thước "
        "leaf và số hàng bootstrap ảnh hưởng trực tiếp đến bộ nhớ, độ chính xác và latency.")
    add_body(doc,
        "Trong lần chạy này nhãn Thích được định nghĩa là rating ≥3,5. Random Forest dùng 100 cây, depth "
        "8, min_samples_leaf=5 và tối đa 250.000 lần lấy mẫu cho mỗi cây. Giới hạn bootstrap giảm chi phí "
        "fit, nhưng cần được ghi cùng kết quả vì mô hình không huấn luyện mỗi cây trên toàn bộ 17,4 triệu "
        "lượt Train.")
    add_heading(doc, "2.2.3. Metric phân loại", 2)
    add_body(doc,
        "Accuracy là tỷ lệ dự đoán đúng; Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 là trung bình "
        "điều hòa giữa Precision và Recall. AUC-ROC đánh giá thứ tự xác suất trên nhiều ngưỡng, không "
        "chỉ tại ngưỡng ra nhãn. Khi lớp Thích chiếm đa số, cần so Accuracy với baseline lớp đa số và đọc "
        "F1/Precision/Recall cùng ma trận nhầm lẫn [15].")

    add_heading(doc, "2.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    add_heading(doc, "2.3.1. K-Means", 2)
    add_body(doc,
        "K-Means tìm K tâm cụm sao cho tổng khoảng cách bình phương từ điểm đến tâm gần nhất nhỏ nhất. "
        "Thuật toán phù hợp với cụm có dạng tương đối lồi và nhạy với thang đo, cách khởi tạo và K [8]. "
        "Đồ án biểu diễn user bằng hồ sơ rating tương đối theo thể loại và tần suất xem thể loại; hồ sơ "
        "được chuẩn hóa trước khi tìm cụm.")
    add_equation(doc, "min(C₁,…,Cₖ) Σⱼ Σ(x∈Cⱼ) ‖x − μⱼ‖²")
    add_heading(doc, "2.3.2. DBSCAN", 2)
    add_body(doc,
        "DBSCAN định nghĩa cụm qua các vùng có mật độ cao, dùng eps để xác định lân cận và min_samples "
        "để xác định điểm lõi. Điểm nằm ngoài các vùng đủ mật độ được đánh dấu noise; K không cần khai "
        "báo trước [9]. DBSCAN có thể phát hiện dạng cụm phi cầu nhưng rất nhạy với thang đo và tham số; "
        "nếu dữ liệu có mật độ khác nhau, nhiều điểm có thể thành noise hoặc thành một cụm duy nhất.")
    add_heading(doc, "2.3.3. Chỉ số đánh giá cụm", 2)
    add_body(doc,
        "Silhouette so khoảng cách trung bình tới cụm đang thuộc với khoảng cách tới cụm gần nhất; giá trị "
        "gần 1 cho thấy tách biệt hơn, gần 0 là chồng lấn, âm là dấu hiệu gán chưa phù hợp. Davies–Bouldin "
        "đo tỷ lệ phân tán trong cụm so với khoảng cách giữa các tâm; thấp hơn thường tốt hơn khi so cùng "
        "biểu diễn. Chỉ số cần báo cùng coverage/noise vì DBSCAN có thể trông tốt nếu bỏ phần lớn điểm.")

    add_heading(doc, "2.4. KHAI PHÁ LUẬT KẾT HỢP APRIORI")
    add_body(doc,
        "Apriori tìm frequent itemset trong các transaction rồi sinh luật X→Y. Tính chất Apriori cho phép "
        "loại một tập ứng viên nếu tập con của nó đã không đạt minimum support: một tập chứa itemset không "
        "phổ biến không thể phổ biến hơn itemset con đó [10]. Trong dự án, mỗi user là một transaction gồm "
        "các phim Train có rating ít nhất 4,0.")
    add_equation(doc, "Support(X→Y)=P(X∪Y);  Confidence(X→Y)=P(Y|X);  Lift(X→Y)=Confidence(X→Y)/Support(Y)")
    add_body(doc,
        "Support đo tỷ lệ giao dịch chứa cả X và Y; Confidence đo tỷ lệ giao dịch chứa Y trong nhóm đã "
        "chứa X; Lift so Confidence với tần suất nền của Y. Lift lớn hơn 1 biểu thị đồng xuất hiện dương "
        "trong mẫu, nhưng không chứng minh rằng X gây ra Y. Cần đọc Lift cùng Support và kiểm tra độ ổn "
        "định trên dữ liệu giữ lại.")

    add_heading(doc, "2.5. NHÓM GIẢM CHIỀU DỮ LIỆU")
    add_heading(doc, "2.5.1. Principal Component Analysis", 2)
    add_body(doc,
        "PCA trừ trung bình từng feature rồi chiếu dữ liệu lên các vector riêng trực giao sắp theo phương "
        "sai giảm dần. Giữ k thành phần đầu tạo một biểu diễn thấp chiều và cho biết tổng tỷ lệ phương sai "
        "được giữ lại. PCA có thể giảm nhiễu hoặc chi phí downstream, nhưng variance được giữ không đồng "
        "nghĩa với thông tin dự đoán rating được giữ [11].")
    add_heading(doc, "2.5.2. TruncatedSVD", 2)
    add_body(doc,
        "TruncatedSVD chỉ tính một số singular vector lớn nhất và không center dữ liệu trước khi phân rã. "
        "Đặc tính không center giúp thao tác với ma trận sparse; phép chiếu đầu ra vẫn là biểu diễn dense "
        "ở k chiều. Scikit-learn phân biệt rõ PCA có center với TruncatedSVD không center [12].")
    add_heading(doc, "2.5.3. Phân biệt giảm chiều và SVD-CF", 2)
    add_body(doc,
        "TruncatedSVD/PCA giảm chiều feature matrix để tái tạo hoặc tạo đầu vào downstream. SVD-CF học "
        "nhân tố user/movie trực tiếp từ các rating quan sát để dự đoán rating. Hai loại SVD giải hai mục "
        "tiêu khác nhau; kết quả cần nằm ở đúng nhóm metric. Phép so sánh XGBoost, XGBoost+PCA và "
        "XGBoost+TruncatedSVD được yêu cầu trong đề tài nhưng chưa có run kết quả tương ứng trong bộ "
        "artifact được dùng cho báo cáo.")

    add_heading(doc, "2.6. Q-LEARNING VÀ GỢI Ý TUẦN TỰ")
    add_heading(doc, "2.6.1. MDP và cập nhật giá trị Q", 2)
    add_body(doc,
        "Một bài toán học tăng cường mô tả bằng trạng thái s, hành động a, reward r, trạng thái kế tiếp "
        "s′ và discount γ. Q-learning cập nhật giá trị hành động hướng về reward quan sát cộng giá trị "
        "tốt nhất tương lai. Với bảng Q rời rạc, điều kiện hội tụ cần mỗi cặp state–action được thăm đủ "
        "nhiều lần [13].")
    add_equation(doc, "Q(s,a) ← Q(s,a) + α[r + γ maxₐ′ Q(s′,a′) − Q(s,a)]")
    add_heading(doc, "2.6.2. DQN trong thực nghiệm hiện tại", 2)
    add_body(doc,
        "Khi không gian state/action lớn, DQN thay bảng Q bằng mạng nơ-ron xấp xỉ Q(s,a), thường kết hợp "
        "replay buffer và target network để ổn định target [14]. Run gần nhất trong project là DQN, không "
        "phải Q-table: state gồm profile thể loại, độ đầy đủ, action mask, số bước còn lại, reward và cờ "
        "feedback ở bước trước; action chọn phim chưa xem trong catalog phổ biến Train.")
    add_body(doc,
        "Reward quan sát được chuẩn hóa thành (rating−2,75)/2,25 trong khoảng [−1,1]; phim không có "
        "feedback nhận reward 0 kèm cờ chưa quan sát, không bị xem là rating xấu. Đánh giá offline vẫn "
        "bị thiên lệch chọn mẫu vì rating chỉ quan sát với các phim user đã chọn; không có propensity/IPS "
        "để ước lượng chính sách counterfactual không thiên lệch.")

    add_chapter(doc, "CHƯƠNG 3: DỮ LIỆU VÀ THIẾT KẾ THỰC NGHIỆM")
    add_heading(doc, "3.1. MÔ TẢ BỘ DỮ LIỆU")
    add_body(doc,
        "Nguồn là MovieLens 25M do GroupLens/University of Minnesota phát hành. Bản làm sạch giữ nguyên "
        "25.000.095 dòng ratings.csv, 62.423 dòng movies.csv và 62.423 dòng links.csv; tags.csv còn "
        "1.093.332 nhãn sau khi chuẩn hóa Unicode, loại 26 nhãn rỗng/placeholder và 2 bản ghi trùng. "
        "Các rating hợp lệ không bị xóa hay lấy mẫu trong bước làm sạch [1].")
    add_table(doc, "Bảng 3.1: Quy mô và đặc trưng của MovieLens 25M",
              [["Chỉ tiêu", "Giá trị", "Diễn giải"],
               ["Người dùng có rating", "162.541", "Trong ratings.csv gốc"],
               ["Phim trong metadata", "62.423", "movies.csv"],
               ["Phim có ít nhất một rating", "59.047", "3.376 phim chưa có rating"],
               ["Tổng rating", "25.000.095", "Rating tường minh"],
               ["Rating trung bình", "3,53385 / 5", "Thang điểm 0,5–5,0; bước 0,5"],
               ["Thể loại", "19", "Một phim có thể thuộc nhiều thể loại"],
               ["Mật độ ma trận user × movie", "0,2464%", "Độ thưa xấp xỉ 99,7536%"],
               ["Thời gian rating", "1995–2019", "Timestamp UTC trong CSV"],
               ["Tag cộng đồng", "1.093.332", "Hiển thị/tham khảo; không phải feature rating chính"]])
    add_body(doc,
        "Ma trận user–movie có 162.541 × 62.423 ô tiềm năng nhưng chỉ 25.000.095 tương tác, do đó không "
        "được tạo thành ma trận dense. Pipeline đọc theo chunk, giữ bảng rating quan sát và tạo feature "
        "theo batch. Dataset không có poster hay mô tả nội dung; website dùng metadata và liên kết IMDb/TMDB "
        "để hỗ trợ duyệt danh mục.")
    add_table(doc, "Bảng 3.2: Phân bố rating trong toàn bộ dataset",
              [["Rating", "Số lượt"], ["0,5", "393.068"], ["1,0", "776.815"],
               ["1,5", "399.490"], ["2,0", "1.640.868"], ["2,5", "1.262.797"],
               ["3,0", "4.896.928"], ["3,5", "3.177.318"], ["4,0", "6.639.798"],
               ["4,5", "2.200.539"], ["5,0", "3.612.474"]])
    figure(doc, charts, "ratings", "Hình 3.1: Phân bố rating MovieLens 25M theo số lượt")
    add_body(doc,
        "Điểm 4,0 xuất hiện nhiều nhất. Trung bình toàn dataset dùng để mô tả, không đưa vào đặc trưng "
        "Train của từng mô hình. Thống kê rating/độ phổ biến để hiển thị trên website cũng tách khỏi "
        "thống kê Train dùng cho RMSE/MAE.")

    add_heading(doc, "3.2. XÁC ĐỊNH BÀI TOÁN THEO TỪNG NHÓM THUẬT TOÁN")
    add_body(doc,
        "Bài toán hồi quy nhận (user, movie, rating history) và trả về số thực trên thang rating. Bài toán "
        "phân loại giữ đặc trưng nhưng chuyển target thành y=1 nếu rating≥3,5, y=0 nếu rating<3,5. Hai "
        "bài toán dùng cùng split nhưng trả lời hai câu hỏi khác nhau; kết quả phân loại không thay thế "
        "sai số rating.")
    add_body(doc,
        "Bài toán phân cụm tạo vector sở thích/frequency theo thể loại cho từng user, rồi đánh giá cả "
        "cấu trúc hình học lẫn ranking tương lai. Apriori nhận transaction user–phim được thích trong "
        "Train; kết quả luật mô tả liên kết đồng xuất hiện. Giảm chiều có hai mục tiêu: chẩn đoán explained "
        "variance/reconstruction và đánh giá mô hình downstream trên cùng split.")
    add_body(doc,
        "Bài toán tăng cường chọn hành động phim theo state profile, catalog, history mask và feedback. "
        "Q-learning/DQN được so với Random và Popularity trong cùng episode/candidate pool. Kết quả chỉ "
        "là mô phỏng trên rating đã quan sát; không được diễn giải như hiệu quả trực tuyến.")
    add_body(doc,
        "Đề tài yêu cầu so sánh XGBoost gốc với XGBoost + PCA và XGBoost + TruncatedSVD. Phần Chương 4 "
        "được để trống vì chưa có artifact cho đúng phép so sánh đó. Các run PCA/TruncatedSVD ở giao diện "
        "thuộc phép giảm chiều/profile khác không được tráo vào làm kết quả XGBoost downstream.")

    add_heading(doc, "3.3. CẤU HÌNH VÀ CÁCH ÁP DỤNG CÁC THUẬT TOÁN")
    add_heading(doc, "3.3.1. Làm sạch, lọc và chia dữ liệu", 2)
    add_body(doc,
        "Bản làm sạch kiểm tra ID, schema, rating và timestamp; chuẩn hóa Unicode; không ghi đè dữ liệu "
        "gốc. Bước chia per-user temporal yêu cầu tối thiểu 10 rating và không cắt một nhóm rating có "
        "timestamp bằng nhau qua hai tập. Vì thế, các run rating/classification/clustering/RL dùng "
        "24.879.426 rating của 159.356 user đủ điều kiện; loại 3.185 user và 120.669 rating do lịch sử "
        "quá ngắn hoặc ranh giới thời gian không tách được.")
    add_body(doc,
        "Các run này chia xấp xỉ 70/15/15 bên trong từng user. Mỗi user dùng lịch sử sớm hơn ở Train, đoạn "
        "kế tiếp ở Validation và đoạn muộn nhất ở Test. Cách chia giữ người dùng hiện diện xuyên split, "
        "nên đánh giá là dự đoán tương lai cho user có lịch sử; nó không đo cold-start user mới.")
    add_table(doc, "Bảng 3.3: Quy mô split trong các run per-user temporal",
              [["Tập", "Rating", "User", "Movie có rating"],
               ["Train", "17.413.070", "159.356", "47.570"],
               ["Validation", "3.722.738", "159.356", "41.408"],
               ["Test", "3.743.618", "159.356", "45.806"],
               ["Tổng dữ liệu đủ điều kiện", "24.879.426", "159.356", "—"]])
    figure(doc, charts, "split", "Hình 3.2: Giao thức chia dữ liệu theo thời gian trong từng user")
    add_body(doc,
        "Apriori dùng protocol khác: chia toàn cục theo timestamp trên 25.000.095 rating đã làm sạch; "
        "Train có 17.500.066 rating và 122.029 user. Clustering fit hồ sơ Train của toàn bộ 159.356 user "
        "đủ điều kiện, còn Validation/Test ranking lấy cohort tối đa 2.000 user. Vì mẫu số và thời gian "
        "đánh giá khác nhau, không so trị số trực tiếp giữa các nhóm.")

    add_heading(doc, "3.3.2. Cấu hình thực nghiệm", 2)
    add_body(doc,
        "Seed các run được chọn là 42. Các kết quả ở Chương 4 trích từ artifact JSON theo run ID tại "
        "project/artifacts/experiment-lab, không lấy số của các run IID cũ hoặc các lần chạy mẫu nhỏ. "
        "Thực nghiệm rating/classification có 3 expanding-time folds trên Train; CV chỉ đánh giá cấu "
        "hình được giữ lại. Validation chọn cấu hình/ngưỡng/epoch, sau đó refit Train+Validation, còn "
        "Test chỉ báo cáo cuối.")
    add_table(doc, "Bảng 3.4: Tham số chính của các mô hình rating",
              [["Mô hình", "Cấu hình ghi trong artifact", "Ghi chú"],
               ["Linear Regression", "Cấu hình mặc định", "LinearRating; OLS, không gọi là Ridge"],
               ["XGBoost Regressor", "150 cây; depth 6; learning rate 0,05; 4 luồng",
                "Seed 42; tuning flag tắt"],
               ["SVD-CF", "32 factors; lr 0,005; λ=0,02; 15 epoch",
                "Biased factorization, SGD trên rating quan sát"],
               ["NCF", "32 factors; hidden 32; lr 0,003; dropout 0,1; batch 2.048",
                "Tối đa 15 epoch; patience 3; epoch tốt nhất = 5"]])
    add_table(doc, "Bảng 3.5: Tham số chính của các nhóm còn lại",
              [["Nhóm", "Cấu hình/điều kiện được ghi nhận"],
               ["Phân loại", "Logistic C=1, max_iter=600; RF 100 cây, depth 8, leaf 5, bootstrap tối đa 250.000 hàng/cây"],
               ["Phân cụm", "K-Means K=2; DBSCAN eps=3,284, min_samples=10; coverage tối thiểu 80%"],
               ["Apriori", "rating thích ≥4,0; min_support=0,08; min_confidence=0,30; tối đa 80 phim, itemset dài 2"],
               ["DQN", "γ=0,9; lr=0,001; epsilon 0,9→0,05; horizon 5; catalog tối đa 30 phim"],
               ["DQN tối ưu", "MLP [64,64]; replay 50.000; batch 128; target update mỗi 250 bước"]])
    add_body(doc,
        "Trong artifact rating/classification, tuning được tắt; không có bằng chứng rằng GridSearchCV, "
        "RandomizedSearchCV hoặc Optuna đã chạy trong các run này. NCF có early stopping và lựa chọn epoch "
        "trên Validation, nhưng điều đó không đồng nghĩa một lượt Optuna search. README cũ có mô tả 5-fold, "
        "song run được dùng ở đây ghi rõ cv_folds=3; báo cáo ưu tiên provenance của kết quả đã chạy.")

    add_heading(doc, "3.3.3. Đặc trưng và quy tắc fit", 2)
    add_body(doc,
        "Feature builder tính đặc trưng theo batch từ rating lịch sử: trung bình/độ phân tán, tỷ lệ rating "
        "cao, sở thích thể loại và tương tác user–movie–genre; lịch sử gần đây có giới hạn. Với mẫu Train, "
        "label của chính hàng đang huấn luyện bị loại khỏi các thống kê tương ứng. Scaler, profile mean, "
        "reducer hoặc mô hình chỉ fit trên Train/fold được phép.")
    add_body(doc,
        "Clustering tạo profile tương đối bằng cách trừ trung bình rating của user khỏi rating theo genre, "
        "sau đó làm trơn bằng shrinkage=10; tần suất xem từng genre được làm trơn theo prior và ghép vào "
        "vector. Những genre không xuất hiện nhận preference tương đối bằng 0. Đơn vị phân cụm là user, "
        "không phải phim.")
    add_body(doc,
        "Apriori giới hạn candidate catalog ở 80 phim phổ biến theo số rating thích trong Train để giữ "
        "transaction boolean ở kích thước có thể xử lý. Tất cả user Train vẫn nằm trong mẫu số support, "
        "kể cả 5.435 transaction rỗng sau giới hạn catalog. Kết quả support phải được đọc theo định nghĩa "
        "mẫu số này.")

    add_heading(doc, "3.3.4. Môi trường và kiến trúc ứng dụng", 2)
    add_body(doc,
        "Runtime của run rating ghi Windows 11, Python 3.12.10, 16 logical CPU, RAM 15,7 GiB, NVIDIA RTX "
        "3050 Laptop GPU, PyTorch CUDA 13.2, scikit-learn 1.9.1, XGBoost 3.4.1, NumPy 2.2.6 và pandas "
        "2.3.3. XGBoost dùng 4 luồng. Thời gian là phép đo perf_counter trên máy cá nhân, phụ thuộc tải "
        "hệ thống; inference được tính trên batch lớn nên không phải latency của một request đơn.")
    figure(doc, charts, "architecture", "Hình 3.3: Luồng dữ liệu và các tầng triển khai")
    add_body(doc,
        "Backend FastAPI tách API, dịch vụ mô hình, xử lý dữ liệu và lưu artifact. Frontend React/TypeScript "
        "cho phép tìm phim, xem chi tiết, lưu watch history, chấm điểm và mở trang thực nghiệm. Tài khoản "
        "ứng dụng lưu bằng SQLite; artifact mô hình và dữ liệu Parquet ở thư mục riêng. Hệ thống không tạo "
        "rating giả khi chưa có model; fallback phổ biến/nội dung cần được ghi nhãn đúng.")

    add_heading(doc, "3.4. PHƯƠNG PHÁP ĐÁNH GIÁ")
    add_heading(doc, "3.4.1. Dự đoán rating", 2)
    add_equation(doc, "MAE = (1/n)Σ|rᵢ−r̂ᵢ|;   RMSE = √[(1/n)Σ(rᵢ−r̂ᵢ)²]")
    add_body(doc,
        "MAE biểu diễn sai số tuyệt đối trung bình theo đơn vị sao; RMSE phạt lỗi lớn mạnh hơn. R² so mô "
        "hình với dự đoán trung bình trên tập đang đánh giá và có thể âm. CV báo mean ± sample standard "
        "deviation trên 3 fold theo thời gian. Thời gian fit cuối gồm xử lý đặc trưng; inference_us_per_row "
        "là thời gian batch Test chia số hàng; model_bytes là kích thước pickle được ghi cùng artifact.")
    add_heading(doc, "3.4.2. Phân loại", 2)
    add_body(doc,
        "Báo Accuracy, F1 lớp Thích, Precision, Recall, AUC-ROC từ xác suất, ngưỡng đã chọn và ma trận "
        "nhầm lẫn theo [[TN,FP],[FN,TP]]. CV có mean ± std; Validation chọn ngưỡng và Test chỉ đánh giá "
        "sau khi khóa. Với nhóm này, accuracy baseline xấp xỉ tỷ lệ lớp đa số, nên chỉ số F1 và ma trận "
        "nhầm lẫn là cần thiết [15].")
    add_heading(doc, "3.4.3. Phân cụm, luật và ranking", 2)
    add_body(doc,
        "Silhouette cao hơn và Davies–Bouldin thấp hơn thường cho thấy cụm tách biệt hơn, nhưng chỉ có "
        "ý nghĩa khi báo số cụm và noise. NDCG@10/Recall@10 đo gợi ý phim được rating thích ở đoạn tương "
        "lai trong candidate catalog; các run clustering dùng 2.000 user cohort. Apriori báo support, "
        "confidence, lift và execution time; nếu chỉ có luật Train thì không suy diễn rằng luật đã tổng "
        "quát hóa sang tương lai.")
    add_heading(doc, "3.4.4. Học tăng cường và giới hạn đo", 2)
    add_body(doc,
        "RL báo cumulative reward, reward trên mỗi action thực tế, tỷ lệ action có feedback, coverage, "
        "Recall/NDCG trong catalog và thời gian. Các policy dùng cùng episode IDs, mask và catalog. So "
        "sánh paired bootstrap theo user được ghi cùng khoảng tin cậy. Reward thiếu feedback bằng 0 theo "
        "môi trường, nhưng chỉ báo quan sát được mới được tính là feedback; coverage thấp giới hạn ý nghĩa.")

    add_chapter(doc, "CHƯƠNG 4: THỰC NGHIỆM VÀ SO SÁNH")
    add_body(doc,
        "Các bảng Chương 4 lấy từ năm run hoàn tất ngày 26/11/2026 trong artifacts của project. Rating, "
        "classification, clustering và RL dùng sample fingerprint bbca2da2…; Apriori dùng protocol global "
        "temporal với fingerprint c16c86be…. Không lấy số từ tài liệu run IID 24.319 rating hoặc bản chạy "
        "sample 80 user. Bảng điểm và thời gian là kết quả của đúng artifact ghi ở Phụ lục A.")

    add_heading(doc, "4.1. NHÓM THUẬT TOÁN HỒI QUY")
    add_body(doc,
        "Bảng 4.1 so sánh bốn mô hình trên cùng cohort per-user temporal. Validation chọn NCF với RMSE "
        "0,8046; CV sử dụng cấu hình được chọn cho mỗi thuật toán trên ba fold expanding-time. Test được "
        "mở sau refit Train+Validation và không được sử dụng để chọn cấu hình.")
    add_table(doc, "Bảng 4.1: Sai số dự đoán rating và độ ổn định qua CV", tables["regression"])
    add_table(doc, "Bảng 4.2: Chi phí huấn luyện, inference và kích thước model", tables["rating_cost"])
    figure(doc, charts, "regression", "Hình 4.1: RMSE CV mean ± std và RMSE Test của bốn mô hình (run 20261126T015401)")
    add_body(doc,
        "SVD-CF có RMSE CV thấp nhất 0,8504 ± 0,0339; NCF đạt 0,8546 ± 0,0433. Khoảng cách trung bình "
        "khoảng 0,0042 sao nhỏ hơn độ lệch chuẩn của từng nhóm fold, vì vậy ba fold không đủ để khẳng "
        "định khác biệt ổn định. Linear và XGBoost có CV RMSE lần lượt 0,8864 và 0,8808; các giá trị "
        "này cao hơn nhưng XGBoost chưa đạt lợi thế so với mô hình factorization ở cấu hình hiện tại.")
    add_body(doc,
        "Validation chọn NCF sít sao hơn SVD: 0,8046 so với 0,8063. Sau refit, Test RMSE/MAE của SVD là "
        "0,8088/0,6078, còn NCF là 0,8424/0,6340. Đây là chênh lệch giữa các tập cần báo trung thực; "
        "không được dùng Test để đổi lại lựa chọn đã khóa. Đường NCF cho thấy Validation MSE tốt nhất "
        "xấp xỉ epoch 5, sau đó tăng nhẹ trong khi Train MSE tiếp tục giảm.")
    figure(doc, charts, "ncf", "Hình 4.2: Learning curve NCF; Train MSE lấy mẫu 50.000 hàng, Validation MSE = RMSE²")
    add_body(doc,
        "NCF dùng 5 epoch tốt nhất sau early stopping; tới epoch 8 Validation RMSE đã tăng từ 0,8046 "
        "lên khoảng 0,8087 trong khi train loss tiếp tục đi xuống, là tín hiệu overfit nhẹ. Train curve "
        "là ước lượng trên 50.000 hàng do artifact ghi nhận; vì vậy không diễn giải như MSE toàn bộ Train.")
    add_body(doc,
        "Về chi phí, Linear fit 58,59 giây; SVD-CF 134,69 giây; NCF 390,89 giây; XGBoost 593,50 giây. "
        "SVD có inference batch nhanh nhất trong bốn mô hình (0,449 µs/rating), NCF 0,725 µs/rating, "
        "Linear 0,771 và XGBoost 1,615. XGBoost không phải lựa chọn tốt nhất về sai số hoặc thời gian "
        "trong lần chạy này; tốc độ có thể khác nếu phần cứng, số luồng hoặc tham số đổi.")

    add_heading(doc, "4.2. NHÓM THUẬT TOÁN PHÂN LOẠI")
    add_table(doc, "Bảng 4.3: Kết quả Logistic Regression và Random Forest", tables["classification"])
    add_table(doc, "Bảng 4.4: Ngưỡng, lỗi phân lớp và chi phí trên Test", tables["classification_cost"])
    figure(doc, charts, "classification", "Hình 4.3: F1 lớp Thích và AUC-ROC trên Test")
    balance = runs["classification"]["class_balance"]
    add_body(doc,
        f"Tỷ lệ lớp Thích là {vi(balance['train']['positive_fraction'] * 100, 2)}% Train, "
        f"{vi(balance['validation']['positive_fraction'] * 100, 2)}% Validation và "
        f"{vi(balance['test']['positive_fraction'] * 100, 2)}% Test. Lớp có mất cân bằng vừa phải; "
        "Accuracy đa số trên Test xấp xỉ 61,03%. RF đạt Accuracy 73,03%, F1 0,8034 và AUC 0,7935; "
        "Logistic đạt 72,78%, F1 0,8033 và AUC 0,7943.")
    add_body(doc,
        "Theo Validation, Random Forest được chọn bằng F1=0,7992; ngưỡng của mô hình là 0,45, còn "
        "Logistic dùng 0,40. F1 Test của hai mô hình gần như bằng nhau (chênh khoảng 0,00008). Logistic "
        "có Recall cao hơn (0,9105 so với 0,9028) và inference nhanh hơn (0,881 so với 2,746 µs/rating); "
        "RF có Precision cao hơn (0,7236 so với 0,7187) và fit nhanh hơn nhờ giới hạn bootstrap. Kết luận "
        "nên gắn với chi phí lỗi: Recall nếu tránh bỏ sót phim thích, Precision nếu tránh đề xuất nhầm.")

    add_heading(doc, "4.3. NHÓM THUẬT TOÁN PHÂN CỤM")
    add_table(doc, "Bảng 4.5: Chất lượng hình học trên profile user–genre", tables["clustering"])
    add_table(doc, "Bảng 4.6: Gợi ý Top-10 trên cùng cohort tương lai", tables["ranking"])
    figure(doc, charts, "clustering", "Hình 4.4: NDCG@10 của baseline phổ biến và các phương pháp gom cụm")
    add_body(doc,
        "K-Means chia 159.356 user Train thành 2 cụm, coverage 100%, Silhouette 0,0969 và DBI 2,9995. "
        "ARI stability là 0,4165, cho thấy cấu trúc không hoàn toàn ổn định khi kiểm tra phân hoạch. "
        "DBSCAN chọn eps=3,284 và min_samples=10 nhưng tạo một cụm lớn với 1,01% noise; do chỉ có một "
        "cụm hợp lệ nên Silhouette/DBI là N/A, không phải 0.")
    add_body(doc,
        "K-Means chỉ nhỉnh hơn Popularity trên Validation NDCG@10 (0,06767 so với 0,06578), nhưng trên "
        "Test NDCG là 0,05911, thấp hơn Popularity 0,05945. DBSCAN gần Popularity trên cả hai đoạn và "
        "không có cấu trúc nhiều cụm để hỗ trợ diễn giải. Do đó, run này chứng minh pipeline phân cụm "
        "chạy trên hồ sơ đầy đủ, nhưng chưa chứng minh phân cụm cải thiện gợi ý tương lai.")
    add_body(doc,
        "Thời gian fit model ghi nhận 0,25 giây với K-Means và 68,74 giây với DBSCAN; tạo profile thêm "
        "10,41 giây. Inference trên 2.000 user lần lượt khoảng 0,128 và 0,387 ms/user. DBSCAN dùng "
        "implementation full-data của project do số user vượt 5.000; việc quy về một cụm khiến điểm "
        "nội bộ không đủ điều kiện để tuyên bố chất lượng.")

    add_heading(doc, "4.4. KHAI PHÁ LUẬT KẾT HỢP APRIORI")
    add_table(doc, "Bảng 4.7: Quy mô dữ liệu và tổng hợp luật Apriori", tables["association"])
    add_table(doc, "Bảng 4.8: Một số luật có Lift cao nhất trong Train", tables["rules"])
    add_body(doc,
        "Run Apriori dùng 122.029 transaction Train, catalog 80 phim phổ biến, 700 frequent itemsets và "
        "1.098 luật với min_support=0,08, min_confidence=0,30. Trung bình theo các luật được lưu là "
        "Support 0,1061, Confidence 0,4740 và Lift 1,9106. Luật đầu tiên nối The Return of the King "
        "với The Two Towers có Support 0,1421, Confidence 0,8346 và Lift 4,606.")
    add_body(doc,
        "Các luật phản ánh đồng xuất hiện trên tập Train; vì run này không ghi kết quả luật trên Validation "
        "hoặc Test nên chưa thể kết luận chúng dự đoán lựa chọn tương lai. Lift cao ở phim cùng series "
        "hợp lý để sinh candidate, nhưng không chứng minh nguyên nhân. Thời gian của pipeline khoảng "
        f"{vi(association_seconds(runs['association']), 2)} giây; file ghi rõ Apriori không có inference "
        "chuẩn như một predictor rating.")

    add_heading(doc, "4.5. NHÓM GIẢM CHIỀU DỮ LIỆU")
    add_heading(doc, "4.5.1. XGBoost Regressor", 2)
    add_heading(doc, "4.5.2. XGBoost Regressor + PCA", 2)
    add_heading(doc, "4.5.3. XGBoost Regressor + TruncatedSVD", 2)

    add_heading(doc, "4.6. NHÓM HỌC TĂNG CƯỜNG")
    add_body(doc,
        "Run gần nhất dùng DQN xấp xỉ Q-learning bằng mạng, catalog tối đa 30 action và horizon 5. "
        "DQN học từ 4.927 episode Train đủ điều kiện với một sweep và 24.416 training steps; giới hạn "
        "ngân sách là 100.000 bước. Policy được chọn bằng reward/action Validation, sau đó checkpoint "
        "được khóa trước Test. Mỗi policy được đánh giá trên cùng 1.000 user episode.")
    add_table(doc, "Bảng 4.9: So sánh policy trên Test", tables["rl"])
    figure(doc, charts, "rl", "Hình 4.5: Reward trung bình trên mỗi action của các policy Test")
    add_body(doc,
        "Popularity đạt reward/action Test 0,03261, cumulative reward 160,78 và NDCG@5 0,2529; DQN "
        "đạt 0,02859, 141,00 và 0,2315; Random đạt 0,02666, 131,44 và 0,2193. Popularity cũng là policy "
        "được Validation chọn. Feedback quan sát chỉ khoảng 5–6% action, nên reward/action chủ yếu dựa "
        "trên mẫu feedback thưa.")
    paired = runs["reinforcement"]["paired_test_uncertainty"]["policies"]["dqn"]
    add_body(doc,
        f"Paired bootstrap theo 1.000 user cho DQN trừ Popularity mean delta "
        f"{vi(paired['mean_delta'], 5)} với khoảng 95% [{vi(paired['ci95'][0], 5)}, "
        f"{vi(paired['ci95'][1], 5)}]; khoảng này chứa 0. Trong protocol offline hiện tại chưa có "
        "bằng chứng DQN vượt Popularity. Reward không có propensity/IPS, không mô hình hóa thay đổi "
        "sở thích hoặc giữ chân người xem; không dùng kết quả này như khẳng định hiệu quả sản phẩm.")
    add_body(doc,
        "DQN fit 49,07 giây; Test inference khoảng 0,074 ms/action và model khoảng 48,9 KiB. Popularity "
        "không cần train, inference khoảng 0,002 ms/action. Với web demo hiện tại, baseline phổ biến là "
        "phương án an toàn hơn cho nhóm tuần tự cho đến khi có đánh giá offline ít thiên lệch hoặc thử "
        "nghiệm với phản hồi người dùng.")

    add_heading(doc, "4.7. TỔNG HỢP VÀ LỰA CHỌN MÔ HÌNH DỰ ĐOÁN RATING")
    add_table(doc, "Bảng 4.10: Cân bằng chất lượng và chi phí của bốn mô hình rating",
              tables["rating_summary"])
    add_body(doc,
        "Theo cơ chế lựa chọn đã ghi trong artifact, NCF là mô hình được chọn bởi Validation (RMSE "
        "0,8046). Tuy nhiên, SVD-CF có CV mean thấp nhất 0,8504 ± 0,0339, inference nhanh nhất trong "
        "bốn model học, fit cuối nhanh hơn NCF khoảng 2,9 lần và có kích thước gần như tương đương. "
        "Chênh lệch CV với NCF nhỏ hơn độ biến thiên fold, nên đây là ứng viên triển khai cân bằng hơn "
        "theo chi phí, không phải bằng chứng SVD vượt NCF có ý nghĩa thống kê.")
    add_body(doc,
        "Nếu phải chọn mô hình cho demo theo bằng chứng hiện có, SVD-CF là ứng viên hợp lý khi ưu tiên "
        "CV và chi phí; NCF vẫn là lựa chọn Validation của run và có RMSE Validation thấp hơn 0,0017. "
        "Test của SVD tốt hơn NCF nhưng không được dùng để thay đổi lựa chọn. Trước triển khai rộng, "
        "nên khóa quy tắc chọn trên Validation/CV của một run xác nhận mới và ghi rõ fallback cho user/"
        "movie lạnh. UI hiện có thể phục vụ nhiều model, nhưng báo cáo không coi các run Experiment "
        "Studio là tự động nạp vào endpoint rating.")

    add_chapter(doc, "CHƯƠNG 5: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN")
    add_heading(doc, "5.1. KẾT LUẬN")
    add_body(doc,
        "Đồ án đã tổ chức một pipeline thực nghiệm trên MovieLens 25M và ứng dụng web với FastAPI, "
        "React/TypeScript, lưu dữ liệu và model artifact riêng. Bản làm sạch giữ nguyên toàn bộ rating "
        "hợp lệ; các run dự đoán rating/classification dùng dữ liệu đủ điều kiện theo chia thời gian "
        "trong từng user, tránh biến rating chưa biết thành số 0.")
    add_body(doc,
        "Trong bốn mô hình dự đoán rating, NCF được Validation chọn nhưng SVD-CF có mean CV RMSE thấp "
        "nhất cùng chi phí thấp hơn; chênh lệch nhỏ và biến thiên fold lớn nên cần xác nhận thêm. "
        "Logistic Regression và Random Forest có F1 Test gần như bằng nhau; Random Forest nhỉnh Accuracy, "
        "Logistic có AUC và latency nhẹ hơn. K-Means tạo hai cụm nhưng không cải thiện NDCG Test so với "
        "Popularity; DBSCAN về một cụm lớn. Apriori tìm được nhiều luật có Lift cao trong Train, song "
        "chưa có kiểm tra chất lượng luật trên dữ liệu tương lai. DQN không vượt baseline Popularity.")
    add_body(doc,
        "Kết quả giảm chiều cho phép so sánh XGBoost với PCA/TruncatedSVD chưa có trong artifact được "
        "dùng; báo cáo giữ trống Chương 4.5 theo yêu cầu, không điền số từ một run PCA khác mục tiêu.")

    add_heading(doc, "5.2. HẠN CHẾ")
    for item in [
        "CV hiện có trong các artifact supervised là 3 fold theo thời gian; yêu cầu 5 fold chưa được đáp ứng trong các run này.",
        "Tuning flag tắt; GridSearchCV/RandomizedSearchCV/Optuna chưa tạo kết quả tương ứng trong artifact được báo cáo.",
        "Bộ chia per-user loại user có dưới 10 rating hoặc không thể tách timestamp; kết quả không đo user mới hoàn toàn và chưa báo cold-start theo đủ ngưỡng.",
        "Các nhóm dùng protocol và mẫu số khác nhau: Apriori chia toàn cục; rating/classification/RL chia trong từng user; không so sánh metric xuyên nhóm.",
        "Clustering có Silhouette thấp và DBSCAN chỉ tạo một cụm; Apriori chưa có metric Validation/Test; các kết luận này mới mô tả run hiện có.",
        "Reward RL dựa trên feedback quan sát thưa, không có propensity/IPS và không mô hình hóa phản hồi sau khuyến nghị.",
        "MovieLens kết thúc năm 2019; kết quả offline không đại diện danh mục hoặc xu hướng xem hiện tại.",
    ]:
        add_bullet(doc, item)

    add_heading(doc, "5.3. HƯỚNG PHÁT TRIỂN")
    for item in [
        "Chạy lặp nhiều seed và hoàn thành protocol 5-fold thời gian nếu cần báo cáo mean ± std trên 5 fold; khóa trước cách chọn model và ngưỡng.",
        "Bật tuning có kiểm soát trên Train/Validation, lưu cấu hình, thời gian, RAM và fingerprint; giữ Test cho đánh giá cuối, không dùng Test để tối ưu.",
        "Hoàn tất phép so sánh XGBoost gốc, XGBoost+PCA và XGBoost+TruncatedSVD trên đúng split; fit reducer trong từng fold và báo Explained Variance, reconstruction error, RMSE/MAE, fit, inference, kích thước.",
        "Đo riêng user/item cold-start, tỷ lệ dự đoán được và chất lượng fallback như mean/bias/popularity; không gộp cold-start vào warm-user RMSE.",
        "Cải thiện phân cụm bằng phân tích đặc trưng, đo stability, coverage và baseline ranking; không chọn cấu hình chỉ vì Silhouette cao trên phần nhỏ được gán cụm.",
        "Đánh giá luật Apriori trên Validation/Test bằng candidate coverage, precision/recall/NDCG và độ ổn định của từng luật.",
        "Với gợi ý tuần tự, xây dựng protocol phản hồi có logging propensity hoặc thử nghiệm có kiểm soát; so DQN với Popularity bằng reward, ranking, coverage và chi phí.",
        "Tích hợp model được chọn vào web demo theo cùng tiêu chí Validation/CV, theo dõi latency request đơn và có fallback rõ khi hồ sơ người dùng rỗng hoặc ID chưa gặp.",
    ]:
        add_bullet(doc, item)

    add_chapter(doc, "TÀI LIỆU THAM KHẢO")
    references = [
        '[1] GroupLens Research, “MovieLens 25M Dataset README,” University of Minnesota. '
        'https://files.grouplens.org/datasets/movielens/ml-25m-README.html.',
        '[2] F. M. Harper and J. A. Konstan, “The MovieLens Datasets: History and Context,” '
        'ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, article 19, 2015. '
        'https://doi.org/10.1145/2827872.',
        '[3] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” '
        'Computer, vol. 42, no. 8, pp. 30–37, 2009. https://doi.org/10.1109/MC.2009.263.',
        '[4] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural Collaborative Filtering,” '
        'Proceedings of the 26th International Conference on World Wide Web, pp. 173–182, 2017. '
        'https://doi.org/10.1145/3038912.3052569.',
        '[5] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” Proceedings of the 22nd '
        'ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 785–794, 2016. '
        'https://doi.org/10.1145/2939672.2939785.',
        '[6] Scikit-learn Developers, “Linear Models” and “LogisticRegression,” documentation. '
        'https://scikit-learn.org/stable/modules/linear_model.html and '
        'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.',
        '[7] L. Breiman, “Random Forests,” Machine Learning, vol. 45, pp. 5–32, 2001. '
        'https://doi.org/10.1023/A:1010933404324.',
        '[8] S. P. Lloyd, “Least Squares Quantization in PCM,” IEEE Transactions on Information Theory, '
        'vol. 28, no. 2, pp. 129–137, 1982. https://doi.org/10.1109/TIT.1982.1056489.',
        '[9] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering '
        'Clusters in Large Spatial Databases with Noise,” Proceedings of KDD-96, pp. 226–231, 1996. '
        'https://www.cs.sfu.ca/~ester/papers/kdd_96.pdf.',
        '[10] R. Agrawal and R. Srikant, “Fast Algorithms for Mining Association Rules in Large Databases,” '
        'Proceedings of the 20th International Conference on Very Large Data Bases, pp. 487–499, 1994. '
        'https://www.vldb.org/conf/1994/P487.PDF.',
        '[11] I. T. Jolliffe and J. Cadima, “Principal Component Analysis: A Review and Recent Developments,” '
        'Philosophical Transactions of the Royal Society A, vol. 374, 2016. '
        'https://doi.org/10.1098/rsta.2015.0202.',
        '[12] Scikit-learn Developers, “TruncatedSVD,” documentation. '
        'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html.',
        '[13] C. J. C. H. Watkins and P. Dayan, “Q-Learning,” Machine Learning, vol. 8, pp. 279–292, 1992. '
        'https://doi.org/10.1023/A:1022676722315.',
        '[14] V. Mnih et al., “Human-Level Control through Deep Reinforcement Learning,” Nature, vol. 518, '
        'pp. 529–533, 2015. https://doi.org/10.1038/nature14236.',
        '[15] Scikit-learn Developers, “Metrics and Scoring: Quantifying the Quality of Predictions,” '
        'documentation. https://scikit-learn.org/stable/modules/model_evaluation.html.',
        '[16] Farama Foundation, “Gymnasium: An API Standard for Reinforcement Learning Environments,” '
        'documentation. https://gymnasium.farama.org/.',
        '[17] Nhóm 05, “MoviePredictionSystem: mã nguồn, dữ liệu, cấu hình và artifact thực nghiệm,” '
        'repository của đồ án, rà soát ngày 26/11/2026.',
    ]
    for reference in references:
        p = doc.add_paragraph(style="Report Reference")
        p.add_run(reference)

    add_chapter(doc, "PHỤ LỤC A: DẤU VẾT THỰC NGHIỆM")
    add_body(doc,
        "Các run dưới đây là nguồn của các bảng Chương 4. Mỗi thư mục chứa job.json/result.json và "
        "provenance; file JSON ghi cấu hình, split, metric, thời gian, runtime, fingerprint và ghi chú. "
        "Không gộp các run cũ có giao thức IID hoặc cohort khác vào các bảng này.")
    add_table(doc, "Bảng A.1: Run ID được trích cho báo cáo",
              [["Nhóm", "Run ID", "Protocol", "Dấu vết dữ liệu"],
               ["Rating", RUN_IDS["rating"], runs["rating"]["protocol"],
                runs["rating"]["sample_sha256"][:16] + "…"],
               ["Classification", RUN_IDS["classification"], runs["classification"]["protocol"],
                runs["classification"]["sample_sha256"][:16] + "…"],
               ["Clustering", RUN_IDS["clustering"], runs["clustering"]["protocol"],
                runs["clustering"]["sample_sha256"][:16] + "…"],
               ["Apriori", RUN_IDS["association"], runs["association"]["protocol"],
                runs["association"]["sample_sha256"][:16] + "…"],
               ["Reinforcement", RUN_IDS["reinforcement"], runs["reinforcement"]["protocol"],
                runs["reinforcement"]["sample_sha256"][:16] + "…"]])


def association_seconds(result: dict) -> float:
    timing = result.get("pipeline_timings", {})
    return float(timing.get("total_pipeline_seconds",
                            next(r["fit_seconds"] for r in result["rows"])))


def create_document(runs: dict, charts: dict[str, Path]) -> Path:
    if not TEMPLATE.is_file():
        raise FileNotFoundError(TEMPLATE)
    doc = Document(TEMPLATE)
    body = doc._element.body
    children = list(body)
    if len(children) < 49:
        raise ValueError("Template body is shorter than its two original cover pages.")
    # The supplied template stores the two original cover pages in body children 0..47.
    for child in children[48:]:
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    if len(doc.sections) != 2:
        raise ValueError(f"Expected two sections after preserving the covers; got {len(doc.sections)}.")
    add_styles(doc)
    build_content(doc, runs, charts)
    doc.core_properties.title = "Báo cáo đồ án: Dự đoán rating phim trên MovieLens 25M"
    doc.core_properties.subject = "Cơ sở lý thuyết, thiết kế thực nghiệm, kết quả và giới hạn"
    doc.core_properties.keywords = "MovieLens 25M, rating prediction, SVD-CF, NCF, Apriori, DQN"
    doc.save(OUTPUT)
    return OUTPUT


def main() -> None:
    runs = {name: load_run(name) for name in RUN_IDS}
    with tempfile.TemporaryDirectory(prefix="movielens_report_") as temporary:
        charts = create_charts(Path(temporary), runs)
        path = create_document(runs, charts)
    print(path)


if __name__ == "__main__":
    main()
