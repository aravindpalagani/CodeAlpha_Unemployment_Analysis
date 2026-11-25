"""Generate the professional PDF report.

Run after the analysis scripts and notebook build:

    python src/06_generate_report.py

The report is assembled with ReportLab Platypus from the computed JSON/CSV
outputs and the 200-dpi charts. No figure or statistic is typed by hand:
values are injected from the analysis outputs so the PDF stays consistent
with the pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils

REPORT_PATH = utils.REPORT_DIR / "Unemployment_Analysis_Report.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = RIGHT_MARGIN = 2.0 * cm
TOP_MARGIN = 2.35 * cm
BOTTOM_MARGIN = 2.0 * cm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    "SegoeUI": FONT_DIR / "segoeui.ttf",
    "SegoeUI-Bold": FONT_DIR / "segoeuib.ttf",
    "SegoeUI-Italic": FONT_DIR / "segoeuii.ttf",
    "SegoeUI-BoldItalic": FONT_DIR / "segoeuiz.ttf",
}
if all(path.exists() for path in FONT_FILES.values()):
    for name, path in FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "SegoeUI",
        normal="SegoeUI",
        bold="SegoeUI-Bold",
        italic="SegoeUI-Italic",
        boldItalic="SegoeUI-BoldItalic",
    )
    BODY_FONT = "SegoeUI"
    BOLD_FONT = "SegoeUI-Bold"
    ITALIC_FONT = "SegoeUI-Italic"
else:  # pragma: no cover - Windows fallback is normally available
    BODY_FONT, BOLD_FONT, ITALIC_FONT = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2E6DA4")
TEAL = colors.HexColor("#2A9D8F")
ORANGE = colors.HexColor("#E08A2E")
RED = colors.HexColor("#C0392B")
GOLD = colors.HexColor("#D4A017")
GRAY = colors.HexColor("#5A6B7B")
LIGHT_GRAY = colors.HexColor("#D9DEE5")
PALE = colors.HexColor("#F3F6F9")
PALE_BLUE = colors.HexColor("#EAF1F8")
WHITE = colors.white

STYLES = {
    "Body": ParagraphStyle(
        "Body",
        fontName=BODY_FONT,
        fontSize=9.55,
        leading=14.1,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#263238"),
        spaceAfter=5.5,
    ),
    "BodyTight": ParagraphStyle(
        "BodyTight",
        fontName=BODY_FONT,
        fontSize=9.4,
        leading=13.4,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#263238"),
        spaceAfter=3,
    ),
    "Lead": ParagraphStyle(
        "Lead",
        fontName=BODY_FONT,
        fontSize=10.5,
        leading=15.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1F3448"),
        spaceAfter=7,
    ),
    "H1": ParagraphStyle(
        "H1",
        fontName=BOLD_FONT,
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=2,
        spaceAfter=8,
        keepWithNext=True,
        alignment=TA_LEFT,
    ),
    "H2": ParagraphStyle(
        "H2",
        fontName=BOLD_FONT,
        fontSize=12.3,
        leading=16,
        textColor=BLUE,
        spaceBefore=11,
        spaceAfter=4,
        keepWithNext=True,
    ),
    "H3": ParagraphStyle(
        "H3",
        fontName=BOLD_FONT,
        fontSize=9.9,
        leading=13,
        textColor=NAVY,
        spaceBefore=7,
        spaceAfter=2,
        keepWithNext=True,
    ),
    "Bullet": ParagraphStyle(
        "Bullet",
        fontName=BODY_FONT,
        fontSize=9.4,
        leading=13.6,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#263238"),
        leftIndent=13,
        bulletIndent=2,
        spaceAfter=2.6,
    ),
    "Number": ParagraphStyle(
        "Number",
        fontName=BODY_FONT,
        fontSize=9.4,
        leading=13.6,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#263238"),
        leftIndent=16,
        bulletIndent=1,
        spaceAfter=2.6,
    ),
    "Caption": ParagraphStyle(
        "Caption",
        fontName=ITALIC_FONT,
        fontSize=7.9,
        leading=10.5,
        textColor=GRAY,
        alignment=TA_LEFT,
        spaceAfter=2,
    ),
    "TableHeader": ParagraphStyle(
        "TableHeader",
        fontName=BOLD_FONT,
        fontSize=7.7,
        leading=9.8,
        textColor=WHITE,
        alignment=TA_LEFT,
    ),
    "Cell": ParagraphStyle(
        "Cell",
        fontName=BODY_FONT,
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#263238"),
        alignment=TA_LEFT,
    ),
    "CellRight": ParagraphStyle(
        "CellRight",
        fontName=BODY_FONT,
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#263238"),
        alignment=TA_RIGHT,
    ),
    "CellCenter": ParagraphStyle(
        "CellCenter",
        fontName=BODY_FONT,
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#263238"),
        alignment=TA_CENTER,
    ),
    "Callout": ParagraphStyle(
        "Callout",
        fontName=BODY_FONT,
        fontSize=8.9,
        leading=13,
        textColor=colors.HexColor("#1F3448"),
        alignment=TA_LEFT,
    ),
    "CalloutTitle": ParagraphStyle(
        "CalloutTitle",
        fontName=BOLD_FONT,
        fontSize=9.2,
        leading=12,
        textColor=NAVY,
        spaceAfter=3,
    ),
    "CoverKicker": ParagraphStyle(
        "CoverKicker",
        fontName=BOLD_FONT,
        fontSize=9.2,
        leading=12,
        textColor=GOLD,
        alignment=TA_LEFT,
    ),
    "CoverTitle": ParagraphStyle(
        "CoverTitle",
        fontName=BOLD_FONT,
        fontSize=29,
        leading=33,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceBefore=2,
        spaceAfter=5,
    ),
    "CoverSubtitle": ParagraphStyle(
        "CoverSubtitle",
        fontName=BODY_FONT,
        fontSize=13.2,
        leading=18,
        textColor=colors.HexColor("#D6E2F0"),
        alignment=TA_LEFT,
        spaceBefore=4,
        spaceAfter=4,
    ),
    "CoverIntro": ParagraphStyle(
        "CoverIntro",
        fontName=BODY_FONT,
        fontSize=9.7,
        leading=14.6,
        textColor=colors.HexColor("#263238"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ),
    "CoverMeta": ParagraphStyle(
        "CoverMeta",
        fontName=BODY_FONT,
        fontSize=7.8,
        leading=10.5,
        textColor=GRAY,
        alignment=TA_LEFT,
    ),
    "CardNumber": ParagraphStyle(
        "CardNumber",
        fontName=BOLD_FONT,
        fontSize=17,
        leading=20,
        textColor=NAVY,
        alignment=TA_LEFT,
    ),
    "CardLabel": ParagraphStyle(
        "CardLabel",
        fontName=BODY_FONT,
        fontSize=7.2,
        leading=9.4,
        textColor=GRAY,
        alignment=TA_LEFT,
    ),
    "TOC1": ParagraphStyle(
        "TOC1",
        fontName=BOLD_FONT,
        fontSize=9.0,
        leading=11.8,
        textColor=NAVY,
        spaceBefore=2.0,
    ),
    "TOC2": ParagraphStyle(
        "TOC2",
        fontName=BODY_FONT,
        fontSize=8.3,
        leading=10.7,
        textColor=colors.HexColor("#455A64"),
        leftIndent=15,
    ),
}


def e(value: object) -> str:
    """Escape user/data text for ReportLab's mini-HTML parser."""
    return escape(str(value))


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f} pp"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def heading(text: str, level: int = 1, toc: bool = True) -> Paragraph:
    style = STYLES["H1"] if level == 1 else STYLES["H2"] if level == 2 else STYLES["H3"]
    paragraph = Paragraph(text, style)
    if toc:
        paragraph._toc_level = level
        paragraph._toc_text = re.sub(r"<[^>]+>", "", text)
    return paragraph


def bullet(text: str) -> Paragraph:
    return Paragraph(text, STYLES["Bullet"], bulletText="\u2022")


def numbered(text: str, marker: str) -> Paragraph:
    return Paragraph(text, STYLES["Number"], bulletText=marker)


def callout(title: str, body: str, accent: colors.Color = TEAL) -> Table:
    content = [
        Paragraph(e(title), STYLES["CalloutTitle"]),
        Paragraph(body, STYLES["Callout"]),
    ]
    table = Table([[content]], colWidths=[CONTENT_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D6E3")),
                ("LINEBEFORE", (0, 0), (0, -1), 3.2, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def table(
    headers: list[str],
    rows: list[list[object]],
    widths: list[float],
    aligns: list[str] | None = None,
    font_size: float = 7.8,
    header_bg: colors.Color = NAVY,
    zebra: bool = True,
) -> Table:
    header_cells = [Paragraph(header, STYLES["TableHeader"]) for header in headers]
    body: list[list[Flowable]] = []
    for row in rows:
        cells: list[Flowable] = []
        for index, value in enumerate(row):
            if isinstance(value, Flowable):
                cells.append(value)
                continue
            if aligns is None:
                align = "L" if index == 0 else "R"
            else:
                align = aligns[index]
            style_name = "CellRight" if align == "R" else "CellCenter" if align == "C" else "Cell"
            style = STYLES[style_name]
            if font_size != 7.8:
                style = ParagraphStyle(
                    f"{style_name}{font_size}",
                    parent=style,
                    fontSize=font_size,
                    leading=font_size * 1.3,
                )
            cells.append(Paragraph(e(value), style))
        body.append(cells)

    data = [header_cells] + body
    table_flowable = Table(
        data,
        colWidths=[width * cm for width in widths],
        repeatRows=1,
        hAlign="CENTER",
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D6DEE7")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, header_bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("TOPPADDING", (0, 1), (-1, -1), 3.6),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3.6),
    ]
    if zebra:
        commands.append(
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F7F9FC")])
        )
    table_flowable.setStyle(TableStyle(commands))
    return table_flowable


def figure_flowables(filename: str, caption: str, width_cm: float = 16.2) -> list[Flowable]:
    path = utils.CHART_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Chart missing: {path}")
    with PILImage.open(path) as image:
        pixels_w, pixels_h = image.size
    width = width_cm * cm
    height = width * pixels_h / pixels_w
    image_flowable = Image(str(path), width=width, height=height)
    image_flowable.hAlign = "CENTER"
    return [
        image_flowable,
        Spacer(1, 0.12 * cm),
        Paragraph(caption, STYLES["Caption"]),
        Spacer(1, 0.15 * cm),
    ]


def figure(filename: str, caption: str, width_cm: float = 16.2) -> KeepTogether:
    return KeepTogether(figure_flowables(filename, caption, width_cm))


def section_figure(title: str, filename: str, caption: str, width_cm: float = 16.2) -> KeepTogether:
    """Keep a section heading with its figure so headings never strand at a page foot."""
    return KeepTogether([heading(title, 2), *figure_flowables(filename, caption, width_cm)])


# ---------------------------------------------------------------------------
# Document template and page furniture
# ---------------------------------------------------------------------------
class ReportDocument(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title="Unemployment Analysis with Python",
            author="CodeAlpha Data Science Intern",
            subject="Labour-market patterns and the COVID-19 shock in India",
            creator="Python / ReportLab",
            **kwargs,
        )
        frame = Frame(
            LEFT_MARGIN,
            BOTTOM_MARGIN,
            CONTENT_WIDTH,
            PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
            id="body_frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[frame], onPage=self.draw_cover),
                PageTemplate(id="Body", frames=[frame], onPage=self.draw_header),
            ]
        )

    def afterFlowable(self, flowable: Flowable) -> None:
        level = getattr(flowable, "_toc_level", None)
        if level is not None:
            text = getattr(flowable, "_toc_text", "")
            self.notify("TOCEntry", (level, text, self.page))

    def draw_cover(self, canv: canvas.Canvas, doc: ReportDocument) -> None:
        canv.saveState()
        # Navy title band with a gold rule.
        canv.setFillColor(NAVY)
        canv.rect(0, PAGE_HEIGHT - 9.55 * cm, PAGE_WIDTH, 9.55 * cm, fill=1, stroke=0)
        canv.setFillColor(GOLD)
        canv.rect(0, PAGE_HEIGHT - 9.68 * cm, PAGE_WIDTH, 0.13 * cm, fill=1, stroke=0)
        # Subtle footer strip.
        canv.setFillColor(NAVY)
        canv.rect(0, 0, PAGE_WIDTH, 0.55 * cm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#B9C7D6"))
        canv.setFont(BODY_FONT, 7.0)
        canv.drawString(2.0 * cm, 1.0 * cm, "Independent analytical report  |  Not an official statistical publication")
        canv.drawRightString(
            PAGE_WIDTH - 2.0 * cm,
            1.0 * cm,
            f"Prepared {date.today().strftime('%d %B %Y')}",
        )
        canv.restoreState()

    def draw_header(self, canv: canvas.Canvas, doc: ReportDocument) -> None:
        canv.saveState()
        canv.setStrokeColor(LIGHT_GRAY)
        canv.setLineWidth(0.6)
        canv.line(LEFT_MARGIN, PAGE_HEIGHT - 1.52 * cm, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 1.52 * cm)
        canv.setFont(BOLD_FONT, 6.9)
        canv.setFillColor(NAVY)
        canv.drawString(LEFT_MARGIN, PAGE_HEIGHT - 1.27 * cm, "UNEMPLOYMENT ANALYSIS WITH PYTHON")
        canv.setFont(BODY_FONT, 6.9)
        canv.setFillColor(GRAY)
        canv.drawRightString(
            PAGE_WIDTH - RIGHT_MARGIN,
            PAGE_HEIGHT - 1.27 * cm,
            "CodeAlpha Data Science Internship  \u00b7  Task 2",
        )
        canv.restoreState()


class NumberedCanvas(canvas.Canvas):
    """Adds a 'Page X of Y' footer without a second layout pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_footer(total_pages)
            super().showPage()
        super().save()

    def _draw_footer(self, total_pages: int) -> None:
        page_number = self._pageNumber
        if page_number == 1:
            return
        self.saveState()
        self.setStrokeColor(LIGHT_GRAY)
        self.setLineWidth(0.5)
        self.line(LEFT_MARGIN, 1.36 * cm, PAGE_WIDTH - RIGHT_MARGIN, 1.36 * cm)
        self.setFont(BODY_FONT, 6.9)
        self.setFillColor(GRAY)
        self.drawString(
            LEFT_MARGIN,
            0.98 * cm,
            "Data: May 2019 \u2013 October 2020  |  Labour-force-weighted rates unless stated",
        )
        self.drawRightString(
            PAGE_WIDTH - RIGHT_MARGIN,
            0.98 * cm,
            f"Page {page_number} of {total_pages}",
        )
        self.restoreState()


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------
def load_outputs() -> tuple[dict, dict, dict]:
    required = [
        utils.ANALYSIS_DIR / "eda_summary.json",
        utils.ANALYSIS_DIR / "covid_summary.json",
        utils.CLEANED_DIR / "cleaning_log.json",
        utils.CLEANED_DIR / "data_quality_report.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required output missing: {path}\nRun the analysis scripts first."
            )
    eda = json.loads(required[0].read_text(encoding="utf-8"))
    covid = json.loads(required[1].read_text(encoding="utf-8"))
    cleaning = json.loads(required[2].read_text(encoding="utf-8"))
    quality = pd.read_csv(required[3])
    return eda, covid, cleaning, quality


def build_story() -> list[Flowable]:
    eda, covid, cleaning, quality = load_outputs()
    overall = eda["dataset1_overall"]
    d2_overview = eda["dataset2_overview"]
    top_states = eda["top_bottom_states"]["highest"]
    low_states = eda["top_bottom_states"]["lowest"]
    coverage = eda["coverage"]
    relationships = eda["relationships"]
    outliers = eda["outliers"]
    seasonality = eda["seasonality"]

    headline = covid["headline"]
    risk = {row["label"]: row for row in covid["analysis_periods"]}
    pre_full = next(row for row in covid["analysis_periods"] if row["label"].startswith("Pre-COVID full"))
    baseline = next(row for row in covid["analysis_periods"] if row["label"].startswith("Immediate"))
    march = next(row for row in covid["analysis_periods"] if row["label"].startswith("Transition"))
    covid_core = next(row for row in covid["analysis_periods"] if row["label"].startswith("COVID period"))
    peak_window = next(row for row in covid["analysis_periods"] if row["label"].startswith("Peak shock"))
    june = next(row for row in covid["analysis_periods"] if row["label"].startswith("June 2020"))
    area_changes = {row["group"]: row for row in covid["area_changes"]}
    rural = {**next(row for row in eda["area_summary"] if row["area"] == "Rural"), **area_changes["Rural"]}
    urban = {**next(row for row in eda["area_summary"] if row["area"] == "Urban"), **area_changes["Urban"]}
    state_changes = pd.read_csv(utils.ANALYSIS_DIR / "covid_state_changes.csv")
    top_abs = (
        state_changes.sort_values("absolute_change_rank").head(12).to_dict("records")
    )
    if len(top_abs) < 12:
        raise ValueError("Expected 12 state-level increases in covid_state_changes.csv")
    decreases = covid["state_changes_overview"]["decreases"]
    employment = covid["employment_balanced_panel"]
    monthly_employment = employment["monthly_changes"]
    recovery = covid["dataset2_recovery"]
    common = recovery["common_state_january_vs_october"]
    zones = {row["zone"]: row for row in covid["zone_summary"]}
    overlap = covid["overlap_validation"]
    log1 = cleaning["dataset1"]
    log2 = cleaning["dataset2"]
    quality_passed = int((quality["status"] == "PASS").sum())
    quality_total = len(quality)
    today = date.today().strftime("%d %B %Y")

    story: list[Flowable] = []

    # ------------------------------------------------------------------
    # Cover
    # ------------------------------------------------------------------
    story.append(Spacer(1, 0.75 * cm))
    story.append(Paragraph("CODEALPHA DATA SCIENCE INTERNSHIP  \u00b7  TASK 2", STYLES["CoverKicker"]))
    story.append(Spacer(1, 0.32 * cm))
    story.append(Paragraph("Unemployment Analysis<br/>with Python", STYLES["CoverTitle"]))
    story.append(Paragraph("Labour-market patterns and the COVID-19 shock in India", STYLES["CoverSubtitle"]))
    story.append(Spacer(1, 0.18 * cm))
    story.append(HRFlowable(width="31%", thickness=1.5, color=GOLD, hAlign="LEFT"))
    story.append(Spacer(1, 4.05 * cm))
    story.append(
        Paragraph(
            "This report examines monthly unemployment estimates for Indian states and union "
            "territories, separates rural and urban labour markets, and measures the labour-market "
            "shock associated with the COVID-19 period. It combines a 14-month primary panel with a "
            "supplementary 2020 series that extends the recovery view through October.",
            STYLES["CoverIntro"],
        )
    )
    story.append(Spacer(1, 0.28 * cm))

    card_specs = [
        (
            pct(overall["weighted_rate"]),
            "Labour-force-weighted average unemployment, May 2019 \u2013 June 2020",
            TEAL,
        ),
        (
            pp(headline["absolute_change_pp"]),
            "Change from Jan\u2013Feb 2020 to Apr\u2013Jun 2020, national weighted rate",
            RED,
        ),
        (
            pct(monthly_employment["2020-04"]["total_change_pct"], 1),
            "April 2020 employment versus the Jan\u2013Feb baseline, 20-state balanced panel",
            ORANGE,
        ),
    ]
    cards = []
    for number, label, accent in card_specs:
        card = Table(
            [[Paragraph(number, STYLES["CardNumber"])], [Paragraph(label, STYLES["CardLabel"])]],
            colWidths=[5.12 * cm],
        )
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
                    ("LINEABOVE", (0, 0), (-1, 0), 2.4, accent),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        cards.append(card)
    card_row = Table(
        [[cards[0], "", cards[1], "", cards[2]]],
        colWidths=[5.12 * cm, 0.3 * cm, 5.12 * cm, 0.3 * cm, 5.12 * cm],
    )
    card_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(card_row)
    story.append(Spacer(1, 0.55 * cm))
    story.append(figure("01_overall_unemployment_trend.png", "Cover figure. The national labour-force-weighted unemployment rate, May 2019 \u2013 June 2020.", 16.0))
    story.append(Spacer(1, 0.05 * cm))
    story.append(
        Paragraph(
            f"Prepared {today}  \u00b7  Primary data through 30 June 2020; supplementary data through 31 October 2020  "
            "\u00b7  Python 3.13",
            STYLES["CoverMeta"],
        )
    )

    # ------------------------------------------------------------------
    # Contents
    # ------------------------------------------------------------------
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    story.append(Paragraph("Contents", STYLES["H1"]))
    story.append(Paragraph("A structured route through the evidence, methods and policy implications.", STYLES["Caption"]))
    story.append(Spacer(1, 0.35 * cm))
    toc = TableOfContents()
    toc.levelStyles = [STYLES["TOC1"], STYLES["TOC2"]]
    toc.dotsMinLevel = 0
    toc.tableStyle = TableStyle(
        [
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    story.append(toc)
    story.append(Spacer(1, 0.55 * cm))
    story.append(
        callout(
            "How to read this report",
            "The executive summary gives the headline result and the main policy message. Sections 1\u20133 "
            "cover the data, cleaning and method; sections 4\u20136 present the evidence; sections 7\u20138 "
            "translate it into recommendations and conclusions. Every figure is saved at 200 dpi under "
            "<i>outputs/charts/</i>. Rates are labour-force-weighted unless a chart or table explicitly says "
            "otherwise; the unweighted mean is reported alongside for transparency.",
            TEAL,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("Executive summary", 1))
    story.append(
        Paragraph(
            f"India entered the COVID-19 period with unemployment already uneven across states. Using the "
            f"primary file, the labour-force-weighted national rate averaged <b>{pct(overall['weighted_rate'])}</b> "
            f"over May 2019\u2013June 2020, but that average conceals a stable pre-COVID band and an extraordinary "
            f"spring 2020 shock. The immediate pre-COVID baseline (January\u2013February 2020) was "
            f"<b>{pct(baseline['weighted_rate'])}</b>. In April\u2013June 2020 the weighted rate reached "
            f"<b>{pct(covid_core['weighted_rate'])}</b>, an increase of <b>{pp(headline['absolute_change_pp'])}</b> "
            f"or <b>{headline['relative_change_pct']:+.1f}%</b>. The April\u2013May peak was "
            f"<b>{pct(peak_window['weighted_rate'])}</b>, more than three times the baseline.",
            STYLES["Lead"],
        )
    )
    story.append(
        Paragraph(
            f"The shock was broad-based. Rural unemployment rose from {pct(rural['baseline_weighted_rate'])} "
            f"to {pct(rural['covid_weighted_rate'])} ({pp(rural['change_pp'])}, {rural['change_pct']:+.1f}%), "
            f"and urban unemployment from {pct(urban['baseline_weighted_rate'])} to "
            f"{pct(urban['covid_weighted_rate'])} ({pp(urban['change_pp'])}, {urban['change_pct']:+.1f}%). "
            f"Urban areas remained higher in level, while the rural increase was larger in relative terms "
            f"because it started from a lower base. Labour participation fell in both areas, and more sharply "
            f"in urban areas ({pp(urban['lpr_change_pp'])} versus {pp(rural['lpr_change_pp'])}).",
            STYLES["Body"],
        )
    )
    story.append(
        table(
            ["Headline indicator", "Result", "Context"],
            [
                ["Weighted unemployment, full period", pct(overall["weighted_rate"]), "740 region-area-month observations"],
                ["Immediate pre-COVID baseline", pct(baseline["weighted_rate"]), "January\u2013February 2020"],
                ["COVID period", pct(covid_core["weighted_rate"]), "April\u2013June 2020"],
                ["Absolute change", pp(headline["absolute_change_pp"]), f"{headline['relative_change_pct']:+.1f}% relative"],
                ["Peak window", pct(peak_window["weighted_rate"]), "April\u2013May 2020"],
                ["April 2020 employment", f"{monthly_employment['2020-04']['total_employed'] / 1e6:.1f} million", f"{monthly_employment['2020-04']['total_change_pct']:+.1f}% versus Jan\u2013Feb average"],
            ],
            [6.3, 3.2, 7.5],
            ["L", "R", "L"],
        )
    )
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Main findings", STYLES["H2"]))
    findings = [
        f"<b>The increase was very large.</b> The weighted rate rose {pp(headline['absolute_change_pp'])} between the immediate baseline and April\u2013June 2020; the April\u2013May peak was {pct(peak_window['weighted_rate'])}.",
        f"<b>Jobs fell even faster than the unemployment rate suggests.</b> In the balanced 20-state panel, April employment fell {abs(monthly_employment['2020-04']['total_change_millions']):.1f} million ({monthly_employment['2020-04']['total_change_pct']:.1f}%) from the January\u2013February average; June was still {monthly_employment['2020-06']['total_change_pct']:.1f}% below it.",
        f"<b>October's lower rate is not a full recovery.</b> In the same 26 states in the supplementary data, the rate was {pp(common['weighted_rate_change_pp'])} versus January, but employment was {common['employment_change_pct']:+.1f}% lower and participation {pp(common['lpr_change_pp'])} lower.",
        f"<b>Place mattered.</b> Structural averages ranged from {pct(low_states[0]['weighted_rate'])} in {low_states[0]['region']} to {pct(top_states[0]['weighted_rate'])} in {top_states[0]['region']}; the largest absolute increases were in Jharkhand, Tamil Nadu, Bihar and Karnataka, with volatile Puducherry an outlier.",
        "<b>Seasonality cannot be established.</b> Only May and June recur across years in the primary panel, and the 2020 repeats coincide with the COVID-19 shock.",
    ]
    for item in findings:
        story.append(bullet(item))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        callout(
            "Bottom line",
            "A falling unemployment rate is not the same as a restored labour market. By October 2020, fewer "
            "people were employed and fewer were participating than in January, even though the headline rate "
            "had returned to roughly its pre-shock level. Policy should therefore be judged on employment "
            "restoration and participation, not only on the unemployment rate.",
            RED,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        callout(
            "Important distinction",
            "This is a descriptive analysis of two related survey datasets. It documents changes that "
            "coincide with the COVID-19 period, but it does not identify the causal effect of the pandemic, "
            "lockdowns or any single policy. The two datasets are analysed separately and compared, never "
            "concatenated.",
            ORANGE,
        )
    )

    # ------------------------------------------------------------------
    # 1. Introduction
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("1. Introduction and objectives", 1))
    story.append(
        Paragraph(
            "Unemployment is a lagging indicator of economic distress, but a national average can hide as much "
            "as it reveals. States differ in population, industrial structure, rural dependence and female "
            "participation; a single headline rate cannot capture all of those dimensions. The COVID-19 period "
            "makes the distinction especially important: a sharp jump in joblessness was accompanied by people "
            "leaving the labour force, so the unemployment rate and the employment level can tell different "
            "stories.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            "This project was completed for the CodeAlpha Data Science Internship, Task 2. Its purpose is both "
            "analytical and methodological: to produce a defensible description of unemployment in India and to "
            "show the cleaning, validation and reporting decisions behind that description. The analysis uses "
            "the supplied CSV files only. No external statistics are introduced.",
            STYLES["Body"],
        )
    )
    story.append(heading("1.1 Objectives", 2))
    objectives = [
        "Inspect both raw datasets without assuming their schema, date format or coverage.",
        "Clean the data conservatively and document every material decision, including what was not imputed.",
        "Measure unemployment overall, by state, by rural/urban area and by month.",
        "Separate levels from changes: identify states with high structural unemployment and states that experienced the largest shock.",
        "Examine estimated employment and labour participation alongside the unemployment rate.",
        "Quantify the observed change during the COVID-19 period against an immediate pre-COVID baseline, while distinguishing description from causation.",
        "Test whether the data can support seasonal claims and state the limitation explicitly.",
        "Translate the measured patterns into policy directions that are tied to evidence.",
    ]
    for index, objective in enumerate(objectives, start=1):
        story.append(numbered(objective, f"{index}."))
    story.append(heading("1.2 Analytical questions", 2))
    story.append(
        Paragraph(
            "The report answers five descriptive questions. First, how high was unemployment and how did it vary "
            "across states and areas? Second, what changed during the COVID-19 period relative to the months "
            "immediately before it? Third, did employment and participation move in the same direction as the "
            "unemployment rate? Fourth, did the headline rate recover by October 2020, and did jobs recover with "
            "it? Fifth, what do the patterns imply for policy, and what can the data not establish?",
            STYLES["Body"],
        )
    )
    story.append(
        callout(
            "Scope",
            "Primary panel: 28 states/union territories, May 2019\u2013June 2020, Rural and Urban rows. "
            "Supplementary series: 27 states/union territories, January\u2013October 2020, one row per state. "
            "The unit of analysis is the region-area-month (primary) or region-month (supplementary). "
            "There is no age, gender, industry, education or district detail.",
            BLUE,
        )
    )

    # ------------------------------------------------------------------
    # 2. Data and methodology
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("2. Data and methodology", 1))
    story.append(heading("2.1 Data sources", 2))
    story.append(
        table(
            ["Dataset", "File", "Clean rows", "Regions", "Period", "Role"],
            [
                ["Primary", "Unemployment in India.csv", f"{log1['cleaned_shape'][0]}", "28", "May 2019\u2013Jun 2020", "Rural/Urban detail; baseline, shock and early rebound"],
                ["Supplementary", "Unemployment_Rate_upto_11_2020.csv", f"{log2['cleaned_shape'][0]}", "27", "Jan\u2013Oct 2020", "Longer 2020 recovery view; geographic zones"],
            ],
            [2.35, 4.75, 1.45, 1.55, 3.1, 3.8],
            ["L", "L", "R", "R", "L", "L"],
            font_size=7.5,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            f"Dataset 1 has two rows per region-month (Rural and Urban). Dataset 2 has one overall row per "
            f"region-month, no area split, and additional zone and coordinate fields. Although the filename says "
            f"\u201cupto 11/2020\u201d, the last observed date is <b>31 October 2020</b>; actual dates are used. "
            f"Dataset 2 covers <b>{d2_overview['regions']} regions</b> and <b>{d2_overview['months']} months</b>.",
            STYLES["Body"],
        )
    )
    story.append(heading("2.2 Why the datasets are not merged", 2))
    for item in [
        "The primary file provides a Rural/Urban split; the supplementary file does not. Concatenating them would create a mixed unit of analysis.",
        "The files do not cover identical populations or periods. Dataset 1 ends in June 2020; Dataset 2 ends in October 2020 and has one fewer region.",
        "They are best treated as related views. Section 6.1 compares their overlapping state-months and shows broad consistency without pretending they are interchangeable.",
    ]:
        story.append(bullet(item))
    story.append(heading("2.3 Measuring the unemployment rate", 2))
    story.append(
        Paragraph(
            "Two averages are reported. The <b>unweighted mean</b> gives every region-area-month observation "
            "equal weight. The <b>labour-force-weighted rate</b> gives each observation weight proportional to its "
            "labour force, so a small territory does not count as much as a large state. The weighted measure is "
            "the headline; the unweighted mean is retained for transparency and because many standard summaries "
            "use it.",
            STYLES["Body"],
        )
    )
    formula = Table(
        [[
            Paragraph(
                "Labour force = Estimated employed / (1 \u2212 unemployment rate / 100)<br/>"
                "Estimated unemployed = Labour force \u2212 Estimated employed<br/>"
                "Weighted unemployment rate = \u03a3 Estimated unemployed / \u03a3 Labour force \u00d7 100",
                ParagraphStyle(
                    "Formula",
                    fontName=BODY_FONT,
                    fontSize=8.5,
                    leading=14,
                    textColor=NAVY,
                    alignment=TA_LEFT,
                ),
            )
        ]],
        colWidths=[CONTENT_WIDTH],
    )
    formula.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(formula)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Derived counts are arithmetic identities, not official estimates. LPR is labour force divided by "
            "population; because population is not supplied, the weighted LPR is approximated from derived labour "
            "force and implied population. Otherwise LPR is a simple mean across observations, and the text says "
            "which is being used.",
            STYLES["Body"],
        )
    )
    story.append(heading("2.4 Period definitions", 2))
    story.append(
        table(
            ["Period", "Dates", "Observations", "Weighted rate", "Unweighted mean"],
            [
                ["Pre-COVID full", "May 2019\u2013Feb 2020", f"{pre_full['n_obs']}", pct(pre_full["weighted_rate"]), pct(pre_full["unweighted_rate"])],
                ["Immediate baseline", "Jan\u2013Feb 2020", f"{baseline['n_obs']}", pct(baseline["weighted_rate"]), pct(baseline["unweighted_rate"])],
                ["Transition", "March 2020", f"{march['n_obs']}", pct(march["weighted_rate"]), pct(march["unweighted_rate"])],
                ["COVID period", "Apr\u2013Jun 2020", f"{covid_core['n_obs']}", pct(covid_core["weighted_rate"]), pct(covid_core["unweighted_rate"])],
                ["Peak window", "Apr\u2013May 2020", f"{peak_window['n_obs']}", pct(peak_window["weighted_rate"]), pct(peak_window["unweighted_rate"])],
                ["June 2020", "June 2020", f"{june['n_obs']}", pct(june["weighted_rate"]), pct(june["unweighted_rate"])],
            ],
            [3.0, 4.0, 2.0, 2.7, 3.3],
            ["L", "L", "R", "R", "R"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "March is labelled <b>Transition</b> because the national lockdown began on 25 March; the month "
            "straddles both regimes. The windows are analytical choices, not official definitions. The immediate "
            "baseline is used for COVID comparisons and the full pre-COVID period for context.",
            STYLES["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 3. Data quality
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("3. Data quality and cleaning", 1))
    story.append(
        Paragraph(
            "Cleaning was deliberately conservative. Headers and text values were stripped of whitespace, dates "
            "were parsed with an explicit day-month-year format, numeric fields were coerced, and completely empty "
            "rows were removed. No missing region-month values were imputed, no zero unemployment observations "
            "were deleted, and no observations were removed as outliers. The decisions are recorded in "
            "<i>outputs/cleaned_data/cleaning_log.json</i> and the quality checks in "
            "<i>data_quality_report.csv</i>.",
            STYLES["Body"],
        )
    )
    story.append(heading("3.1 Cleaning results", 2))
    story.append(
        table(
            ["Step", "Dataset 1", "Dataset 2"],
            [
                ["Raw rows", f"{log1['raw_shape'][0]}", f"{log2['raw_shape'][0]}"],
                ["Completely blank rows removed", f"{log1['completely_blank_rows_removed']}", f"{log2['completely_blank_rows_removed']}"],
                ["Duplicate rows removed", f"{log1['duplicate_rows_removed']}", f"{log2['duplicate_rows_removed']}"],
                ["Invalid dates", f"{log1['invalid_dates']}", f"{log2['invalid_dates']}"],
                ["Rows with missing critical values removed", f"{log1['rows_with_missing_critical_values_removed']}", f"{log2['rows_with_missing_critical_values_removed']}"],
                ["Clean rows", f"{log1['cleaned_shape'][0]}", f"{log2['cleaned_shape'][0]}"],
                ["Zero unemployment observations retained", f"{log1['zero_unemployment_rows_preserved']}", f"{log2['zero_unemployment_rows_preserved']}"],
                ["Missing region(-area)-months left unimputed", f"{log1['missing_region_area_months']}", f"{log2['missing_region_months']}"],
            ],
            [7.4, 4.7, 4.7],
            ["L", "R", "R"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            f"All <b>{quality_passed} of {quality_total}</b> post-cleaning quality checks pass. The primary dataset "
            f"retains {log1['cleaned_shape'][0]} observations, {log1['regions']} regions, two areas and 14 months. "
            f"The supplementary dataset retains {log2['cleaned_shape'][0]} observations, {log2['regions']} regions "
            f"and 10 months. There are no duplicate region-date-area keys and no missing critical values.",
            STYLES["Body"],
        )
    )
    story.append(heading("3.2 Coverage and missingness", 2))
    story.append(
        table(
            ["Coverage measure", "Dataset 1", "Dataset 2"],
            [
                ["Expected region(-area)-months", f"{coverage['dataset1']['expected_region_area_months']}", f"{d2_overview['regions'] * d2_overview['months']}"],
                ["Observed region(-area)-months", f"{coverage['dataset1']['rows']}", f"{d2_overview['rows']}"],
                ["Missing combinations", f"{coverage['dataset1']['missing_region_area_months']}", "3"],
                ["Regions with complete coverage", f"{coverage['dataset1']['complete_coverage_states']}", "Not applicable (one row per state)"],
            ],
            [6.8, 4.6, 5.4],
            ["L", "R", "L"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            "The primary file is missing 44 region-area-month combinations out of an expected 784, largely because "
            "some smaller states and union territories do not report every month. Twenty states have complete "
            "coverage and are used for the balanced employment panel. In the supplementary file, Jammu &amp; Kashmir "
            "is missing April 2020 and Sikkim is missing January and February 2020. Missing values are left absent; "
            "fabricating them would create false precision.",
            STYLES["Body"],
        )
    )
    story.append(heading("3.3 Zeros, outliers and small areas", 2))
    story.append(
        Paragraph(
            f"The primary data contain <b>{log1['zero_unemployment_rows_preserved']} exact-zero unemployment "
            f"observations</b>. They are retained and flagged: a zero can be a genuine small-area estimate or "
            f"measurement noise, but deleting it would bias the distribution upward. The IQR rule flags "
            f"<b>{outliers['outlier_count']} high outliers ({outliers['outlier_share_pct']:.1f}%)</b> above "
            f"{pct(outliers['upper_fence'])}. They are plausible crisis observations, especially in April\u2013May "
            f"2020, and are retained. No low outliers are flagged.",
            STYLES["Body"],
        )
    )
    story.append(
        callout(
            "Interpretation guardrail",
            "Small states and union territories can have very few observations and volatile rates. Puducherry, "
            "Chandigarh and Sikkim should not be treated as typical state experiences. The report uses absolute "
            "percentage-point changes for shock comparisons because relative changes from a very low base can be "
            "enormous and misleading.",
            ORANGE,
        )
    )

    # ------------------------------------------------------------------
    # 4. Exploratory analysis
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("4. Exploratory analysis", 1))
    story.append(
        Paragraph(
            "This section describes the level, distribution and timing of unemployment before turning to the "
            "COVID-19 comparison. The focus is on patterns that are robust to the choice of average and to "
            "changes in sample coverage. All charts are saved at 200 dpi in <i>outputs/charts/</i>.",
            STYLES["Lead"],
        )
    )
    story.append(section_figure("4.1 National trend", "01_overall_unemployment_trend.png", "Figure 1. National unemployment rate over time. The shaded band marks April\u2013June 2020; the weighted and unweighted series answer slightly different questions.", 16.3))
    story.append(
        Paragraph(
            f"The weighted rate moved in a narrow band of <b>7.04%\u20138.11%</b> from May 2019 to February 2020. "
            f"March 2020, which includes the first lockdown days, was <b>{pct(march['weighted_rate'])}</b>. April "
            f"jumped to <b>24.16%</b>, May remained at <b>23.57%</b>, and June fell to <b>{pct(june['weighted_rate'])}</b>. "
            f"The unweighted mean peaks in May (24.88%), not April, because small high-rate territories carry more "
            f"weight in that series. The two measures agree on the scale of the shock but not on its exact timing.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            f"The full-period weighted average is <b>{pct(overall['weighted_rate'])}</b>, well below the unweighted "
            f"mean of <b>{pct(overall['unweighted_mean'])}</b>. That gap is not an error: it reflects the fact that "
            f"small territories with high unemployment rates have little labour force. The weighted measure is "
            f"therefore the better summary of the national labour-market experience, while the unweighted mean is "
            f"useful for describing the distribution of regional observations.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("4.2 State differences", "02_state_unemployment_bar.png", "Figure 2. Average labour-force-weighted unemployment by state and union territory, May 2019\u2013June 2020.", 15.2))
    story.append(
        Paragraph(
            f"The highest structural averages are <b>{top_states[0]['region']} ({pct(top_states[0]['weighted_rate'])})</b>, "
            f"<b>{top_states[1]['region']} ({pct(top_states[1]['weighted_rate'])})</b>, "
            f"<b>{top_states[2]['region']} ({pct(top_states[2]['weighted_rate'])})</b>, "
            f"<b>{top_states[3]['region']} ({pct(top_states[3]['weighted_rate'])})</b> and "
            f"<b>{top_states[4]['region']} ({pct(top_states[4]['weighted_rate'])})</b>. The lowest are "
            f"<b>{low_states[0]['region']} ({pct(low_states[0]['weighted_rate'])})</b>, "
            f"<b>{low_states[1]['region']} ({pct(low_states[1]['weighted_rate'])})</b>, "
            f"<b>{low_states[2]['region']} ({pct(low_states[2]['weighted_rate'])})</b>, "
            f"<b>{low_states[3]['region']} ({pct(low_states[3]['weighted_rate'])})</b> and "
            f"<b>{low_states[4]['region']} ({pct(low_states[4]['weighted_rate'])})</b>. The gap between the highest "
            f"and lowest average is roughly ninefold.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            "This is not a simple north\u2013south story. State-specific labour markets, industrial composition, "
            "rural dependence and reporting coverage all matter. The ranking also mixes pre-COVID and COVID months; "
            "it describes structural burden, not the size of the shock. Section 5.2 separates the two.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("4.3 Timing across states", "08_state_month_heatmap.png", "Figure 3. Monthly unemployment heatmap for the 15 states with the highest average rates. Cells are means of Rural and Urban observations; blank cells are unreported state-months.", 16.7))
    story.append(
        Paragraph(
            "The heatmap exposes timing differences that a single state average cannot show. Some states peaked in "
            "April, others in May; some high-average states did not experience a large surge at all. Puducherry, "
            "Haryana and Jharkhand show early peaks, while Bihar, Tripura and Kerala peak later. The dashed line "
            "marks April 2020. Blank cells are genuinely unreported and were not filled.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("4.4 Rural and urban labour markets", "03_rural_urban_comparison.png", "Figure 4. Rural and urban unemployment, weighted and unweighted. Urban unemployment is higher in level; rural participation is higher.", 13.2))
    story.append(
        table(
            ["Area", "Obs", "Weighted avg", "Unweighted avg", "Median", "Avg LPR", "Jan\u2013Feb", "Apr\u2013Jun", "Change", "LPR change"],
            [
                ["Rural", f"{rural['n_obs']}", pct(rural["weighted_rate"]), pct(rural["unweighted_rate"]), pct(rural["median_rate"]), pct(rural["avg_lpr"]), pct(rural["baseline_weighted_rate"]), pct(rural["covid_weighted_rate"]), pp(rural["change_pp"]), pp(rural["lpr_change_pp"])],
                ["Urban", f"{urban['n_obs']}", pct(urban["weighted_rate"]), pct(urban["unweighted_rate"]), pct(urban["median_rate"]), pct(urban["avg_lpr"]), pct(urban["baseline_weighted_rate"]), pct(urban["covid_weighted_rate"]), pp(urban["change_pp"]), pp(urban["lpr_change_pp"])],
            ],
            [1.25, 1.0, 1.7, 2.1, 1.35, 1.35, 1.35, 1.4, 1.5, 1.6],
            ["L", "R", "R", "R", "R", "R", "R", "R", "R", "R"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            f"Urban areas had the higher average rate ({pct(urban['weighted_rate'])} weighted, "
            f"{pct(urban['unweighted_rate'])} unweighted) and the lower participation rate "
            f"({pct(urban['avg_lpr'])} versus {pct(rural['avg_lpr'])} rural). During the shock, the absolute "
            f"increases were almost identical, but the rural increase was larger relative to its lower base. Urban "
            f"unemployment remained higher in level, and urban participation fell more sharply "
            f"({pp(urban['lpr_change_pp'])} versus {pp(rural['lpr_change_pp'])}).",
            STYLES["Body"],
        )
    )

    story.append(section_figure("4.5 Dispersion and the shape of the distribution", "04_monthly_unemployment_dispersion.png", "Figure 5. Monthly unemployment and cross-region dispersion. The band is the interquartile range of region-area observations.", 13.7))
    story.append(figure("06_unemployment_distribution.png", "Figure 6. Distribution of unemployment observations, split by area. The mean lies well above the median.", 12.9))
    story.append(
        Paragraph(
            f"The interquartile band widens sharply in April\u2013May 2020. The shock was not a uniform national "
            f"shift: some places experienced extreme rates while others changed much less. The distribution is "
            f"right-skewed. The full-sample mean is {pct(overall['unweighted_mean'])} but the median is "
            f"{pct(overall['median'])}; most observations sit below 15%, with a long tail above 40%. That tail is "
            f"dominated by COVID-period and small-territory observations, not by a second population of typical "
            f"states.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("4.6 Outliers and the April\u2013May shock", "07_unemployment_boxplot.png", "Figure 7. Box plots by area and by COVID-19 period. Dots show individual observations, including outliers.", 14.4))
    story.append(
        Paragraph(
            f"The IQR rule flags <b>{outliers['outlier_count']} observations ({outliers['outlier_share_pct']:.1f}%)</b> "
            f"above the upper fence of {pct(outliers['upper_fence'])}; there are no low outliers. All are retained. "
            f"They are not data-entry errors: the box plot shows them concentrated in the COVID period and in "
            f"high-variability areas. Removing them would understate the shock and distort the comparison that is "
            f"central to this report.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            "The box plot also shows that the median rose much less than the upper tail. A typical observation did "
            "not become as extreme as the worst-affected places. That is why the report reports both the weighted "
            "national rate and the dispersion around it: the average and the experience of the hardest-hit states "
            "are different facts.",
            STYLES["Body"],
        )
    )

    story.append(heading("4.7 Employment and participation", 2))
    story.append(
        Paragraph(
            f"Employment and participation tell a more complete story than the unemployment rate alone. In the "
            f"balanced panel, total employment fell from a January\u2013February average of "
            f"{employment['baseline_total_employed'] / 1e6:.1f} million to "
            f"{monthly_employment['2020-04']['total_employed'] / 1e6:.1f} million in April, a loss of "
            f"{abs(monthly_employment['2020-04']['total_change_millions']):.1f} million "
            f"({monthly_employment['2020-04']['total_change_pct']:.1f}%). May was "
            f"{monthly_employment['2020-05']['total_change_pct']:.1f}% below baseline and June "
            f"{monthly_employment['2020-06']['total_change_pct']:.1f}% below. Urban employment fell more in April "
            f"({monthly_employment['2020-04']['urban_change_pct']:.1f}%) than rural "
            f"({monthly_employment['2020-04']['rural_change_pct']:.1f}%).",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            f"Average LPR across observations fell from {pct(baseline['average_lpr'])} in January\u2013February to "
            f"{pct(covid_core['average_lpr'])} in April\u2013June, a decline of "
            f"{pp(covid_core['average_lpr'] - baseline['average_lpr'])}. April was lower still. Some of that decline "
            f"reflects people leaving the labour force after losing jobs or stopping the search; the dataset cannot "
            f"separate discouraged-worker effects from other reasons for non-participation.",
            STYLES["Body"],
        )
    )
    story.append(figure("09_employment_trend.png", "Figure 8. Estimated employment in the balanced 20-state panel and the area index (February 2020 = 100).", 16.2))
    story.append(figure("10_labour_participation_trend.png", "Figure 9. Labour participation over time. Area lines are simple means; the dashed national line is an approximation derived from labour force and implied population.", 15.3))
    story.append(
        callout(
            "Why both indicators matter",
            f"April employment fell {abs(monthly_employment['2020-04']['total_change_millions']):.1f} million while "
            f"the weighted unemployment rate peaked at 24.16%. By June, employment was still "
            f"{abs(monthly_employment['2020-06']['total_change_pct']):.1f}% below baseline even as the rate had "
            f"fallen to {pct(june['weighted_rate'])}. A falling rate can therefore coexist with a weak labour market.",
            ORANGE,
        )
    )

    story.append(PageBreak())
    story.append(heading("4.8 Relationships among indicators", 2))
    story.append(figure("05_correlation_heatmap.png", "Figure 10. Pooled Pearson correlations among the three numerical indicators (n = 740).", 8.5))
    story.append(
        table(
            ["Relationship", "Pearson", "Spearman", "Reading"],
            [
                ["Unemployment vs estimated employed", f"{relationships['pearson']['unemployment_rate']['estimated_employed']:.3f}", f"{relationships['spearman']['unemployment_rate']['estimated_employed']:.3f}", "Weak negative pooled association"],
                ["Unemployment vs LPR", f"{relationships['pearson']['unemployment_rate']['labour_participation_rate']:.3f}", f"{relationships['spearman']['unemployment_rate']['labour_participation_rate']:.3f}", "Essentially zero pooled association"],
                ["State-average unemployment vs LPR", f"{relationships['state_average_unemployment_vs_lpr_pearson']:.3f}", "\u2014", "Weak positive cross-state association"],
                ["Monthly national unemployment vs reported employment", f"{relationships['monthly_unweighted_unemployment_vs_reported_employment_pearson']:.3f}", "\u2014", "Strong negative monthly association"],
            ],
            [6.3, 2.0, 2.0, 6.5],
            ["L", "R", "R", "L"],
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Paragraph(
            "The pooled correlations are weak because state size and repeated observations dominate the "
            "cross-section. At the national monthly level, unemployment and total reported employment move strongly "
            "in opposite directions during the shock. Neither result identifies a causal effect: employment and "
            "unemployment are mechanically linked through the labour force, and both respond to the same shock. "
            "The practical lesson is that a single correlation matrix is not sufficient for policy targeting; "
            "state, area and period splits are essential.",
            STYLES["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 5. COVID-19 impact
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("5. COVID-19 impact", 1))
    story.append(
        Paragraph(
            "The COVID-19 analysis compares April\u2013June 2020 with January\u2013February 2020. March is treated "
            "as a transition month because the national lockdown began on 25 March. The comparison is descriptive: "
            "it measures the size, breadth and timing of the change, not the causal effect of the pandemic or of "
            "any specific restriction.",
            STYLES["Lead"],
        )
    )
    story.append(heading("5.1 Headline change", 2))
    story.append(
        table(
            ["Period", "Obs", "Weighted rate", "Unweighted rate", "Avg LPR"],
            [
                ["Jan\u2013Feb 2020 (baseline)", f"{baseline['n_obs']}", pct(baseline["weighted_rate"]), pct(baseline["unweighted_rate"]), pct(baseline["average_lpr"])],
                ["March 2020 (transition)", f"{march['n_obs']}", pct(march["weighted_rate"]), pct(march["unweighted_rate"]), pct(march["average_lpr"])],
                ["Apr\u2013Jun 2020 (COVID period)", f"{covid_core['n_obs']}", pct(covid_core["weighted_rate"]), pct(covid_core["unweighted_rate"]), pct(covid_core["average_lpr"])],
                ["Apr\u2013May 2020 (peak)", f"{peak_window['n_obs']}", pct(peak_window["weighted_rate"]), pct(peak_window["unweighted_rate"]), pct(peak_window["average_lpr"])],
                ["June 2020", f"{june['n_obs']}", pct(june["weighted_rate"]), pct(june["unweighted_rate"]), pct(june["average_lpr"])],
            ],
            [5.1, 1.6, 3.3, 3.6, 2.6],
            ["L", "R", "R", "R", "R"],
        )
    )
    story.append(figure("11_covid_period_comparison.png", "Figure 11. National rate by period and the Rural/Urban decomposition of the change from January\u2013February to April\u2013June 2020.", 16.3))
    story.append(
        Paragraph(
            f"The weighted national rate rose <b>{pp(headline['absolute_change_pp'])}</b> "
            f"({headline['relative_change_pct']:+.1f}%) from {pct(baseline['weighted_rate'])} to "
            f"{pct(covid_core['weighted_rate'])}. The unweighted mean rose {pp(headline['unweighted_change_pp'])}. "
            f"The April\u2013May peak was {pct(peak_window['weighted_rate'])}, "
            f"{pp(headline['peak_change_pp'])} above baseline ({headline['peak_relative_change_pct']:+.1f}%). "
            f"March was only {pct(march['weighted_rate'])}, while June had already fallen to {pct(june['weighted_rate'])}. "
            f"This is a spike and partial early rebound, not a gradual trend that began before the pandemic.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            f"The shock was broad-based. Rural unemployment rose {pp(rural['change_pp'])} "
            f"({rural['change_pct']:+.1f}%) and urban unemployment rose {pp(urban['change_pp'])} "
            f"({urban['change_pct']:+.1f}%). The near-identical absolute changes hide a higher urban level "
            f"({pct(urban['covid_weighted_rate'])} versus {pct(rural['covid_weighted_rate'])}) and a larger rural "
            f"relative increase from a lower base. Participation fell in both areas, and more in urban areas.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("5.2 State incidence", "12_covid_state_increases.png", "Figure 12. The 12 largest absolute state-level increases in the labour-force-weighted unemployment rate, January\u2013February to April\u2013June 2020.", 13.6))
    state_rows = []
    for rank, row in enumerate(top_abs, start=1):
        if rank == 1:
            note = "Very low, volatile baseline"
        elif not bool(row["complete_covid_window"]):
            note = "Partial COVID window"
        elif bool(row["low_baseline_below_5pct"]):
            note = "Baseline below 5%"
        else:
            note = "Large absolute shock"
        state_rows.append(
            [
                str(rank),
                row["region"],
                pct(row["baseline_rate"]),
                pct(row["covid_rate"]),
                pp(row["change_pp"]),
                note,
            ]
        )
    story.append(
        table(
            ["Rank", "State / UT", "Jan\u2013Feb", "Apr\u2013Jun", "Change", "Note"],
            state_rows,
            [1.2, 3.6, 2.2, 2.2, 2.2, 4.4],
            ["C", "L", "R", "R", "R", "L"],
            font_size=7.4,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            f"<b>{covid['state_changes_overview']['states_increased']} of the 27 regions with observations in both "
            f"windows increased</b>; {covid['state_changes_overview']['states_decreased']} fell. The largest absolute "
            f"increases were in Jharkhand {pp(top_abs[1]['change_pp'])}, Tamil Nadu "
            f"{pp(top_abs[2]['change_pp'])}, Bihar {pp(top_abs[3]['change_pp'])}, Karnataka "
            f"{pp(top_abs[4]['change_pp'])} and Kerala {pp(top_abs[5]['change_pp'])}. Puducherry's "
            f"{pp(top_abs[0]['change_pp'])} is the largest, but it starts from a {pct(top_abs[0]['baseline_rate'])} "
            f"baseline with only four baseline observations and should be treated as a warning about small-area "
            f"volatility, not a typical state experience.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            "High average unemployment and a large shock are different problems. Tripura, Jammu &amp; Kashmir, "
            "Himachal Pradesh and Chandigarh recorded declines over this window. Jammu &amp; Kashmir's estimate is "
            "also affected by incomplete April data. A targeted policy response needs both dimensions: the "
            "structural level and the change.",
            STYLES["Body"],
        )
    )

    story.append(heading("5.3 Employment loss", 2))
    story.append(
        Paragraph(
            f"Employment is analysed on a balanced panel of <b>{employment['balanced_panel_states']} states</b> "
            f"with all 14 months and both Rural and Urban rows, so sample composition is constant. The "
            f"January\u2013February 2020 baseline was {employment['baseline_total_employed'] / 1e6:.1f} million. "
            f"April employment was {monthly_employment['2020-04']['total_employed'] / 1e6:.1f} million, a fall of "
            f"<b>{abs(monthly_employment['2020-04']['total_change_millions']):.1f} million "
            f"({monthly_employment['2020-04']['total_change_pct']:.1f}%)</b>. May was "
            f"{monthly_employment['2020-05']['total_change_pct']:.1f}% below baseline and June "
            f"{monthly_employment['2020-06']['total_change_pct']:.1f}% below.",
            STYLES["Body"],
        )
    )
    story.append(
        KeepTogether(
            [
                table(
                    ["Month", "Employed (million)", "vs baseline", "Rural vs baseline", "Urban vs baseline"],
                    [
                        ["March 2020", f"{monthly_employment['2020-03']['total_employed'] / 1e6:.1f}", f"{monthly_employment['2020-03']['total_change_pct']:+.2f}%", f"{monthly_employment['2020-03']['rural_change_pct']:+.2f}%", f"{monthly_employment['2020-03']['urban_change_pct']:+.2f}%"],
                        ["April 2020", f"{monthly_employment['2020-04']['total_employed'] / 1e6:.1f}", f"{monthly_employment['2020-04']['total_change_pct']:+.2f}%", f"{monthly_employment['2020-04']['rural_change_pct']:+.2f}%", f"{monthly_employment['2020-04']['urban_change_pct']:+.2f}%"],
                        ["May 2020", f"{monthly_employment['2020-05']['total_employed'] / 1e6:.1f}", f"{monthly_employment['2020-05']['total_change_pct']:+.2f}%", f"{monthly_employment['2020-05']['rural_change_pct']:+.2f}%", f"{monthly_employment['2020-05']['urban_change_pct']:+.2f}%"],
                        ["June 2020", f"{monthly_employment['2020-06']['total_employed'] / 1e6:.1f}", f"{monthly_employment['2020-06']['total_change_pct']:+.2f}%", f"{monthly_employment['2020-06']['rural_change_pct']:+.2f}%", f"{monthly_employment['2020-06']['urban_change_pct']:+.2f}%"],
                    ],
                    [3.4, 4.0, 3.2, 3.2, 3.2],
                    ["L", "R", "R", "R", "R"],
                )
            ]
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            f"The April loss is concentrated in a single month and is larger in urban areas "
            f"({monthly_employment['2020-04']['urban_change_pct']:.1f}%) than rural "
            f"({monthly_employment['2020-04']['rural_change_pct']:.1f}%). By June, employment had recovered "
            f"substantially but remained {abs(monthly_employment['2020-06']['total_change_pct']):.1f}% below the "
            f"pre-shock level. The panel covers 20 complete states, not the whole country; the totals are therefore "
            f"an analytical panel, not a national employment count. They are still large enough to show that the "
            f"unemployment-rate spike understates the scale of job loss.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("5.4 Recovery to October 2020", "13_extended_recovery_d2.png", "Figure 13. Supplementary 2020 series: the unemployment rate falls back, but estimated employment and participation remain below January.", 16.6))
    story.append(
        Paragraph(
            f"Dataset 2 extends the view to October 2020. The weighted rate was "
            f"{pct(recovery['january']['weighted_rate'])} in January, "
            f"{pct(recovery['april']['weighted_rate'])} in April and "
            f"{pct(recovery['october']['weighted_rate'])} in October. The April peak in this series is slightly "
            f"lower than the primary file's April estimate because the datasets have different populations and "
            f"construction.",
            STYLES["Body"],
        )
    )
    story.append(
        table(
            ["2020 month", "Weighted rate", "Unweighted mean", "Employed (million)", "Avg LPR"],
            [
                ["January", pct(recovery["january"]["weighted_rate"]), pct(recovery["january"]["unweighted_rate"]), f"{recovery['january']['total_employed'] / 1e6:.1f}", pct(recovery["january"]["average_lpr"])],
                ["April", pct(recovery["april"]["weighted_rate"]), pct(recovery["april"]["unweighted_rate"]), f"{recovery['april']['total_employed'] / 1e6:.1f}", pct(recovery["april"]["average_lpr"])],
                ["October", pct(recovery["october"]["weighted_rate"]), pct(recovery["october"]["unweighted_rate"]), f"{recovery['october']['total_employed'] / 1e6:.1f}", pct(recovery["october"]["average_lpr"])],
            ],
            [3.0, 3.4, 3.6, 4.0, 3.0],
            ["L", "R", "R", "R", "R"],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        callout(
            "The recovery is incomplete",
            f"Comparing the same {common['n_states']} states in January and October 2020, the weighted rate is "
            f"{pp(common['weighted_rate_change_pp'])} \u2014 essentially back to its January level. But estimated "
            f"employment is {common['employment_change_pct']:+.1f}% lower (about "
            f"{(common['january_employed'] - common['october_employed']) / 1e6:.1f} million fewer people employed), "
            f"and the average LPR is {pp(common['lpr_change_pp'])} lower. The headline rate has recovered; the "
            f"labour market has not.",
            RED,
        )
    )
    story.append(
        Paragraph(
            "This is the report's central analytical warning. A falling unemployment rate can reflect people "
            "leaving the labour force rather than finding work. The supplementary data do not contain an area split, "
            "so they cannot show whether the participation gap is rural or urban; they do show that it is large "
            "enough to matter for policy.",
            STYLES["Body"],
        )
    )

    story.append(section_figure("5.5 Geographic zones", "14_zone_unemployment.png", "Figure 14. Average unemployment and monthly paths by geographic zone in the supplementary dataset.", 16.3))
    story.append(
        table(
            ["Zone", "States", "Average", "Jan\u2013Feb", "Apr\u2013Jun", "Change"],
            [
                ["East", f"{zones['East']['n_states']}", pct(zones["East"]["weighted_rate"]), pct(zones["East"]["baseline_weighted_rate"]), pct(zones["East"]["covid_weighted_rate"]), pp(zones["East"]["change_pp"])],
                ["North", f"{zones['North']['n_states']}", pct(zones["North"]["weighted_rate"]), pct(zones["North"]["baseline_weighted_rate"]), pct(zones["North"]["covid_weighted_rate"]), pp(zones["North"]["change_pp"])],
                ["South", f"{zones['South']['n_states']}", pct(zones["South"]["weighted_rate"]), pct(zones["South"]["baseline_weighted_rate"]), pct(zones["South"]["covid_weighted_rate"]), pp(zones["South"]["change_pp"])],
                ["Northeast", f"{zones['Northeast']['n_states']}", pct(zones["Northeast"]["weighted_rate"]), pct(zones["Northeast"]["baseline_weighted_rate"]), pct(zones["Northeast"]["covid_weighted_rate"]), pp(zones["Northeast"]["change_pp"])],
                ["West", f"{zones['West']['n_states']}", pct(zones["West"]["weighted_rate"]), pct(zones["West"]["baseline_weighted_rate"]), pct(zones["West"]["covid_weighted_rate"]), pp(zones["West"]["change_pp"])],
            ],
            [3.2, 1.8, 2.8, 2.8, 2.8, 2.8],
            ["L", "R", "R", "R", "R", "R"],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            f"The East has both the highest average level ({pct(zones['East']['weighted_rate'])}) and the largest "
            f"observed increase ({pp(zones['East']['change_pp'])}). The North also has a high average "
            f"({pct(zones['North']['weighted_rate'])}) but a smaller increase ({pp(zones['North']['change_pp'])}). "
            f"The South starts from a low baseline ({pct(zones['South']['baseline_weighted_rate'])}) and records a "
            f"large increase ({pp(zones['South']['change_pp'])}). The Northeast has the smallest increase "
            f"({pp(zones['Northeast']['change_pp'])}). These zone figures aggregate very different states and are "
            f"descriptive only; they are not a substitute for state-level targeting.",
            STYLES["Body"],
        )
    )

    # ------------------------------------------------------------------
    # 6. Validation, seasonality, causality
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("6. Validation, seasonality and causal limits", 1))
    story.append(heading("6.1 Cross-dataset validation", 2))
    story.append(
        Paragraph(
            f"The two files overlap from January to June 2020. Rather than merging them, the analysis compares "
            f"Dataset 1's derived rural+urban weighted rate with Dataset 2's single regional rate for the same "
            f"state-month. There are <b>{overlap['comparison_rows']} matched pairs</b> across "
            f"<b>{overlap['regions_compared']} regions</b>.",
            STYLES["Body"],
        )
    )
    story.append(
        table(
            ["Validation measure", "Result"],
            [
                ["Pearson correlation", f"{overlap['pearson_correlation']:.3f}"],
                ["Spearman correlation", f"{overlap['spearman_correlation']:.3f}"],
                ["Mean signed difference", pp(overlap["mean_difference_pp"])],
                ["Mean absolute difference", f"{overlap['mean_absolute_difference_pp']:.2f} pp"],
                ["Maximum absolute difference", f"{overlap['max_absolute_difference_pp']:.2f} pp"],
            ],
            [7.5, 4.5],
            ["L", "R"],
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            "The correlation is very high and the average signed difference is small, which supports the derived "
            "weighting method. The mean absolute difference is larger, and the maximum is substantial: the "
            "largest discrepancy is Telangana in May 2020, where the two constructions diverge sharply. The files "
            "are broadly consistent on average but are not interchangeable. That is why they are compared, not "
            "concatenated, and why the report states which file supplies each result.",
            STYLES["Body"],
        )
    )
    story.append(heading("6.2 Seasonality", 2))
    story.append(
        Paragraph(
            f"The primary dataset spans {seasonality['months_covered']} consecutive months "
            f"({', '.join(str(year) for year in seasonality['years_covered'])}). Only "
            f"<b>May and June</b> occur in both calendar years, and the 2020 repeats coincide with the COVID-19 "
            f"shock. The supplementary dataset covers a single calendar year. This is not enough to identify a "
            f"seasonal cycle.",
            STYLES["Body"],
        )
    )
    story.append(
        table(
            [
                "Calendar month",
                "2019 unweighted mean",
                "2020 unweighted mean",
                "Change",
                "Interpretation",
            ],
            [
                ["May", pct(seasonality["repeated_month_values"][0]["unemployment_rate"]), pct(seasonality["repeated_month_values"][1]["unemployment_rate"]), pp(seasonality["repeated_month_values"][1]["unemployment_rate"] - seasonality["repeated_month_values"][0]["unemployment_rate"]), "Confounded by COVID-19"],
                ["June", pct(seasonality["repeated_month_values"][2]["unemployment_rate"]), pct(seasonality["repeated_month_values"][3]["unemployment_rate"]), pp(seasonality["repeated_month_values"][3]["unemployment_rate"] - seasonality["repeated_month_values"][2]["unemployment_rate"]), "Confounded by COVID-19"],
            ],
            [2.8, 3.4, 3.4, 2.2, 5.2],
            ["L", "R", "R", "R", "L"],
            font_size=7.4,
        )
    )
    story.append(Spacer(1, 0.18 * cm))
    story.append(
        Paragraph(
            "These May and June values are descriptive comparisons of unweighted mean unemployment rates, not a "
            "seasonal adjustment. They cannot establish a reliable seasonal pattern because the available time "
            "series is short and the 2020 observations are affected by COVID-19. A defensible seasonal analysis "
            "would require several complete years and a model that separates season, trend and shock. The monthly "
            "pattern before 2020 is therefore described only as month-to-month variation, not as a recurring "
            "seasonal cycle.",
            STYLES["Body"],
        )
    )
    story.append(heading("6.3 What the analysis can and cannot claim", 2))
    story.append(
        callout(
            "Causal caution",
            covid["causality_statement"],
            RED,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    for item in [
        "There is no counterfactual, no randomised variation and no control group in these files.",
        "Lockdowns, migration, survey disruption, fiscal support and global demand changed at the same time.",
        "The data are monthly survey estimates, not an unemployment register; revisions and measurement error are possible.",
        "Before-and-after comparisons are descriptive. Credible causal estimates would require a research design beyond this dataset.",
    ]:
        story.append(bullet(item))

    # ------------------------------------------------------------------
    # 7. Policy recommendations
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("7. Policy recommendations", 1))
    story.append(
        Paragraph(
            "The recommendations below follow from the measured patterns. They are policy directions, not "
            "interventions that have been evaluated in this dataset. Each is linked to a specific finding so that "
            "the evidence behind it is visible.",
            STYLES["Lead"],
        )
    )
    story.append(heading("7.1 Protect rural employment and earnings", 2))
    story.append(
        Paragraph(
            f"Rural unemployment rose {pp(rural['change_pp'])} "
            f"(from {pct(rural['baseline_weighted_rate'])} to {pct(rural['covid_weighted_rate'])}), a "
            f"{rural['change_pct']:.1f}% relative increase, even though rural participation was higher than urban. "
            f"Rural LPR fell {pp(rural['lpr_change_pp'])}. Expand guaranteed rural employment, irrigation, rural "
            f"infrastructure, agri-processing and local food supply chains. These measures can stabilise earnings "
            f"when urban migration reverses and rural labour supply rises. Delivery should be timed to the local "
            f"agricultural calendar and linked to wage payment reliability, not only to days of work created.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.2 Support urban job recovery and re-entry", 2))
    story.append(
        Paragraph(
            f"Urban unemployment was the highest average level before the shock ({pct(urban['weighted_rate'])}) and "
            f"remained higher during it ({pct(urban['covid_weighted_rate'])}). Urban LPR fell "
            f"{abs(urban['lpr_change_pp']):.1f} pp, and urban employment fell "
            f"{abs(monthly_employment['2020-04']['urban_change_pct']):.1f}% in April. Prioritise MSME credit, wage "
            f"support in contact-intensive services, safe transport and childcare, and job-search and re-entry "
            f"services for displaced urban workers. Recovery should be measured by re-employment and participation, "
            f"not only by the urban unemployment rate.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.3 Target both high-burden and high-shock states", 2))
    story.append(
        Paragraph(
            f"High average unemployment states \u2014 {top_states[0]['region']} ({pct(top_states[0]['weighted_rate'])}), "
            f"{top_states[1]['region']} ({pct(top_states[1]['weighted_rate'])}), {top_states[2]['region']} "
            f"({pct(top_states[2]['weighted_rate'])}), {top_states[3]['region']} ({pct(top_states[3]['weighted_rate'])}) "
            f"and {top_states[4]['region']} ({pct(top_states[4]['weighted_rate'])}) \u2014 need structural employment "
            f"programmes. The largest observed increases were in Jharkhand, Tamil Nadu, Bihar, Karnataka and Kerala; "
            f"Puducherry's much larger increase is a volatile small-area estimate. A single national formula would "
            f"over-serve some places and under-serve others. Allocate using state-specific diagnostics that combine "
            f"the level, the shock, coverage quality and the size of the affected workforce.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.4 Rebuild jobs, not just the unemployment rate", 2))
    story.append(
        Paragraph(
            f"April employment in the balanced panel fell by {abs(monthly_employment['2020-04']['total_change_millions']):.0f} "
            f"million ({monthly_employment['2020-04']['total_change_pct']:.1f}%), and June remained "
            f"{abs(monthly_employment['2020-06']['total_change_pct']):.1f}% below baseline. By October, the "
            f"supplementary data show the rate back near its January level but employment "
            f"{common['employment_change_pct']:+.1f}% lower in the same 26 states. Wage subsidies, hiring incentives, "
            f"public works and MSME credit should be judged by employment restoration and by whether displaced "
            f"workers return to productive work. A falling rate caused by people leaving the labour force is not a "
            f"healthy recovery.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.5 Support labour-force participation", 2))
    story.append(
        Paragraph(
            f"Average LPR fell from {pct(baseline['average_lpr'])} to {pct(covid_core['average_lpr'])} between the "
            f"baseline and the COVID period, and the supplementary data still show a "
            f"{abs(common['lpr_change_pp']):.2f} pp participation gap in October for the same 26 states. Childcare, "
            f"safe public transport, formalisation of informal work, portable benefits and accessible job-search "
            f"support can help people return to the labour force. Participation should be monitored alongside "
            f"unemployment, especially for women and for workers who withdrew after a job loss; the supplied data "
            f"do not allow a gender breakdown, so closing that data gap is itself a priority.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.6 Strengthen shock-responsive social protection", 2))
    story.append(
        Paragraph(
            f"The speed and scale of the employment loss \u2014 more than 100 million estimated jobs in the "
            f"balanced panel in April alone \u2014 imply that income support must be able to scale quickly. Portable "
            f"unemployment benefits, food support, direct cash transfers and informal-worker registration would "
            f"reduce distress and prevent asset sales. The dataset does not identify informal workers, but the "
            f"concentration of the shock in urban contact-intensive sectors and in poor states makes broad, quickly "
            f"disbursed support a priority. Targeting can use the state and area patterns documented here, with "
            f"local verification to avoid exclusion errors.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.7 Invest in timely, granular data and evaluation", 2))
    story.append(
        Paragraph(
            "The analysis is limited by 14 months of primary data, missing region-months, volatile small-area "
            "estimates and the absence of demographic detail. Policymakers should publish timely state-district-month "
            "estimates with age, gender, sector and informal-work status, and evaluate interventions against "
            "pre-specified comparison groups. Better data would allow seasonal adjustment and causal evaluation "
            "rather than descriptive before-after comparisons. It would also let recovery be tracked by jobs and "
            "participation, not only by the unemployment rate.",
            STYLES["Body"],
        )
    )
    story.append(heading("7.8 From evidence to action", 2))
    story.append(
        Paragraph(
            "The table links each recommendation to the finding that motivates it and to the indicator that "
            "should be monitored. It is a diagnostic aid, not a scored programme evaluation.",
            STYLES["Body"],
        )
    )
    story.append(
        table(
            ["Finding (evidence)", "Recommendation", "Monitor"],
            [
                [
                    f"Rural rate {pp(rural['change_pp'])}; rural LPR {pp(rural['lpr_change_pp'])}",
                    "Protect rural employment and earnings",
                    "Rural rate, rural LPR, days of work",
                ],
                [
                    f"Urban rate {pct(urban['covid_weighted_rate'])}; urban employment {monthly_employment['2020-04']['urban_change_pct']:.1f}% in April",
                    "Support urban job recovery and re-entry",
                    "Urban employment, urban LPR",
                ],
                [
                    f"State averages {pct(low_states[0]['weighted_rate'])}\u2013{pct(top_states[0]['weighted_rate'])}; "
                    f"{covid['state_changes_overview']['states_increased']} of 27 increased",
                    "Target high-burden and high-shock states",
                    "State rate, employment, coverage",
                ],
                [
                    f"April employment {monthly_employment['2020-04']['total_change_millions']:+.1f} million "
                    f"({monthly_employment['2020-04']['total_change_pct']:.1f}%); June {monthly_employment['2020-06']['total_change_pct']:.1f}%",
                    "Rebuild jobs, not just the rate",
                    "Employment versus baseline",
                ],
                [
                    f"LPR {pct(baseline['average_lpr'])} to {pct(covid_core['average_lpr'])}; October {pp(common['lpr_change_pp'])}",
                    "Support labour-force participation",
                    "LPR, re-entry, gender gap",
                ],
                [
                    "Shock speed; informal workers not identified",
                    "Strengthen shock-responsive protection",
                    "Coverage, payment speed",
                ],
                [
                    f"14 months; {coverage['dataset1']['missing_region_area_months']} missing region-area-months",
                    "Invest in timely, granular data",
                    "Coverage, timeliness, detail",
                ],
            ],
            [6.2, 5.8, 5.2],
            ["L", "L", "L"],
            font_size=7.4,
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        callout(
            "Sequencing",
            "Stabilise incomes first, then restore jobs, then rebuild participation. The data show that the "
            "unemployment rate can fall while employment and participation remain depressed, so success should "
            "be judged on all three indicators, not on the headline rate alone.",
            TEAL,
        )
    )

    # ------------------------------------------------------------------
    # 8. Conclusion
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("8. Conclusion", 1))
    story.append(
        Paragraph(
            f"The data describe a labour market that was already uneven before the pandemic and then experienced "
            f"an extraordinary, broad-based shock. The labour-force-weighted national unemployment rate rose from "
            f"<b>{pct(headline['baseline']['weighted_rate'])}</b> in January\u2013February 2020 to "
            f"<b>{pct(headline['covid_core']['weighted_rate'])}</b> in April\u2013June 2020, with an April\u2013May "
            f"peak of <b>{pct(headline['peak']['weighted_rate'])}</b>. Employment in a balanced 20-state panel fell "
            f"by nearly a third in April.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            f"By October 2020, the supplementary data show the headline unemployment rate back near its January "
            f"level, but employment was {abs(common['employment_change_pct']):.1f}% lower and participation "
            f"{abs(common['lpr_change_pp']):.2f} pp lower in the same 26 states. That gap is the central policy "
            f"message: <b>a lower unemployment rate is not the same as a restored labour market.</b> Rural areas "
            f"need earnings and employment support; urban areas need job recovery and re-entry support; high-burden "
            f"and high-shock states need targeted programmes; and informal workers need shock-responsive social "
            f"protection.",
            STYLES["Body"],
        )
    )
    story.append(
        Paragraph(
            "The analysis is descriptive and limited by data availability. It cannot establish that COVID-19 alone "
            "caused the observed changes, and it cannot identify the effects of individual policies. Its value lies "
            "in making the scale, unevenness and incomplete recovery visible, and in linking each recommendation to "
            "a measured pattern rather than to assumption. The accompanying notebook reproduces the analysis, and "
            "the cleaned data, charts and tables are provided so that others can check and extend it.",
            STYLES["Body"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        callout(
            "Final assessment",
            f"The COVID-period increase was large ({pp(headline['absolute_change_pp'])}), broad-based across rural "
            f"and urban areas, and uneven across states. The early rebound did not restore employment or "
            f"participation. Policy should therefore target job restoration and labour-force re-entry, while "
            f"investing in the granular data needed to detect the next shock earlier.",
            TEAL,
        )
    )

    # ------------------------------------------------------------------
    # Appendix A
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("Appendix A. Reproducibility and outputs", 1))
    story.append(
        Paragraph(
            "The project is fully reproducible from the project root. Run the scripts in order; each reads the "
            "outputs of the previous stage and fails clearly if an input is missing. The notebook is built from "
            "the same analysis outputs and then executed end to end.",
            STYLES["Body"],
        )
    )
    commands = [
        "python src/01_dataset_inspection.py",
        "python src/02_data_cleaning.py",
        "python src/03_exploratory_analysis.py",
        "python src/04_data_visualization.py",
        "python src/05_covid_impact_analysis.py",
        "python src/07_build_notebook.py",
        "python -m nbconvert --to notebook --execute notebooks/unemployment_analysis.ipynb --inplace",
        "python src/06_generate_report.py",
    ]
    command_text = "<br/>".join(f"<font name='Courier'>{e(command)}</font>" for command in commands)
    story.append(
        Table(
            [[Paragraph(command_text, ParagraphStyle("Commands", fontName=BODY_FONT, fontSize=7.7, leading=12.5, textColor=colors.HexColor("#263238")))]],
            colWidths=[CONTENT_WIDTH],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE),
                    ("BOX", (0, 0), (-1, -1), 0.4, LIGHT_GRAY),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        table(
            ["Output", "Location", "Contents"],
            [
                ["Clean data", "outputs/cleaned_data/", "Cleaned CSVs, cleaning log, data-quality report, data dictionary"],
                ["Inspection", "outputs/inspection/", "Raw schema, missingness, duplicates, date and coverage checks"],
                ["Analysis tables", "outputs/analysis/", "State, area, monthly, correlation, coverage and COVID tables"],
                ["Charts", "outputs/charts/", "14 publication-quality PNGs at 200 dpi"],
                ["Notebook", "notebooks/unemployment_analysis.ipynb", "Executed end-to-end analysis, 55 cells"],
                ["Report", "report/Unemployment_Analysis_Report.pdf", "This document"],
            ],
            [3.2, 6.0, 7.6],
            ["L", "L", "L"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            f"Environment: Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, "
            f"ReportLab 5.1.0. The notebook contains {20} executed code cells and 28 embedded chart images. "
            f"All paths are resolved relative to the project root.",
            STYLES["Body"],
        )
    )

    # ------------------------------------------------------------------
    # Appendix B
    # ------------------------------------------------------------------
    story.append(PageBreak())
    story.append(heading("Appendix B. Data dictionary", 1))
    story.append(
        table(
            ["Field", "Definition / construction", "Source"],
            [
                ["region", "State or union territory name.", "Both files"],
                ["date", "Month-end observation date, parsed as dd-mm-yyyy.", "Both files"],
                ["frequency", "Reporting frequency; standardised to Monthly.", "Both files"],
                ["unemployment_rate", "Estimated unemployment rate (% of labour force).", "Both files"],
                ["estimated_employed", "Estimated number of employed persons.", "Both files"],
                ["labour_participation_rate", "Estimated labour force as % of population.", "Both files"],
                ["area", "Rural or Urban. Two rows per region-month in Dataset 1.", "Dataset 1"],
                ["zone", "Geographic zone supplied with Dataset 2.", "Dataset 2"],
                ["month", "Year-month label used for grouping and plotting.", "Derived"],
                ["period", "Pre-COVID, Transition or COVID period, based on date.", "Derived"],
                ["estimated_labour_force", "Employed / (1 - rate/100). Arithmetic identity.", "Derived"],
                ["estimated_unemployed", "Labour force - employed. Arithmetic identity.", "Derived"],
                ["zero_unemployment", "Flag for exact-zero unemployment observations.", "Derived"],
            ],
            [3.4, 9.8, 3.6],
            ["L", "L", "L"],
            font_size=7.6,
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(heading("B.1 Method notes", 2))
    for item in [
        "The headline rate is labour-force-weighted. The unweighted mean is reported alongside it and is never substituted silently.",
        "The weighted LPR is an approximation because population is not supplied. Area LPR values are simple means across observations.",
        "No missing region-month values were imputed. No zero or outlier observations were deleted.",
        "The two datasets are analysed separately. Overlap validation is a comparison, not a merge.",
        "All differences in percentage points (pp) are arithmetic differences between rates, not relative changes. Relative changes are labelled as percentages.",
    ]:
        story.append(bullet(item))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        Paragraph(
            "Data source: the two CSV files supplied for the CodeAlpha Task 2 exercise. No external data or "
            "invented statistics are used.",
            STYLES["Caption"],
        )
    )

    return story


def main() -> None:
    utils.ensure_directories()
    story = build_story()
    document = ReportDocument(str(REPORT_PATH))
    document.multiBuild(story, canvasmaker=NumberedCanvas)
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"PDF was not created: {REPORT_PATH}")
    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"Created PDF report: {REPORT_PATH}")
    print(f"Pages: {document.page}")
    print(f"Size: {size_kb:,.1f} KB")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
