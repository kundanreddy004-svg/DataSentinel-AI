"""
PDF audit report generation for DataSentinel AI.

Public API:
    generate_audit_pdf(
        risk_result=None,
        leakage_report=None,
        drift_result=None,
        dataset_name="Uploaded dataset",
    ) -> bytes

The function returns PDF bytes suitable for Streamlit st.download_button(data=...).
It handles pandas DataFrames explicitly and never evaluates a DataFrame as a boolean.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


def _is_missing(value: Any) -> bool:
    """Return True for scalar missing values without ambiguous array/DataFrame truth tests."""
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, type(pd.NA))):
            return bool(result)
        # pd.isna may return an array/Series for non-scalar values.
        return False
    except (TypeError, ValueError):
        return False


def _display(value: Any, limit: int = 500) -> str:
    """Convert values to safe, readable text for PDF cells."""
    if _is_missing(value):
        return "—"
    if isinstance(value, (dict, list, tuple, set)):
        text = str(value)
    elif isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    """Escape text so characters such as &, < and > don't break ReportLab paragraphs."""
    from xml.sax.saxutils import escape

    return Paragraph(escape(_display(value)).replace("\n", "<br/>"), style)


def _as_rows(value: Any, max_rows: int = 40) -> tuple[list[str], list[list[Any]]]:
    """
    Normalize DataFrame/list/dict/scalar values to table headers and rows.
    Does not use truthiness checks on pandas objects.
    """
    if value is None:
        return [], []

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return [], []
        frame = value.copy()
        headers = [str(column) for column in frame.columns]
        rows = frame.head(max_rows).values.tolist()
        return headers, rows

    if isinstance(value, pd.Series):
        if value.empty:
            return [], []
        return ["Field", "Value"], [[index, item] for index, item in value.head(max_rows).items()]

    if isinstance(value, dict):
        if not value:
            return [], []
        # Nested tabular data gets rendered as key/value pairs to avoid losing information.
        return ["Field", "Value"], [[key, item] for key, item in list(value.items())[:max_rows]]

    if isinstance(value, (list, tuple)):
        if not value:
            return [], []
        if all(isinstance(item, dict) for item in value):
            keys: list[str] = []
            for item in value:
                for key in item:
                    if key not in keys:
                        keys.append(str(key))
            return keys, [[item.get(key, "") for key in keys] for item in value[:max_rows]]
        return ["Item"], [[item] for item in value[:max_rows]]

    return ["Value"], [[value]]


def _make_table(
    headers: list[str],
    rows: list[list[Any]],
    body_style: ParagraphStyle,
    header_style: ParagraphStyle,
    available_width: float,
) -> Table | None:
    if not headers or not rows:
        return None

    data = [[_paragraph(header, header_style) for header in headers]]
    for row in rows:
        normalized = list(row)
        if len(normalized) < len(headers):
            normalized.extend([""] * (len(headers) - len(normalized)))
        data.append([_paragraph(item, body_style) for item in normalized[: len(headers)]])

    column_count = len(headers)
    col_width = available_width / max(column_count, 1)
    table = Table(
        data,
        colWidths=[col_width] * column_count,
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#233044")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4D9E0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F8")]),
            ]
        )
    )
    return table


def _append_section(
    story: list[Any],
    title: str,
    value: Any,
    styles: Any,
    available_width: float,
    empty_message: str = "No results were available for this section in the current session.",
) -> None:
    story.append(Paragraph(title, styles["Heading2"]))
    headers, rows = _as_rows(value)
    if not headers or not rows:
        story.append(Paragraph(empty_message, styles["BodyText"]))
        story.append(Spacer(1, 5 * mm))
        return

    body_style = styles["TableBody"]
    header_style = styles["TableHeader"]
    table = _make_table(headers, rows, body_style, header_style, available_width)
    if table is not None:
        story.append(table)
        if len(rows) >= 40:
            story.append(Paragraph("Showing the first 40 rows.", styles["SmallNote"]))
    story.append(Spacer(1, 6 * mm))


def _risk_summary(risk_result: Any) -> dict[str, Any]:
    if not isinstance(risk_result, dict):
        return {}
    summary_keys = (
        "rows",
        "columns",
        "missing_cells",
        "missing_percentage",
        "duplicate_rows",
        "risk_score",
        "risk_level",
    )
    return {key: risk_result[key] for key in summary_keys if key in risk_result}


def generate_audit_pdf(
    risk_result: Any = None,
    leakage_report: Any = None,
    drift_result: Any = None,
    dataset_name: str = "Uploaded dataset",
) -> bytes:
    """
    Build a complete PDF audit report and return it as bytes.

    Parameters may be None when that analysis has not been run. DataFrame values
    are checked using .empty and are never used directly in if/and/or expressions.
    """
    buffer = BytesIO()
    page_size = A4
    doc = SimpleDocTemplate(
        buffer,
        pagesize=page_size,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="DataSentinel AI Audit Report",
        author="DataSentinel AI",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#233044"),
            fontSize=21,
            leading=26,
            spaceAfter=5 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableBody",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9,
            textColor=colors.white,
            fontName="Helvetica-Bold",
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallNote",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#555555"),
            spaceBefore=2 * mm,
        )
    )

    available_width = page_size[0] - doc.leftMargin - doc.rightMargin
    story: list[Any] = [
        Paragraph("DataSentinel AI", styles["ReportTitle"]),
        Paragraph("Dataset Quality & Risk Audit Report", styles["Heading1"]),
        Spacer(1, 2 * mm),
        Paragraph(f"<b>Dataset:</b> {_paragraph(dataset_name, styles['BodyText']).getPlainText()}", styles["BodyText"]),
        Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]),
        Spacer(1, 5 * mm),
    ]

    summary = _risk_summary(risk_result)
    if summary:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        summary_rows = [[key.replace("_", " ").title(), value] for key, value in summary.items()]
        summary_table = _make_table(
            ["Metric", "Value"],
            summary_rows,
            styles["TableBody"],
            styles["TableHeader"],
            available_width,
        )
        if summary_table is not None:
            story.append(summary_table)
        story.append(Spacer(1, 6 * mm))
    else:
        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(Paragraph("Risk summary was not available in the current session.", styles["BodyText"]))
        story.append(Spacer(1, 5 * mm))

    if isinstance(risk_result, dict):
        issues = risk_result.get("issues")
    else:
        issues = None
    _append_section(
        story,
        "Dataset Risk Findings",
        issues,
        styles,
        available_width,
        "No risk findings were supplied, or the issue list is empty.",
    )

    _append_section(
        story,
        "Target Leakage Screening",
        leakage_report,
        styles,
        available_width,
        "Leakage screening was not run, or no results were returned.",
    )

    if isinstance(drift_result, dict):
        drift_report = drift_result.get("report")
        reference_only = drift_result.get("reference_only_columns")
        current_only = drift_result.get("current_only_columns")
        drift_thresholds = {
            key: drift_result[key]
            for key in ("numeric_threshold", "categorical_threshold")
            if key in drift_result
        }
        if drift_thresholds:
            story.append(Paragraph("Drift Detection Settings", styles["Heading2"]))
            settings_rows = [[key.replace("_", " ").title(), value] for key, value in drift_thresholds.items()]
            settings_table = _make_table(
                ["Setting", "Value"],
                settings_rows,
                styles["TableBody"],
                styles["TableHeader"],
                available_width,
            )
            if settings_table is not None:
                story.append(settings_table)
            story.append(Spacer(1, 4 * mm))
        _append_section(
            story,
            "Dataset Drift Results",
            drift_report,
            styles,
            available_width,
            "Drift detection was not run, or its report is empty.",
        )
        schema_rows = []
        if reference_only is not None:
            if isinstance(reference_only, (list, tuple, set)):
                schema_rows.append(["Columns only in reference", ", ".join(map(str, reference_only)) or "None"])
            else:
                schema_rows.append(["Columns only in reference", reference_only])
        if current_only is not None:
            if isinstance(current_only, (list, tuple, set)):
                schema_rows.append(["Columns only in current", ", ".join(map(str, current_only)) or "None"])
            else:
                schema_rows.append(["Columns only in current", current_only])
        if schema_rows:
            story.append(Paragraph("Schema Differences", styles["Heading2"]))
            schema_table = _make_table(
                ["Difference", "Columns"],
                schema_rows,
                styles["TableBody"],
                styles["TableHeader"],
                available_width,
            )
            if schema_table is not None:
                story.append(schema_table)
            story.append(Spacer(1, 5 * mm))
    else:
        _append_section(
            story,
            "Dataset Drift Results",
            None,
            styles,
            available_width,
            "Drift detection was not run in the current session.",
        )

    story.append(Paragraph("Interpretation Notes", styles["Heading2"]))
    story.append(
        Paragraph(
            "Automated checks identify potential data-quality and modeling risks; they do not, by themselves, "
            "prove that a dataset is defective or that a model is invalid. Review findings in the context of "
            "the data collection process, target definition, sample size, and intended use.",
            styles["BodyText"],
        )
    )

    def add_page_number(canvas: Any, document: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(page_size[0] - doc.rightMargin, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError("PDF generation did not produce a valid PDF file.")
    return pdf_bytes
