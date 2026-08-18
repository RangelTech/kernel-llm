"""Output tools: chart (Plotly JSON), XLSX and PDF — all consuming dataset
artifacts by reference, never re-running queries nor passing raw data
through the model context. Registered on the same MCP catalog."""

import io
import json

from app.tools import _context, catalog

_MAX_CHART_POINTS = 5_000


async def _load_dataset(artifact_id: str) -> tuple[dict | None, str]:
    """(dataset, error). Dataset payloads are {"columns": [...], "rows": [...]}."""
    from app.storage import get_artifact, load_payload

    record = await get_artifact(artifact_id)
    if record is None:
        return None, f"ERRO: artifact {artifact_id} não encontrado"
    if record["kind"] != "dataset":
        return None, f"ERRO: artifact {artifact_id} não é um dataset"
    context = _context()
    if record["tenant_id"] and str(record["tenant_id"]) != str(context.get("tenant_id")):
        return None, "ERRO: artifact pertence a outra empresa"
    try:
        payload = json.loads(load_payload(record["storage_path"]))
    except Exception as exc:  # noqa: BLE001 — reported to the model
        return None, f"ERRO ao carregar dataset: {exc}"
    return payload, ""


def _column_values(dataset: dict, column: str) -> list | None:
    names = [c["name"] for c in dataset["columns"]]
    if column not in names:
        return None
    index = names.index(column)
    return [row[index] for row in dataset["rows"][:_MAX_CHART_POINTS]]


@catalog.tool()
async def generate_chart(
    artifact_id: str,
    chart_type: str,
    x_column: str,
    y_columns: str,
    title: str = "",
) -> str:
    """Gera um gráfico interativo (Plotly) a partir de um dataset artifact.
    chart_type: bar | line | scatter | pie. y_columns: nomes separados por
    vírgula. Retorna um chart artifact que o usuário vê renderizado no chat."""
    dataset, error = await _load_dataset(artifact_id)
    if error:
        return error
    if chart_type not in ("bar", "line", "scatter", "pie"):
        return f"ERRO: chart_type inválido: {chart_type}"

    x_values = _column_values(dataset, x_column)
    if x_values is None:
        available = ", ".join(c["name"] for c in dataset["columns"])
        return f"ERRO: coluna x '{x_column}' não existe. Colunas: {available}"

    traces = []
    for y_name in [c.strip() for c in y_columns.split(",") if c.strip()]:
        y_values = _column_values(dataset, y_name)
        if y_values is None:
            available = ", ".join(c["name"] for c in dataset["columns"])
            return f"ERRO: coluna y '{y_name}' não existe. Colunas: {available}"
        if chart_type == "pie":
            traces.append({"type": "pie", "labels": x_values, "values": y_values, "name": y_name})
        else:
            traces.append(
                {
                    "type": "scatter" if chart_type == "line" else chart_type,
                    "mode": "lines+markers" if chart_type == "line" else None,
                    "x": x_values,
                    "y": y_values,
                    "name": y_name,
                }
            )
    figure = {
        "data": [{k: v for k, v in t.items() if v is not None} for t in traces],
        "layout": {"title": {"text": title}, "template": "plotly_dark", "height": 420},
    }

    from app.storage import register_artifact

    context = _context()
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="chart",
        title=title or f"Gráfico {chart_type}",
        schema_json={"chart_type": chart_type, "x": x_column, "y": y_columns},
        preview_json=None,
        row_count=len(x_values),
        payload=json.dumps(figure, ensure_ascii=False, default=str).encode(),
    )
    return json.dumps(descriptor, ensure_ascii=False)


@catalog.tool()
async def export_xlsx(artifact_id: str, title: str = "") -> str:
    """Exporta um dataset artifact para uma planilha XLSX que o usuário pode
    baixar no chat."""
    dataset, error = await _load_dataset(artifact_id)
    if error:
        return error

    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = (title or "Dados")[:31]
    sheet.append([c["name"] for c in dataset["columns"]])
    for row in dataset["rows"]:
        sheet.append([str(v) if isinstance(v, (dict, list)) else v for v in row])
    buffer = io.BytesIO()
    workbook.save(buffer)

    from app.storage import register_artifact

    context = _context()
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="file",
        title=(title or "Planilha") + ".xlsx",
        schema_json=None,
        preview_json=None,
        row_count=len(dataset["rows"]),
        payload=buffer.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        extension="xlsx",
    )
    return json.dumps(descriptor, ensure_ascii=False)


@catalog.tool()
async def generate_pdf(title: str, content_markdown: str, artifact_id: str = "") -> str:
    """Gera um documento PDF a partir de texto (markdown simples: títulos #,
    listas -, parágrafos). Se artifact_id de um dataset for informado, a
    tabela é anexada ao final. O usuário baixa o PDF no chat."""
    from fpdf import FPDF

    def latin1(text: str) -> str:
        # Core PDF fonts are latin-1 only; degrade unsupported chars visibly
        # instead of crashing the tool.
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(0, 9, latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for raw_line in content_markdown.splitlines():
        line = latin1(raw_line.rstrip())
        if not line:
            pdf.ln(3)
            continue
        if line.startswith("#"):
            pdf.set_font("helvetica", "B", 13)
            pdf.multi_cell(0, 8, line.lstrip("# "), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("helvetica", "", 11)
        elif line.startswith(("- ", "* ")):
            pdf.set_font("helvetica", "", 11)
            # Core PDF fonts are latin-1; stick to ASCII bullets.
            pdf.multi_cell(0, 6, "  - " + line[2:], new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("helvetica", "", 11)
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")

    if artifact_id:
        dataset, error = await _load_dataset(artifact_id)
        if error:
            return error
        pdf.ln(4)
        pdf.set_font("helvetica", "B", 11)
        names = [c["name"] for c in dataset["columns"]]
        pdf.multi_cell(0, 6, latin1(" | ".join(names)), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 9)
        for row in dataset["rows"][:200]:
            line = latin1(" | ".join(str(v) for v in row))
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

    payload = bytes(pdf.output())

    from app.storage import register_artifact

    context = _context()
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="file",
        title=title + ".pdf",
        schema_json=None,
        preview_json=None,
        row_count=None,
        payload=payload,
        content_type="application/pdf",
        extension="pdf",
    )
    return json.dumps(descriptor, ensure_ascii=False)
