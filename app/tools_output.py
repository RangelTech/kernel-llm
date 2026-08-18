"""Output tools: chart (matplotlib PNG), XLSX and PDF — all consuming dataset
artifacts by reference, never re-running queries nor passing raw data
through the model context. Registered on the same MCP catalog."""

import io
import json

import matplotlib

# Precisa vir antes de "import matplotlib.pyplot": ambiente headless (Cloud
# Run) não tem display, e sem forçar o backend Agg o pyplot tenta abrir um
# backend gráfico e quebra.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 — depende do matplotlib.use acima

from app.tools import _context, catalog  # noqa: E402

_MAX_CHART_POINTS = 5_000

# Paleta consistente com os outros documentos que a Rangel Tech já gera
# (skill `relatorio`, `~/.claude/skills/relatorio/helpers.py`), adaptada para
# o fundo escuro do tema do chat.
_CHART_BG = "#0f172a"
_CHART_FG = "#e2e8f0"
_CHART_GRID = "#334155"
_CHART_PALETTE = ["#4f8ef7", "#f7b84f", "#4fd18a", "#f76f6f", "#b98af7", "#4fd1c5"]
_CHART_DPI = 160


def _new_figure():
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=_CHART_DPI)
    fig.patch.set_facecolor(_CHART_BG)
    ax.set_facecolor(_CHART_BG)
    ax.tick_params(colors=_CHART_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_CHART_GRID)
    ax.grid(True, color=_CHART_GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _style_axes(ax, title: str):
    if title:
        ax.set_title(title, color=_CHART_FG, fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.label.set_color(_CHART_FG)
    ax.yaxis.label.set_color(_CHART_FG)
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor(_CHART_BG)
        legend.get_frame().set_edgecolor(_CHART_GRID)
        for text in legend.get_texts():
            text.set_color(_CHART_FG)


def _draw_bar(ax, x_values, series, horizontal=False):
    n = len(series)
    positions = range(len(x_values))
    width = 0.8 / max(n, 1)
    for i, (name, y_values) in enumerate(series):
        offset = (i - (n - 1) / 2) * width
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        if horizontal:
            ys = [p + offset for p in positions]
            ax.barh(ys, y_values, height=width, label=name, color=color)
        else:
            xs = [p + offset for p in positions]
            ax.bar(xs, y_values, width=width, label=name, color=color)
    if horizontal:
        ax.set_yticks(list(positions))
        ax.set_yticklabels([str(v) for v in x_values])
        ax.invert_yaxis()
    else:
        ax.set_xticks(list(positions))
        ax.set_xticklabels([str(v) for v in x_values], rotation=0 if len(x_values) <= 8 else 45, ha="right")


def _draw_stacked_bar(ax, x_values, series):
    positions = range(len(x_values))
    bottoms = [0.0] * len(x_values)
    for i, (name, y_values) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        ax.bar(list(positions), y_values, bottom=bottoms, label=name, color=color, width=0.6)
        bottoms = [b + v for b, v in zip(bottoms, y_values)]
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(v) for v in x_values], rotation=0 if len(x_values) <= 8 else 45, ha="right")


def _draw_line(ax, x_values, series):
    for i, (name, y_values) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        ax.plot(x_values, y_values, marker="o", label=name, color=color, linewidth=2)


def _draw_area(ax, x_values, series):
    for i, (name, y_values) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        ax.fill_between(range(len(x_values)), y_values, step=None, alpha=0.35, color=color, label=name)
        ax.plot(range(len(x_values)), y_values, color=color, linewidth=1.6)
    ax.set_xticks(list(range(len(x_values))))
    ax.set_xticklabels([str(v) for v in x_values], rotation=0 if len(x_values) <= 8 else 45, ha="right")


def _draw_scatter(ax, x_values, series):
    for i, (name, y_values) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        ax.scatter(x_values, y_values, label=name, color=color)


def _draw_pie(ax, x_values, series):
    # Pizza só faz sentido com uma série; usa a primeira.
    _, y_values = series[0]
    colors = [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(x_values))]
    ax.pie(
        y_values,
        labels=[str(v) for v in x_values],
        colors=colors,
        autopct="%1.0f%%",
        textprops={"color": _CHART_FG, "fontsize": 9},
    )
    ax.set_facecolor(_CHART_BG)


def _draw_hist(ax, x_values, series):
    # Histograma: a "série y" não se aplica — cada coluna de x_values vira uma
    # distribuição própria (a primeira, ou as várias colunas passadas em
    # y_columns quando x_column repete uma coluna numérica).
    for i, (name, y_values) in enumerate(series):
        color = _CHART_PALETTE[i % len(_CHART_PALETTE)]
        ax.hist(y_values, bins=min(20, max(5, len(y_values) // 3 or 5)), alpha=0.75, label=name, color=color)


# Dispatch por tipo de gráfico — aceitar um chart_type novo é só registrar uma
# função aqui, não reescrever a tool inteira.
_CHART_DRAWERS = {
    "bar": lambda ax, x, series: _draw_bar(ax, x, series, horizontal=False),
    "barh": lambda ax, x, series: _draw_bar(ax, x, series, horizontal=True),
    "line": _draw_line,
    "area": _draw_area,
    "scatter": _draw_scatter,
    "pie": _draw_pie,
    "hist": _draw_hist,
    "stacked-bar": _draw_stacked_bar,
}


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
    """Gera um gráfico (PNG via matplotlib) a partir de um dataset artifact.
    chart_type: bar | barh | line | area | scatter | pie | hist | stacked-bar.
    y_columns: nomes separados por vírgula. Retorna um artifact de imagem
    (kind="image") que o usuário vê renderizado no chat — funciona em
    qualquer canal (web, WhatsApp, Instagram), não só no navegador."""
    dataset, error = await _load_dataset(artifact_id)
    if error:
        return error
    drawer = _CHART_DRAWERS.get(chart_type)
    if drawer is None:
        available = ", ".join(sorted(_CHART_DRAWERS))
        return f"ERRO: chart_type inválido: {chart_type}. Tipos suportados: {available}"

    x_values = _column_values(dataset, x_column)
    if x_values is None:
        available = ", ".join(c["name"] for c in dataset["columns"])
        return f"ERRO: coluna x '{x_column}' não existe. Colunas: {available}"

    series = []
    for y_name in [c.strip() for c in y_columns.split(",") if c.strip()]:
        y_values = _column_values(dataset, y_name)
        if y_values is None:
            available = ", ".join(c["name"] for c in dataset["columns"])
            return f"ERRO: coluna y '{y_name}' não existe. Colunas: {available}"
        series.append((y_name, y_values))
    if not series:
        return "ERRO: y_columns vazio — informe ao menos uma coluna"

    fig, ax = _new_figure()
    try:
        drawer(ax, x_values, series)
        _style_axes(ax, title)
        if len(series) > 1 and chart_type != "pie":
            ax.legend(loc="best", fontsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    payload = buffer.getvalue()

    from app.storage import register_artifact

    context = _context()
    descriptor = await register_artifact(
        tenant_id=context.get("tenant_id"),
        chat_id=context.get("chat_id"),
        agent_name=context.get("agent", ""),
        kind="image",
        title=title or f"Gráfico {chart_type}",
        schema_json={"chart_type": chart_type, "x": x_column, "y": y_columns},
        preview_json=None,
        row_count=len(x_values),
        payload=payload,
        content_type="image/png",
        extension="png",
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
