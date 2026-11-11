import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.cwd();
const skillDir = process.env.CODEX_PRESENTATION_SKILL_DIR ?? path.join(
  process.env.HOME,
  ".codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations",
);
const runtimePython = process.env.RUNTIME_PYTHON;
if (!runtimePython) throw new Error("Set RUNTIME_PYTHON to the bundled workspace Python executable");
const buildDir = path.join(workspaceDir, ".build", "yingqi-presentation");
const outputDir = path.join(workspaceDir, "deliverables");
const figureDir = path.join(workspaceDir, "experiments", "reports", "same_window", "figures");
const finalPath = path.join(outputDir, "yingqi_same_window_ablation.pptx");
const fontFamily = "Arial";

const { finalizePresentation, applyPresentationChartFont } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href
);

const C = {
  ink: "#172026",
  muted: "#5E6A71",
  paper: "#F7F7F4",
  white: "#FFFFFF",
  line: "#D9DEDF",
  red: "#C5413B",
  teal: "#177A76",
  blue: "#3267A8",
  gold: "#B17A21",
  paleRed: "#F6E5E2",
  paleTeal: "#DDEDEA",
  paleBlue: "#E4EBF4",
  paleGold: "#F3EBD9",
};

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontFamily,
    fontSize: 18,
    color: C.ink,
    ...style,
  };
  return shape;
}

function addRect(slide, position, fill, line = "none", radius = false) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position,
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
  });
}

function addTitle(slide, title, number) {
  addText(slide, title, { left: 72, top: 42, width: 1050, height: 56 }, {
    fontSize: 32,
    bold: true,
  });
  addText(slide, String(number).padStart(2, "0"), { left: 1140, top: 50, width: 64, height: 32 }, {
    fontSize: 14,
    bold: true,
    color: C.red,
  });
  addRect(slide, { left: 72, top: 108, width: 1136, height: 2 }, C.line);
}

function addFooter(slide, text = "EE6405 · Yingqi · Fair-window ablation") {
  addText(slide, text, { left: 72, top: 680, width: 920, height: 20 }, {
    fontSize: 10,
    color: C.muted,
  });
}

function addChartFont(chart, size = 13) {
  applyPresentationChartFont(chart, { fontFamily, fontSize: size });
}

function setNotes(slide, lines) {
  slide.speakerNotes.text = lines.join("\n");
}

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(figureDir, { recursive: true });

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });

// Slide 1
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addRect(slide, { left: 0, top: 0, width: 22, height: 720 }, C.red);
  addText(slide, "FAIR-WINDOW MECHANISM ABLATION", { left: 92, top: 92, width: 600, height: 30 }, {
    fontSize: 14,
    bold: true,
    color: C.red,
  });
  addText(slide, "Separating discourse relations\nfrom text coverage and pooling", {
    left: 92, top: 158, width: 820, height: 190,
  }, { fontSize: 46, bold: true, color: C.ink });
  addText(slide, "A controlled extension of the RACE reproduction", {
    left: 94, top: 378, width: 640, height: 38,
  }, { fontSize: 22, color: C.muted });
  addRect(slide, { left: 936, top: 112, width: 210, height: 420 }, C.ink);
  addText(slide, "512", { left: 960, top: 178, width: 170, height: 110 }, {
    fontSize: 72, bold: true, color: C.white,
  });
  addText(slide, "shared tokens", { left: 960, top: 292, width: 170, height: 34 }, {
    fontSize: 18, color: "#C9D0D2",
  });
  addRect(slide, { left: 960, top: 370, width: 160, height: 4 }, C.red);
  addText(slide, "4 conditions\n3 seeds\n12 GPU runs pending", {
    left: 960, top: 398, width: 170, height: 106,
  }, { fontSize: 20, bold: true, color: C.white });
  addText(slide, "Yingqi · EE6405 NLP Project · 11 November 2026", {
    left: 92, top: 630, width: 700, height: 24,
  }, { fontSize: 14, color: C.muted });
  setNotes(slide, [
    "Scope: Yingqi contribution package for the EE6405 group project.",
    "Source: docs/SAME_WINDOW_ABLATION.md and experiments/reports/same_window/REPORT.md.",
  ]);
}

// Slide 2
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "Why the original comparison is inconclusive", 2);
  const chart = slide.charts.add("bar", {
    position: { left: 72, top: 145, width: 720, height: 450 },
    categories: ["TF-IDF", "RoBERTa CE", "RoBERTa + SupCon", "RACE CE", "RACE + SupCon"],
    series: [{
      name: "HART Macro-F1",
      values: [79.40, 90.72, 91.15, 89.92, 90.37],
      points: [
        { idx: 0, fill: C.gold },
        { idx: 1, fill: C.blue },
        { idx: 2, fill: C.teal },
        { idx: 3, fill: C.red },
        { idx: 4, fill: "#93403D" },
      ],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 45, varyColors: true },
    hasLegend: false,
    xAxis: { minimumScale: 70, maximumScale: 95, majorGridlines: { style: "solid", fill: C.line, width: 1 }, numberFormatCode: "0.0" },
    yAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
    plotAreaFill: C.paper,
  });
  addChartFont(chart, 13);
  addText(slide, "RACE trails RoBERTa by about 0.8 points under both objectives.", {
    left: 820, top: 154, width: 360, height: 80,
  }, { fontSize: 23, bold: true, color: C.red });
  const confounds = [
    ["Visible text", "RACE uses full-document structure; RoBERTa stops at 512 tokens."],
    ["Readout", "Root graph pooling and CLS pooling answer different questions."],
    ["Capacity", "The graph model adds about 3.38M trainable parameters."],
    ["Mechanism", "EDU segmentation and graph relations change together."],
  ];
  confounds.forEach(([label, body], index) => {
    const top = 270 + index * 82;
    addText(slide, label, { left: 820, top, width: 150, height: 28 }, {
      fontSize: 16, bold: true, color: C.ink,
    });
    addText(slide, body, { left: 950, top, width: 240, height: 58 }, {
      fontSize: 14, color: C.muted,
    });
  });
  addFooter(slide);
  setNotes(slide, [
    "Source: README.md, experiments/reports/comparison/summary.json.",
    "Values are three-seed means except TF-IDF, which is deterministic.",
    "The comparison is descriptive because architecture, input scope, pooling and parameter count differ.",
  ]);
}

// Slide 3
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "Four conditions with one visible text budget", 3);
  addText(slide, "Every condition starts from the same base-pretrained model and sees the exact same 512-token prefix.", {
    left: 72, top: 128, width: 1080, height: 34,
  }, { fontSize: 19, color: C.muted });
  const table = slide.tables.add({
    rows: 5,
    columns: 4,
    left: 72,
    top: 190,
    width: 1136,
    height: 390,
    columnWidths: [140, 390, 380, 226],
    values: [
      ["Condition", "Mechanism", "Question", "Total / trainable"],
      ["text", "RoBERTa CLS + MLP", "Same-window lexical baseline", "125.24M / 7.68M"],
      ["graph", "RACE on the visible-prefix RST graph", "Total graph-pipeline effect", "128.62M / 11.06M"],
      ["shuffled", "Relation assignments permuted per graph", "Does relation type carry signal?", "128.62M / 11.06M"],
      ["edu", "Equal-weight EDU pooling, no GNN", "Does graph processing add value?", "125.31M / 7.75M"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let column = 0; column < 4; column += 1) {
    table.getCell(0, column).fill = C.ink;
    table.getCell(0, column).text.style = { fontFamily, fontSize: 15, bold: true, color: C.white };
  }
  for (let row = 1; row < 5; row += 1) {
    table.getCell(row, 0).text.style = { fontFamily, fontSize: 16, bold: true, color: [C.blue, C.red, C.gold, C.teal][row - 1] };
    for (let column = 1; column < 4; column += 1) {
      table.getCell(row, column).text.style = { fontFamily, fontSize: 14, color: C.ink };
    }
  }
  addText(slide, "Primary matched-seed contrasts: graph − edu, graph − shuffled, graph − text", {
    left: 72, top: 612, width: 1080, height: 34,
  }, { fontSize: 18, bold: true, color: C.red });
  addFooter(slide);
  setNotes(slide, [
    "Source: docs/SAME_WINDOW_ABLATION.md and experiments/reports/same_window/smoke_summary.json.",
    "All conditions use cross-entropy for the primary mechanism comparison.",
  ]);
}

// Slide 4
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "The full same-window cache is ready", 4);
  const chart = slide.charts.add("bar", {
    position: { left: 72, top: 160, width: 660, height: 420 },
    categories: ["Train", "Validation", "Test"],
    series: [{ name: "Records", values: [11190, 1600, 3194], fill: C.blue }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
    hasLegend: false,
    xAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { majorGridlines: { style: "solid", fill: C.line, width: 1 }, numberFormatCode: "#,##0" },
    dataLabels: { showValue: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
    plotAreaFill: C.paper,
  });
  addChartFont(chart, 13);
  addText(slide, "15,984", { left: 810, top: 168, width: 310, height: 76 }, {
    fontSize: 52, bold: true, color: C.ink,
  });
  addText(slide, "verified records", { left: 814, top: 242, width: 260, height: 32 }, {
    fontSize: 19, color: C.muted,
  });
  const checks = [
    ["0", "failed or pending"],
    ["0", "cross-split exact visible duplicates"],
    ["0", "EDU nodes outside the visible prefix"],
    ["100%", "source-block separation retained"],
  ];
  checks.forEach(([value, label], index) => {
    const top = 314 + index * 66;
    addText(slide, value, { left: 814, top, width: 90, height: 38 }, {
      fontSize: 25, bold: true, color: index === 3 ? C.teal : C.red,
    });
    addText(slide, label, { left: 912, top: top + 3, width: 270, height: 34 }, {
      fontSize: 15, color: C.ink,
    });
  });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/preparation.json.",
    "Protocol SHA-256: 82112974df569083226aad59b31d25bd89c5e0afa1328c3f81c80431916705a9.",
    "The cache is external because it contains source text and large graph records.",
  ]);
}

// Slide 5
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "The relation control changes most edge types", 5);
  const chart = slide.charts.add("doughnut", {
    position: { left: 92, top: 150, width: 560, height: 470 },
    categories: ["Changed", "Unchanged"],
    series: [{
      name: "Relation nodes",
      values: [467369, 170669],
      points: [{ idx: 0, fill: C.red }, { idx: 1, fill: C.line }],
    }],
    doughnutOptions: { holeSize: 64, firstSliceAngle: 270 },
    hasLegend: false,
    dataLabels: { showPercent: true, showCategoryName: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
  });
  addChartFont(chart, 14);
  addText(slide, "73.25%", { left: 724, top: 168, width: 350, height: 82 }, {
    fontSize: 54, bold: true, color: C.red,
  });
  addText(slide, "of relation nodes and directed edge types changed", {
    left: 728, top: 250, width: 410, height: 68,
  }, { fontSize: 22, bold: true, color: C.ink });
  const facts = [
    ["15,984", "graphs audited"],
    ["36", "graphs unchanged after permutation"],
    ["17", "graphs with at most one relation type"],
    ["2,552,152", "directed edges checked"],
  ];
  facts.forEach(([value, label], index) => {
    const top = 348 + index * 62;
    addText(slide, value, { left: 728, top, width: 150, height: 32 }, {
      fontSize: 20, bold: true, color: C.teal,
    });
    addText(slide, label, { left: 892, top: top + 2, width: 275, height: 30 }, {
      fontSize: 15, color: C.muted,
    });
  });
  addText(slide, "Topology and each graph's relation histogram remain fixed.", {
    left: 728, top: 610, width: 440, height: 34,
  }, { fontSize: 17, bold: true, color: C.ink });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/relation_control_audit.json.",
    "The control is deterministic from item_id and control seed 20261109 and does not read labels.",
  ]);
}

// Slide 6
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "CPU smoke validates the training path", 6);
  const chart = slide.charts.add("bar", {
    position: { left: 72, top: 160, width: 760, height: 430 },
    categories: ["text", "graph", "shuffled", "edu"],
    series: [
      { name: "Total parameters", values: [125.2393, 128.617148, 128.617148, 125.305092], fill: C.blue },
      { name: "Trainable parameters", values: [7.68154, 11.059388, 11.059388, 7.747332], fill: C.red },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 45 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false },
    xAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { majorGridlines: { style: "solid", fill: C.line, width: 1 }, numberFormatCode: "0.0\"M\"" },
    dataLabels: { showValue: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
    plotAreaFill: C.paper,
  });
  addChartFont(chart, 12);
  addText(slide, "4 / 4 passed", { left: 884, top: 172, width: 290, height: 60 }, {
    fontSize: 38, bold: true, color: C.teal,
  });
  const items = [
    "Finite forward and backward passes",
    "Classifier weights updated",
    "Last encoder layer updated",
    "Frozen embeddings unchanged",
    "Identical visible-token fingerprint",
    "Test partition never evaluated",
  ];
  items.forEach((item, index) => {
    const top = 270 + index * 52;
    addText(slide, "✓", { left: 886, top, width: 28, height: 28 }, {
      fontSize: 18, bold: true, color: C.teal,
    });
    addText(slide, item, { left: 924, top, width: 275, height: 32 }, {
      fontSize: 15, color: C.ink,
    });
  });
  addText(slide, "Engineering evidence only. Predictive quality still requires GPU training.", {
    left: 884, top: 604, width: 320, height: 48,
  }, { fontSize: 15, bold: true, color: C.red });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/smoke_summary.json.",
    "The integrated repository entry point was also run against the complete v3 cache for all four conditions.",
  ]);
}

// Slide 7
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "High HART scores do not transfer to MixSet", 7);
  const chart = slide.charts.add("bar", {
    position: { left: 72, top: 150, width: 880, height: 470 },
    categories: ["RoBERTa CE", "RACE SupCon", "CE/RACE blend", "3-model stack", "Length + structure"],
    series: [
      { name: "HART", values: [90.75, 90.84, 92.10, 93.61, 52.67], fill: C.blue },
      { name: "MixSet subset", values: [39.01, 33.38, 36.86, 36.22, 29.16], fill: C.red },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 38 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false },
    xAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 0, maximumScale: 100, majorGridlines: { style: "solid", fill: C.line, width: 1 }, numberFormatCode: "0\"%\"" },
    dataLabels: { showValue: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
    plotAreaFill: C.paper,
  });
  addChartFont(chart, 11);
  addText(slide, "359 texts", { left: 1000, top: 178, width: 170, height: 46 }, {
    fontSize: 29, bold: true, color: C.ink,
  });
  addText(slide, "91 source groups", { left: 1000, top: 226, width: 190, height: 32 }, {
    fontSize: 17, color: C.muted,
  });
  addText(slide, "The external subset does not preserve the HART ranking or the stacking gain.", {
    left: 1000, top: 316, width: 200, height: 108,
  }, { fontSize: 20, bold: true, color: C.red });
  addText(slide, "Diagnostic subset\nOne neural seed\nNo SOTA claim", {
    left: 1000, top: 500, width: 190, height: 90,
  }, { fontSize: 16, color: C.muted });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/exploratory_results.json.",
    "MixSet values refer to a conservative 359-text, 91-source-group subset, not the full official benchmark.",
    "Stacking uses one underlying neural seed and the HART test set had already been inspected.",
  ]);
}

// Slide 8
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "AI polishing sharply increases false positives", 8);
  const chart = slide.charts.add("bar", {
    position: { left: 110, top: 168, width: 620, height: 420 },
    categories: ["Before polishing", "After AI polishing"],
    series: [{
      name: "Creator false-positive rate",
      values: [4, 40],
      points: [{ idx: 0, fill: C.teal }, { idx: 1, fill: C.red }],
    }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 70, varyColors: true },
    hasLegend: false,
    xAxis: { line: { style: "solid", fill: C.line, width: 1 } },
    yAxis: { minimumScale: 0, maximumScale: 50, majorGridlines: { style: "solid", fill: C.line, width: 1 }, numberFormatCode: "0\"%\"" },
    dataLabels: { showValue: true, position: "outEnd" },
    chartFill: C.paper,
    chartLine: { style: "solid", fill: C.paper, width: 0 },
    plotAreaFill: C.paper,
  });
  addChartFont(chart, 13);
  addText(slide, "10×", { left: 825, top: 178, width: 260, height: 88 }, {
    fontSize: 58, bold: true, color: C.red,
  });
  addText(slide, "increase in false positives for paired human-origin texts", {
    left: 828, top: 270, width: 350, height: 74,
  }, { fontSize: 21, bold: true, color: C.ink });
  addRect(slide, { left: 828, top: 390, width: 340, height: 2 }, C.line);
  addText(slide, "9.76%", { left: 828, top: 422, width: 200, height: 60 }, {
    fontSize: 40, bold: true, color: C.gold,
  });
  addText(slide, "creator recall on 41 original AI texts", {
    left: 828, top: 486, width: 340, height: 50,
  }, { fontSize: 17, color: C.muted });
  addText(slide, "Observational evidence. Editing may change style cues, but this comparison does not establish causality.", {
    left: 828, top: 574, width: 360, height: 68,
  }, { fontSize: 15, bold: true, color: C.red });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/exploratory_results.json.",
    "The comparison uses paired source families and a fixed 0.5 creator threshold.",
    "Do not describe the result as a causal effect of AI polishing.",
  ]);
}

// Slide 9
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "Formal GPU matrix and required evidence", 9);
  const table = slide.tables.add({
    rows: 5,
    columns: 4,
    left: 72,
    top: 160,
    width: 690,
    height: 350,
    columnWidths: [210, 160, 160, 160],
    values: [
      ["Condition", "3407", "42", "2026"],
      ["text", "Pending", "Pending", "Pending"],
      ["graph", "Pending", "Pending", "Pending"],
      ["shuffled", "Pending", "Pending", "Pending"],
      ["edu", "Pending", "Pending", "Pending"],
    ],
  });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  for (let column = 0; column < 4; column += 1) {
    table.getCell(0, column).fill = C.ink;
    table.getCell(0, column).text.style = { fontFamily, fontSize: 15, bold: true, color: C.white };
  }
  for (let row = 1; row < 5; row += 1) {
    table.getCell(row, 0).text.style = { fontFamily, fontSize: 16, bold: true, color: C.ink };
    for (let column = 1; column < 4; column += 1) {
      table.getCell(row, column).fill = C.paleGold;
      table.getCell(row, column).text.style = { fontFamily, fontSize: 14, bold: true, color: C.gold };
    }
  }
  addText(slide, "Each run must save", { left: 820, top: 164, width: 330, height: 42 }, {
    fontSize: 24, bold: true, color: C.ink,
  });
  const outputs = [
    "Frozen config and code hashes",
    "Development curve and selected epoch",
    "Best checkpoint and SHA-256",
    "Validation and test probabilities",
    "Macro-F1, class F1 and confusion matrix",
    "Runtime and peak GPU memory",
  ];
  outputs.forEach((item, index) => {
    const top = 230 + index * 54;
    addText(slide, String(index + 1), { left: 820, top, width: 26, height: 28 }, {
      fontSize: 15, bold: true, color: C.red,
    });
    addText(slide, item, { left: 858, top, width: 340, height: 34 }, {
      fontSize: 15, color: C.ink,
    });
  });
  addText(slide, "Primary analysis: matched-seed differences with mean and sample SD", {
    left: 72, top: 558, width: 690, height: 38,
  }, { fontSize: 18, bold: true, color: C.red });
  addFooter(slide);
  setNotes(slide, [
    "Source: docs/SAME_WINDOW_ABLATION.md.",
    "The test partition must remain closed until the best checkpoint is selected on validation Macro-F1.",
  ]);
}

// Slide 10
{
  const slide = deck.slides.add();
  slide.background.fill = C.paper;
  addTitle(slide, "Evidence boundary", 10);
  addText(slide, "Supported now", { left: 92, top: 158, width: 380, height: 44 }, {
    fontSize: 28, bold: true, color: C.teal,
  });
  const supported = [
    "The fair-window cache is complete and hash-verified.",
    "The four model conditions have runnable training code.",
    "The relation control changes 73.25% of edge types while preserving topology.",
    "Existing checkpoints show weak external transfer and editing sensitivity.",
  ];
  supported.forEach((item, index) => {
    addText(slide, "✓", { left: 94, top: 224 + index * 76, width: 32, height: 32 }, {
      fontSize: 20, bold: true, color: C.teal,
    });
    addText(slide, item, { left: 138, top: 220 + index * 76, width: 430, height: 58 }, {
      fontSize: 18, color: C.ink,
    });
  });
  addRect(slide, { left: 630, top: 148, width: 2, height: 430 }, C.line);
  addText(slide, "Pending", { left: 690, top: 158, width: 380, height: 44 }, {
    fontSize: 28, bold: true, color: C.red,
  });
  const pending = [
    "Twelve formal GPU runs across four conditions and three seeds.",
    "Matched-seed uncertainty for graph, relation and EDU effects.",
    "Independent confirmation on an unseen evaluation protocol.",
    "Any claim that discourse graphs help or fail to help detection.",
  ];
  pending.forEach((item, index) => {
    addText(slide, "○", { left: 692, top: 224 + index * 76, width: 32, height: 32 }, {
      fontSize: 20, bold: true, color: C.red,
    });
    addText(slide, item, { left: 736, top: 220 + index * 76, width: 430, height: 58 }, {
      fontSize: 18, color: C.ink,
    });
  });
  addText(slide, "Current contribution: a reproducible mechanism test, not a performance claim", {
    left: 170, top: 616, width: 940, height: 42,
  }, { fontSize: 23, bold: true, color: C.ink });
  addFooter(slide);
  setNotes(slide, [
    "Source: experiments/reports/same_window/REPORT.md.",
    "This slide defines what the final paper may and may not claim before GPU completion.",
  ]);
}

const figureMap = new Map([
  [1, "01_original_comparison.png"],
  [3, "02_cache_readiness.png"],
  [4, "03_relation_control.png"],
  [5, "04_parameter_and_smoke.png"],
  [6, "05_external_transfer.png"],
  [7, "06_editing_sensitivity.png"],
]);

for (const [index, slide] of deck.slides.items.entries()) {
  const png = await deck.export({ slide, format: "png", scale: 1 });
  const bytes = new Uint8Array(await png.arrayBuffer());
  await fs.writeFile(path.join(buildDir, `slide-${String(index + 1).padStart(2, "0")}.png`), bytes);
  if (figureMap.has(index)) await fs.writeFile(path.join(figureDir, figureMap.get(index)), bytes);
}

const montage = await deck.export({ format: "webp", montage: true });
await fs.writeFile(path.join(buildDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const candidatePath = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);

await finalizePresentation({
  explicitTotalSlideCount: 10,
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-heading-fit",
    "--require-native-table-slide", "3",
    "--require-native-table-slide", "9",
  ],
  requiredNativeTableOwnerSlides: [3, 9],
  requiredNativeChartOwnerSlides: [2, 4, 5, 6, 7, 8],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
  fontPolicy: { basis: "design", families: [fontFamily] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation.json"),
});

console.log(JSON.stringify({ finalPath, figureDir, slides: deck.slides.items.length }, null, 2));
