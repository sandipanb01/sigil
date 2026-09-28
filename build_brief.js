// build_brief.js 
//
// Regenerate after any change to a quoted number: audit_claims.py checks the
// counts in this file against the code. Run:  node build_brief.js
//
// Licence: Apache-2.0.

const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const NAVY = "0B1E33", GREEN = "1B7F5A", MUTED = "5A6773", RED = "C4342B";
const FONT = "Arial";

function p(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.before ?? 0, after: opts.after ?? 120,
               line: opts.line ?? 264 },
    alignment: opts.align,
    border: opts.border,
    children: (Array.isArray(runs) ? runs : [runs]).map(
      (r) => (typeof r === "string"
        ? new TextRun({ text: r, font: FONT, size: opts.size ?? 20,
                        color: opts.color ?? NAVY })
        : new TextRun({ font: FONT, size: opts.size ?? 20,
                        color: opts.color ?? NAVY, ...r }))),
  });
}

function h(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.before ?? 260, after: opts.after ?? 110 },
    children: [new TextRun({ text, font: FONT, bold: true,
                             size: opts.size ?? 24,
                             color: opts.color ?? NAVY })],
  });
}

const HAIRLINE = {
  bottom: { style: BorderStyle.SINGLE, size: 6, color: "D8DEE5", space: 8 },
};

function cell(text, { bold = false, color = NAVY, width, shade } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text, font: FONT, size: 17, bold, color })],
    })],
  });
}

const COLS = [1560, 2500, 1500, 1500];
const TOTAL = COLS.reduce((a, b) => a + b, 0);

function resultsTable() {
  const head = ["GB/s", "device", "measured", "predicted"];
  const rows = [
    ["300", "L4", "PTQ1_0", "PTQ1_0"],
    ["864", "L40S", "PTQ1_0", "PTQ1_0"],
    ["960", "RTX 6000 Ada", "PTQ1_0", "PTQ1_0"],
    ["1,008", "RTX 4090", "PTQ1_0", "PTQ1_0"],
    ["1,792", "RTX 5090", "PQ2_0", "PQ2_0"],
    ["1,792", "RTX PRO 6000", "PQ2_0", "PQ2_0"],
    ["2,039", "A100 SXM", "PQ2_0", "PQ2_0"],
    ["3,350", "H100 SXM", "PQ2_0", "PQ2_0"],
  ];
  return new Table({
    columnWidths: COLS,
    width: { size: TOTAL, type: WidthType.DXA },
    rows: [
      new TableRow({
        children: head.map((t, i) =>
          cell(t, { bold: true, color: MUTED, width: COLS[i], shade: "F2F4F7" })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((t, i) => cell(t, {
          width: COLS[i],
          bold: i >= 2,
          color: i >= 2 ? (t === "PTQ1_0" ? "1F5FA8" : "C9861A") : NAVY,
        })),
      })),
    ],
  });
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
      },
    },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "SIGIL-Edge", font: FONT, bold: true,
                                 size: 40, color: NAVY })],
      }),
      p("On-device Indic document intelligence for Snapdragon-powered HP PCs",
        { size: 22, color: MUTED, after: 260, border: HAIRLINE }),

      p([{ text: "Snapdragon X2 Plus ships 80 TOPS of INT8 Hexagon NPU against " +
                 "152 GB/s of LPDDR5X. At batch 1 -- the normal on-device case " +
                 "-- decode re-reads the weights and the KV cache once per " +
                 "token, so " },
          { text: "the binding quantity is the stored bit width on the wire, " +
                  "not TOPS", bold: true },
          { text: ". We tested that on Qualcomm's own measurements rather than " +
                  "asserting it: across Qwen3 0.6B to 8B on the X Elite NPU " +
                  "under Qualcomm's QAIRT runtime, decode time is a straight " +
                  "line in bytes moved per token, " +
                  "with R-squared of at least 0.99 under every way of counting " +
                  "the LM head and the KV cache." }]),

      h("Qualcomm's measurements, read through the scaling book"),
      p("Our roofline first reproduces 17 of 17 worked answers from the " +
        "scaling book How to Scale Your Model. Only then is it applied to the " +
        "90 X-series LLM measurements that ship inside Qualcomm's own " +
        "qai_hub_models package -- nothing downloaded."),
      p([{ text: "On X2 Elite the engine barely matters for decode; the " +
                 "software does. ", bold: true },
          { text: "Qwen3-4B decodes at 33.7 / 33.5 / 36.2 tokens/s on CPU / " +
                  "GPU / NPU, each on its best runtime, while its prefill " +
                  "spans 461 to 2307 tokens/s. On X Elite the same model " +
                  "decodes 3.23x faster under QAIRT than under Genie, on the " +
                  "same NPU." }]),
      p([{ text: "Prefill over decode is the critical batch. ", bold: true },
          { text: "If prefill is compute-bound and decode bandwidth-bound, " +
                  "their ratio is the book's B_crit and cannot depend on model " +
                  "size. On the X2 Elite NPU it is 62.8-70.6 across three sizes " +
                  "(CV 0.05); on the CPU of the same chip, 13.7-17.1. That is " +
                  "how many draft tokens a speculative verifier checks for the " +
                  "price of one step -- so verification belongs on the NPU." }]),
      p([{ text: "Past ~10-13K tokens, KV traffic outweighs the weights. ",
           bold: true },
          { text: "Qwen3-4B on X2 Elite crosses over at 9,770 tokens, or about " +
                  "12,700 without the one point leave-one-out flags. For long " +
                  "Indic documents, KV quantisation is the lever. Two of " +
                  "Qualcomm's published pairs are physically impossible; we " +
                  "flag them and leave them out of the cross-model fits " +
                  "rather than explain them away." }]),

      h("The target, stated exactly"),
      p([{ text: "HP ships seven Snapdragon machines. Each is mapped to its " +
                 "silicon, its RAM budget and its AI Hub profiling device, " +
                 "and the gaps are named rather than papered over. The " },
          { text: "HP OmniBook Ultra 14-kg000 carries a Snapdragon X2 Plus, " +
                  "for which AI Hub offers no device at all", bold: true },
          { text: " -- it can only be measured on the metal, and X2 Elite " +
                  "numbers are never presented as X2 Plus numbers. And the " },
          { text: "8 GB OmniBook 3", bold: true },
          { text: " leaves roughly 4-5 GB after Windows, which rules out an " +
                  "8B model at INT4 resident. That machine, not the 64 GB " +
                  "flagship, is what the architecture has to respect." }]),
      p([{ text: "Qualcomm's package carries 491 measured entries for X Elite " +
                 "CRD and 487 for X2 Elite CRD, and " },
          { text: "none for X Plus 8-Core CRD", bold: true },
          { text: " -- the device that stands in for four of the seven HP " +
                  "machines. That gap has now been filled, without a single " +
                  "model weight reaching the laptop. One Qwen3-1.7B decoder " +
                  "layer was built locally at its real dimensions, uploaded " +
                  "once, and put through AI Hub Workbench's compile, quantise " +
                  "and profile jobs at fp16, w8a16 and w4a16. " },
          { text: "At fp16 the layer runs in 6.314 ms, 53 operators, all of " +
                  "them on the NPU, 81.7 MB peak memory", bold: true },
          { text: " (job j5ql4e34p) -- 18.6 GB/s, 13.8% of the 135 GB/s peak, " +
                  "on a layer profiled in isolation. The quantised paths " +
                  "compiled and then failed on the device: w8a16 with " +
                  "QNN_COMMON_ERROR_MEM_ALLOC, w4a16 \"failed after " +
                  "compiling\". " },
          { text: "The two failures are a result, not a gap", bold: true },
          { text: " -- both graphs converted cleanly, so this is a runtime " +
                  "allocation limit on the 8-core part, and it is the first " +
                  "public evidence of where its profiler stops. The ledger " +
                  "for the whole run: 0 bytes of model weights down, 7,275 " +
                  "bytes of profile JSON, against a hard 2 MB ceiling." }]),
      p([{ text: "On the NPU question, the honest answer has two halves. " },
          { text: "Prefill runs on the Hexagon NPU", bold: true },
          { text: " -- INT4 weights, static shapes, via GenieX or ONNX Runtime " +
                  "QNN -- where Qualcomm's X2 Elite data shows it 5x faster " +
                  "than the CPU. " },
          { text: "Decode is bound by the bus", bold: true },
          { text: ", so what decides its engine is the container width, and " +
                  "stock QNN has no ternary matmul: the ternary container runs " +
                  "on the ARM CPU. Claiming the whole pipeline runs on the NPU " +
                  "is the claim that gets taken apart under questioning." }]),

      h("1. The container rule -- up to 35% of decode throughput, for free"),
      p([{ text: "The " },
          { text: "same", italics: true },
          { text: " 1-bit Bonsai-27B weights ship in two containers. The GGUF " +
                  "Q1_0 pack is 3,803,452,480 bytes (1.131 bits/weight); the " +
                  "MLX pack is 5,129,115,752 bytes (1.525 bpw), because MLX " +
                  "stores an affine scale " },
          { text: "and", italics: true },
          { text: " a zero-point per group. Identical weights, identical " +
                  "accuracy, " },
          { text: "35% more bytes across the wire on every token", bold: true },
          { text: " at zero context, and 11% at 128K, because KV traffic is " +
                  "shared." }]),
      p("Throughput is published on ten machines, and for both ternary " +
        "containers (PTQ1_0 at 1.768 bpw, PQ2_0 at 2.143 bpw) on eight of " +
        "them. That is a " +
        "falsification test, and nothing in our engine was fitted to it. Our " +
        "roofline predicts only a sign -- bandwidth-starved parts should " +
        "prefer the narrower stream, compute-starved parts the cheaper " +
        "unpack:"),
      resultsTable(),
      p([{ text: "Eight of eight, separating monotonically, with the " +
                 "crossover bracketed to 1,008-1,792 GB/s. " },
          { text: "Every Snapdragon part sits 4.4-19.7x below that bracket",
            bold: true },
          { text: ", so on Snapdragon the narrower container always wins with " +
                  "no benchmarking required. Nothing is retrained and no " +
                  "accuracy is spent." }],
        { before: 200 }),

      h("2. A measured KV-compression budget"),
      p("We sweep INT4/INT3/INT2 KV quantisation across scale granularities " +
        "(per-token, group-128, group-32), with and without a randomised " +
        "Hadamard rotation, and report WikiText-2 perplexity against an FP16 " +
        "baseline. The finding we lead with is that scale granularity " +
        "dominates rotation choice. Reproducible in under two hours on a free " +
        "Colab T4."),

      h("3. Zero added operators, with shipped precedent"),
      p("Orthogonal head-dimension transforms fold exactly into W_q/W_k and " +
        "W_v/W_o, leaving attention logits and block outputs unchanged -- " +
        "exact to ~1e-15 in float64, and 3.15e-07 end to end in fp32 on a " +
        "random-init test model; the Colab run repeats it on real weights. " +
        "The deployed " +
        "graph gains zero operators, which is critical on Hexagon, where a " +
        "novel per-token operator means no NPU path at all and a silent CPU " +
        "fallback. Bonsai 2 27B ships exactly this: a blockwise Hadamard " +
        "basis folded into the stored weights, with metadata costing 297,903 " +
        "bytes of an 8.6 GB pack -- 0.0035%. Its widths are exactly 5, 6 and " +
        "17 blocks of 1024, a divisibility constraint we now enforce before " +
        "export."),

      h("4. Certificates, because quantised models fail silently"),
      p([{ text: "The model's own runtime note: " },
          { text: '"Ordinary MLX loaders do not apply the required ' +
                  'transforms."', italics: true },
          { text: " A loader that does not know about the rotated basis " },
          { text: "returns wrong output rather than an error", bold: true },
          { text: ". On Snapdragon the exposure is wider, since a model " +
                  "reaching Hexagon has been through QNN conversion, graph " +
                  "optimisation and quantisation. Borrowing the protocol of " +
                  "machine-checked proofs -- an independent checker that " +
                  "trusts nothing the producer built -- we ship a ~3 KB JSON " +
                  "beside the weights declaring the scheme, the transform, " +
                  "seeded probes and a tolerance derived from the declared " +
                  "dtype. It refuses seven of seven injected faults at four " +
                  "problem sizes; a check took about half a millisecond on a " +
                  "laptop CPU, with nothing a device lacks." }]),

      h("5. When another refinement round is real"),
      p([{ text: "Transferring the correction-cycle structure of a " +
                 "Navier-Stokes blowup proof gives a law for post-training " +
                 "quantisation refinement, measured on synthetic matrices: " },
          { text: "kept ~ 1 - 1.75 * d/n", bold: true },
          { text: ", where d is the matrix input dimension and n the " +
                  "calibration token count. At n/d = 0.5 refinement improves " +
                  "calibration error 2.0x while making held-out error " },
          { text: "worse", bold: true },
          { text: " (0.77x). For Bonsai 2 27B, the law predicts that a 128 x " +
                  "512 calibration set keeps only 54% of any gain on " +
                  "down_proj -- the " +
                  "matrix that always binds." }]),

      h("6. The benchmark history is a simulator"),
      p("Every 90-minute sweep writes a complete record over the " +
        "configurations it tried, so a cheaper search policy can be scored " +
        "against it for nothing. Greedy neighbourhood search reaches the same " +
        "answer in four evaluations against the sweep's eighteen -- 4.5x " +
        "fewer calls, no GPU. Replay is valid only inside the realized search " +
        "space, so a policy asking for an unevaluated configuration is marked " +
        "unreliable and never ranked. Both shipped pools are flagged as too " +
        "degenerate to rank policies fairly, and the tool says so."),

      h("Deployment"),
      p("Three real paths, no bespoke runtime: GenieX for NPU/GGUF serving " +
        "with an OpenAI-compatible endpoint, run on the HP laptop itself; the " +
        "ONNXRuntime-QNN path for document parsing and RAG on the Snapdragon " +
        "NPU (Windows ARM64); and AI Hub Workbench for compile-and-profile " +
        "jobs on the X-series devices, with nothing downloaded to the " +
        "developer's machine."),
      p([{ text: "Both findings land on a shipping app. AnythingLLM is an " +
                 "MIT-licensed local-first document-chat desktop app whose " +
                 "Snapdragon NPU port ships Llama-3.2-3B, Llama-3.1-8B and " +
                 "Phi-3.5-mini -- the same class of model, on the same chips. " },
          { text: "The runtime matters more than the model choice", bold: true },
          { text: ": on X Elite at w4a16, Qualcomm's own measurements give " +
                  "Llama-3.2-3B 11.32 tok/s under Genie against 19.82 under " +
                  "QAIRT, and Llama-3.1-8B 5.02 against 10.72 -- 1.75x and " +
                  "2.14x, same weights, same silicon. And that app's NPU " +
                  "engine currently fails on Snapdragon X Plus (X1P42100), " +
                  "because its bundled CPU-detection table does not recognise " +
                  "the part -- an open issue at the time of writing. That is " +
                  "exactly the tier profiled above, and the proxy for four of " +
                  "the seven HP machines." }]),

      h("Use case"),
      p("Indic-language document intelligence: statutory forms, land records, " +
        "exam papers and health documents across Hindi, Marathi, Gujarati, " +
        "Punjabi, Telugu, Kannada, Tamil and Malayalam. These are exactly the " +
        "documents that must not leave the device, in exactly the markets " +
        "where connectivity is unreliable and Snapdragon share is highest. " +
        "Qualcomm already ships an Indic 1.1B model, IndusQ, on AI Hub; this " +
        "extends that direction into a full on-device document pipeline."),

      h("What we tested and rejected", { color: RED }),
      p("Reported rather than buried, because a measured negative result plus " +
        "a working deployment path is defensible under questioning and an " +
        "unverified grand claim is not. Learned rotation on a Gaussianity " +
        "objective lost to a random Hadamard on every integer condition. " +
        "Isotropy-based out-of-distribution routing is mathematically " +
        "impossible. Low-bit models were predicted to resist iterative " +
        "refinement; they do not. Our own catalogue claimed a 27B model at " +
        "1-bit was 3.4 GB -- 12% under the real file. And we once said Qualcomm " +
        "publishes no X-series numbers; its package holds 978 measured X " +
        "Elite and X2 Elite entries, and the real gap is X Plus 8-Core.",
        { color: MUTED }),

      h("Verification"),
      p([{ text: "872 self-tests across 14 modules, a 292-check adversarial " +
                 "stress harness, and a 131-check claim audit that re-derives " +
                 "the quoted numbers from the code. ", bold: true },
          { text: "Nothing in the project downloads a model: the harness reads " +
                  "every file, every command it emits and every document. " +
                  "Every source, status and correction is recorded in " +
                  "PROVENANCE.md, including the corrections to our own " +
                  "published figures." }],
        { color: GREEN }),
    ],
  }],
});

Packer.toBuffer(doc).then((b) => {
  fs.writeFileSync("SIGIL_Edge_Brief_Description.docx", b);
  console.log("  wrote SIGIL_Edge_Brief_Description.docx");
});
