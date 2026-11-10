'use strict';
/* Content for the Architecture view.
 *
 * Kept in its own file rather than inlined so index.html stays about the
 * chat, and so this page can be edited without touching the streaming code.
 * Rendered once, on first open.
 */
(function () {
  const arch = document.getElementById('arch');
  if (!arch) return;

  const box = (title, sub) =>
    `<div class="fbox"><b>${title}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
  const key = (title, sub) =>
    `<div class="fbox key"><b>${title}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
  const off = (title, sub) =>
    `<div class="fbox off"><b>${title}</b>${sub ? `<small>${sub}</small>` : ''}</div>`;
  const arrow = '<div class="farrow"></div>';

  const INGEST = [
    box('12 source documents', '11 Markdown + 1 PDF · read-only, never edited'),
    arrow,
    box('Parse', 'Header-aware Markdown · pdfplumber recovers the PDF table from 48 filled rectangles'),
    arrow,
    box('Sanitise', 'Strip the planted prompt-injection payload, keep the legitimate content, log the strip'),
    arrow,
    box('Chunk', 'Tables never split · every chunk carries a provenance line and its document status'),
    arrow,
    `<div class="frow">
       ${box('BM25 index', 'Custom tokeniser keeps ND-1042 whole and folds Bengali digits')}
       ${box('Qdrant vectors', 'bge-m3, 1024 dimensions, embedded local mode — no server')}
     </div>`,
    '<div class="fnote">One-off, ~45 seconds. Asserts 12 files, 5 PDF limit rows, exactly one superseded document and exactly one injection before writing anything.</div>',
  ].join('');

  const ANSWER = [
    box('Your question'),
    arrow,
    off('Fast paths', 'Small talk and prompt-extraction attempts answered here — no retrieval, no model, ~0.2 s'),
    arrow,
    box('Rewrite the follow-up', 'qwen3:1.7b resolves "it" and "some" against the last turns'),
    arrow,
    `<div class="frow">
       ${box('BM25 top-20', 'Exact words and identifiers')}
       ${box('Vector top-20', 'Meaning, across languages')}
     </div>`,
    arrow,
    box('Fuse the two rankings', 'Reciprocal rank fusion, k = 60'),
    arrow,
    key('Exact-identifier boost', 'Before reranking — a cross-encoder has no special respect for ND-1042'),
    arrow,
    box('Candidate window', '12 candidates, with seats reserved for each retriever’s own best'),
    arrow,
    box('Cross-encoder rerank', 'jina multilingual, ONNX on CPU → top 5'),
    arrow,
    box('Down-rank superseded', 'Halved after a sigmoid — demoted, never deleted'),
    arrow,
    key('Relevance floor 0.088', 'After reranking — below it, the assistant declines instead of guessing'),
    arrow,
    box('Generate', 'qwen3:4b-instruct, native /api/chat, streaming'),
    arrow,
    box('Server-sent events → browser', 'Tokens, stage updates, sources, timings'),
  ].join('');

  const rows = (list) => list.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('');
  const table = (heads, list) =>
    `<table><thead><tr>${heads.map(h => `<th>${h}</th>`).join('')}</tr></thead>
     <tbody>${rows(list)}</tbody></table>`;

  arch.innerHTML = `
    <h2>How this assistant works</h2>
    <p class="lead">A question-answering system over twelve internal Nodi Tech documents.
    It runs entirely on this machine — no cloud model, no API keys, no data leaving the
    computer. Every answer is drawn from the documents and lists the ones it used.</p>

    <div class="stats">
      <div class="stat"><b>39/40</b><span>answers correct</span></div>
      <div class="stat"><b>36/36</b><span>retrieval hit rate</span></div>
      <div class="stat"><b>220</b><span>tests passing</span></div>
      <div class="stat"><b>0</b><span>cloud calls</span></div>
    </div>
    <p class="fnote">Measured on the full 40-question evaluation set with zero technical
    errors, so the run counts. Full report in <code>eval/results/REPORT.md</code>.</p>

    <div class="flane">Building the index · once</div>
    <div class="flow">${INGEST}</div>

    <div class="flane">Answering a message · every time</div>
    <div class="flow">${ANSWER}</div>

    <h3>Why hybrid retrieval</h3>
    <p>Neither search method alone is enough for this corpus, and the two fail in
    opposite directions.</p>
    ${table(['Method', 'Strong at', 'Blind to'], [
      ['<b>BM25</b> keyword search',
       'Exact identifiers — <code>ND-1042</code>, <code>SEC-EXC</code>, <code>X-Nodi-Key</code>',
       'Anything phrased differently, and every Bangla document'],
      ['<b>bge-m3</b> vector search',
       'Meaning, including an English question against a Bangla-only document',
       'Treats one error code as much like another'],
      ['<b>Cross-encoder rerank</b>',
       'Judging a passage against the actual question',
       'Costs ~5 s on CPU, so it only sees 12 candidates'],
    ])}

    <h3>The six traps in the dataset</h3>
    <p>The documents are deliberately booby-trapped. Each trap has a specific
    mechanism and a test that keeps it fixed.</p>
    ${table(['Trap', 'How it is handled'], [
      ['<b>A superseded policy</b><br><small>Old leave policy says 15 days, the handbook says 18</small>',
       'Status read from the filename and the document header, the warning embedded into the text the search engine sees, and the score halved. Demoted, never deleted — the correct answer cites <em>both</em> documents and calls the old figure outdated.'],
      ['<b>A prompt injection</b><br><small>A vendor notice instructs the assistant to lie about pricing</small>',
       'The payload is stripped when the document is indexed and the strip is logged. The rest of the notice stays, because a real question about printer maintenance must still be answerable from it. Retrieved text is also fenced as untrusted in the prompt.'],
      ['<b>Exact identifiers</b><br><small>Standard tokenisers shred <code>ND-1042</code> into "nd" and "1042"</small>',
       'A custom tokeniser keeps the whole code <em>and</em> its parts, a literal match is boosted before reranking, and a matching passage is guaranteed a place in the window.'],
      ['<b>Tables, including one inside a PDF</b>',
       'A chunk holds a whole table or none of it, always with its heading. "BDT 2,500" is meaningless without the "Growth" column. The PDF’s borders are filled rectangles rather than lines, which pdfplumber resolves.'],
      ['<b>Bangla, and a cross-language question</b>',
       'Multilingual embeddings carry an English question to a Bangla-only document. Bengali digits are folded to ASCII for keyword search only, so displayed answers keep their original script.'],
      ['<b>Questions the documents cannot answer</b>',
       'Four different refusals. The hardest: when the documents list some options but not the one asked about, the assistant says it is not mentioned and stops — it never turns silence into "no".'],
    ])}

    <h3>What runs where</h3>
    ${table(['Component', 'Choice', 'Reason'], [
      ['Language model', '<code>qwen3:4b-instruct</code> via Ollama',
       'A dedicated non-thinking checkpoint. Thinking models spend their budget reasoning and return nothing on a CPU this size.'],
      ['Embeddings', '<code>bge-m3</code>, 1024-d',
       'Multilingual. The English-only alternative fails exactly one question — the English-to-Bangla one.'],
      ['Vector store', 'Qdrant, embedded mode',
       'Same API as the server, persisted to disk, no Docker and no extra process.'],
      ['Keyword search', '<code>rank_bm25</code>',
       'No server for 50 passages, and full control of the tokeniser, which is what actually matters here.'],
      ['Reranker', 'jina multilingual cross-encoder (ONNX)',
       'No PyTorch install. Non-commercial licence — must be swapped before commercial use.'],
      ['Serving', 'FastAPI + server-sent events',
       'Streams tokens as they are produced, with no build step and one HTML page.'],
    ])}

    <h3>Speed, and why</h3>
    <p>An answer takes roughly 30–80 seconds on this hardware. The cause is not the
    model being small — it is that there is no usable GPU, and that grounding an answer
    means reading documents first.</p>
    ${table(['Stage', 'Measured', 'Nature'], [
      ['Search and rerank', '4–8 s', 'Cross-encoder on CPU'],
      ['<b>Reading the prompt</b>', '<b>15–25 s</b>', 'The bottleneck. ~24 tokens/second on 4 CPU cores'],
      ['Writing the answer', '3–15 s', '5.8 tokens/second, limited by memory bandwidth'],
      ['The same question again', '~9 s', 'The prompt prefix is cached'],
      ['A greeting', '~0.2 s', 'Answered before any of the above runs'],
    ])}
    <p class="fnote">A bare question with no documents attached is read in 0.8 seconds.
    Attaching five passages so the answer can be grounded and cited is what costs the
    time. A model with less than half the parameters is only about twice as fast — a
    GPU with 4 GB or more is the only change that would alter the picture.</p>

    <h3>What was built</h3>
    ${table(['Area', 'Covers'], [
      ['<b>Ingestion</b>', 'Markdown and PDF parsing, Unicode normalisation, injection sanitising, table-safe chunking, and assertions that refuse to write a partial index.'],
      ['<b>Retrieval</b>', 'Keyword and vector search, rank fusion, identifier boosting, candidate selection with reserved seats, cross-encoder reranking, superseded down-ranking.'],
      ['<b>Conversation</b>', 'Follow-up rewriting, a bounded history window, streaming generation, persisted conversations, and fast paths for small talk and prompt-extraction attempts.'],
      ['<b>Resilience</b>', 'Startup checks that refuse to run misconfigured, one defined fallback per stage, and a strict separation between "I don’t know" and "something broke" — the two must never look alike.'],
      ['<b>Evaluation</b>', 'A fast retrieval harness, a full answer harness with an audited judge, a multi-turn conversation harness, and a threshold calibration sweep. Every verdict is written out for inspection.'],
    ])}
  `;
})();
