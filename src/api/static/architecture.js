'use strict';
/* Content for the Architecture view.
 *
 * Kept in its own file rather than inlined so index.html stays about the
 * chat, and so this page can be edited without touching the streaming code.
 * Rendered once, on first open.
 *
 * Written for someone who has not seen the code: plain sentences first, the
 * number second, and the reason last. Every figure on this page is measured,
 * not estimated -- `plan.md` 11 and 12 are the logs they come from.
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

  const rows = (list) => list.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('');
  const table = (heads, list) =>
    `<table><thead><tr>${heads.map(h => `<th>${h}</th>`).join('')}</tr></thead>
     <tbody>${rows(list)}</tbody></table>`;

  /* ------------------------------------------------------- diagrams --- */

  const INGEST = [
    box('12 company documents', '11 Markdown files + 1 PDF. Read-only — never edited'),
    arrow,
    box('Read them', 'Headings understood as structure. The PDF table is rebuilt from 48 filled rectangles'),
    arrow,
    box('Clean them', 'Remove the planted instruction attack, keep the real content, write down what was removed'),
    arrow,
    box('Cut into passages', '50 passages. A table is never cut in half, and every passage keeps its heading and its status'),
    arrow,
    `<div class="frow">
       ${box('Word index', 'For exact wording: keeps ND-1042 whole, folds Bengali digits to 0-9')}
       ${box('Meaning index', 'bge-m3 turns each passage into 1024 numbers that capture what it means')}
     </div>`,
    '<div class="fnote">Runs once, in about 11 seconds. It refuses to write a half-built index: it first checks that all 12 files are present, that the PDF table has its 5 rows, and that there is exactly one superseded document and exactly one planted attack.</div>',
  ].join('');

  const ANSWER = [
    box('Your question'),
    arrow,
    off('Shortcuts', 'A greeting, or an attempt to extract the hidden instructions, is answered right here — no search, no model, about 0.2 s'),
    arrow,
    box('Fill in what you left out', '"And for sick leave?" becomes a question that stands on its own, using the last few turns'),
    arrow,
    `<div class="frow">
       ${box('Search by words', 'Top 20 — exact terms and codes')}
       ${box('Search by meaning', 'Top 20 — works across languages')}
     </div>`,
    arrow,
    box('Merge the two lists', 'Reciprocal rank fusion. A passage both searches liked rises'),
    arrow,
    key('Protect exact codes', 'Done BEFORE re-ranking, because the re-ranker has no special respect for ND-1042'),
    arrow,
    box('Shortlist of 12', 'With seats reserved for each search’s own favourite, so neither can be outvoted'),
    arrow,
    box('Re-read all 12 properly', 'A second, slower model reads each passage against your actual question and re-orders them. Best 5 survive'),
    arrow,
    box('Push outdated policies down', 'Score halved — demoted, never deleted'),
    arrow,
    key('Is the best one good enough?', 'Below a measured cut-off of 0.088 the assistant says it does not know, instead of guessing'),
    arrow,
    box('Write the answer', 'qwen3:4b reads the 5 passages and writes, quoting where each fact came from'),
    arrow,
    box('Stream it to this page', 'Word by word, with the step list and the sources'),
  ].join('');

  /* ---------------------------------------------------------- page ---- */

  arch.innerHTML = `
    <h2>How this assistant works</h2>
    <p class="lead">This is a question-answering system for twelve internal Nodi Tech
    documents. You ask in plain English or Bangla; it finds the right passages, reads
    them, and writes an answer that names the documents it used. It will say
    <em>I don’t know</em> rather than invent something.</p>

    <h3><span class="num">1</span>Everything runs on this computer</h3>

    <div class="callout">
      <p><b>No cloud model. No API key. No account. Your documents and your
      questions never leave this machine.</b></p>
      <p>The language model, the search and the documents are all local, and that has
      been checked rather than assumed: the whole pipeline was re-run with the model
      library forced into offline mode and every stage still worked. Internal HR,
      security and pricing documents never travel to a third-party service in order to
      be searched.</p>
      <p>For completeness, there is exactly <b>one</b> outbound call in the system, and
      it carries no document and no question: on start-up the model library asks
      HuggingFace whether a newer copy of the re-ranker exists. Setting
      <code>HF_HUB_OFFLINE=1</code> removes even that, with no loss of function.</p>
    </div>

    <p>Four models do the work. They are served by <code>Ollama</code>, a local model
    runtime, and all four stay loaded in memory at once so no answer has to wait for a
    model to be swapped in.</p>

    ${table(['Model', 'Job', 'Size on disk', 'Detail'], [
      ['<b>qwen3:4b-instruct-2507</b>', 'Writes the answers',
       '2.5 GB',
       '4 billion parameters, compressed to 4 bits each (<code>Q4_K_M</code>). An ' +
       '<em>instruct</em> build, not a reasoning one — a reasoning model would ' +
       'spend its whole budget thinking out loud and return nothing.'],
      ['<b>bge-m3</b>', 'Understands meaning',
       '1.2 GB',
       'Turns a passage or a question into 1024 numbers. Multilingual, which is why ' +
       'an English question can find a Bangla-only document.'],
      ['<b>qwen3:1.7b</b>', 'Fixes follow-up questions',
       '1.4 GB',
       'Small and fast. Rewrites “and what about sick leave?” into something ' +
       'that can be searched on its own.'],
      ['<b>jina-reranker-v2</b>', 'Re-reads the shortlist',
       '1.1 GB',
       'Runs as ONNX on the CPU, so no PyTorch install is needed. Non-commercial ' +
       'licence — an MIT alternative is configured and must be swapped in before ' +
       'any commercial use.'],
    ])}

    <p class="fnote">The two language models are <b>quantised</b>: each parameter is
    stored in about 4 bits instead of 16, which makes them roughly four times smaller
    and four times cheaper to run, for a small, measured loss of quality. The two
    embedding models are small enough that they are kept at full 16-bit precision.
    Everything is forced onto the CPU — <code>num_gpu: 0</code> — so the
    behaviour is identical on a machine with no usable graphics card.</p>

    <h3><span class="num">2</span>What happens when you ask something</h3>

    <p>In plain terms, five things happen. You can watch all of them live under
    <b>Steps</b> beneath any answer, each with its own sub-steps and its own timing.</p>

    <ol class="plain">
      <li><b>It works out what you really asked.</b> If your question refers back to the
      conversation — “and for contractors?” — a small model rewrites
      it into a complete question first. A greeting never gets this far: it is answered
      instantly.</li>
      <li><b>It searches twice, in two different ways.</b> Once for your exact words,
      once for your meaning. The two are good at opposite things, so both run every
      time and the results are merged.</li>
      <li><b>It shortlists 12 passages and re-reads them.</b> The merge is fast but
      rough. A second, slower model then reads each of the 12 against your actual
      question and re-orders them properly. The best 5 go forward.</li>
      <li><b>It checks that the best one is actually relevant.</b> If nothing clears a
      measured quality bar, it stops and says the documents do not cover this. This is
      the step that prevents confident nonsense.</li>
      <li><b>It writes the answer from those 5 passages only.</b> The passages are given
      to the model as quoted, untrusted material, with instructions to answer only from
      them and to cite each one. The answer streams to your screen as it is written.</li>
    </ol>

    <h3><span class="num">3</span>How the documents were indexed</h3>

    <p>Searching 12 whole documents would be useless — the model can only read a
    few thousand words at a time, and most of a document is irrelevant to any one
    question. So the documents are cut into <b>50 passages</b>, and the passages are
    what gets searched.</p>

    <p>How they are cut matters more than it sounds. A passage must be small enough to
    be precise and large enough to still make sense on its own. Two rules do most of
    the work: <b>a table is never split</b>, because “BDT 2,500” means nothing
    without the “Growth plan” column next to it; and <b>every passage carries
    its heading</b> and a note saying which document it came from and whether that
    document is still in force.</p>

    <div class="flane">Building the index · done once</div>
    <div class="flow">${INGEST}</div>

    <h3><span class="num">4</span>Why there are two searches</h3>

    <p>Neither kind of search is good enough alone, and they fail in opposite
    directions — which is exactly why running both and merging works.</p>

    ${table(['Search', 'Good at', 'Blind to'], [
      ['<b>By words</b><br><small>BM25, a classic keyword index</small>',
       'Exact strings: <code>ND-1042</code>, <code>SEC-EXC</code>, <code>X-Nodi-Key</code>. ' +
       'If you type the code, it finds the code.',
       'Anything worded differently from the document — and every Bangla document, ' +
       'if you asked in English.'],
      ['<b>By meaning</b><br><small>bge-m3 embeddings in a vector index</small>',
       'Recognising that “time off” and “annual leave” are the same ' +
       'thing, in either language.',
       'Precision between near-identical strings: one error code looks much like ' +
       'another.'],
      ['<b>Re-reading</b><br><small>Cross-encoder, runs on the shortlist only</small>',
       'Judging a passage against the full question rather than against a summary of it. ' +
       'It is the most accurate step by a wide margin.',
       'Cost. It is far too slow to run on everything, which is why it only ever sees ' +
       '12 candidates.'],
    ])}

    <div class="flane">Answering a message · every time</div>
    <div class="flow">${ANSWER}</div>

    <h3><span class="num">5</span>Why the first answer is slow and the next is quick</h3>

    <p>Before a model can write a single word, it has to <b>read</b> everything it has
    been given — the instructions, the conversation so far, and the five retrieved
    passages. That reading phase is called <em>prefill</em>, and on a CPU it is most of
    the wait. The writing itself is comparatively quick.</p>

    <p>The saving grace is the <b>KV cache</b>. As the model reads, it builds an
    internal representation of the text so far. If the next request starts with exactly
    the same text, that work can be reused instead of redone.</p>

    <div class="callout">
      <p><b>This is why the system prompt is never personalised.</b> The instruction
      block is about 740 tokens and is byte-identical on every single turn, so it is
      read once and reused from cache forever after. Interpolating even a timestamp
      into it would invalidate the cache and add that cost back to every question.</p>
      <p>The retrieved passages — about 390 tokens — are different every
      question, so they are the part you actually pay for. Ask the <em>same</em>
      question twice and the whole prefix matches: reading drops from <b>5.13 s to 0.08 s</b> for the
      same 1,322 tokens, and the whole answer lands in under three seconds.</p>
    </div>

    <p class="fnote">The cache matches on the longest common <em>prefix</em>, so the
    order of the prompt is deliberate: fixed instructions first, then the conversation,
    then the passages, then your question. Anything that changes is placed as late as
    possible.</p>

    <h3><span class="num">6</span>Speed, measured</h3>

    <p>The project was built on a modest desktop and later moved, unchanged, to a
    faster one. Same code, same models, same settings — so the two columns below
    isolate the effect of the hardware alone.</p>

    ${table(['Stage', 'Original machine', 'This machine'], [
      ['Reading the prompt <small>(prefill — the bottleneck)</small>',
       '48.6 s <small>&nbsp;23.9 tok/s</small>', '<b>5.2 s</b> <small>&nbsp;250.9 tok/s</small>'],
      ['Writing the answer', '5.8 tok/s', '<b>12.7 tok/s</b>'],
      ['Search and re-read', '8.6 s', '<b>0.9 s</b>'],
      ['A typical full answer', '60–90 s', '<b>about 9 s</b>'],
      ['The same question again <small>(prefix cached)</small>',
       '—', '<b>2.7 s</b> <small>&nbsp;prefill 0.08 s</small>'],
      ['A greeting', '~0.2 s', '~0.2 s'],
      ['Building the index', '~45 s', '<b>11.5 s</b>'],
    ])}

    <p class="fnote">The gain is not mostly extra cores — it is <b>AVX-512</b>, a
    CPU instruction set the older chip lacked entirely. It accelerates exactly the
    integer arithmetic that quantised models spend their time on, which is why reading
    got about 10× faster while writing — limited by memory speed, not
    arithmetic — only doubled. Both machines deliberately run with the graphics
    card switched off.</p>

    <h3><span class="num">7</span>Accuracy, and how it is checked</h3>

    <div class="stats">
      <div class="stat"><b>39/40</b><span>answers correct</span></div>
      <div class="stat"><b>36/36</b><span>right document found</span></div>
      <div class="stat"><b>225</b><span>tests passing</span></div>
      <div class="stat"><b>0</b><span>cloud calls</span></div>
    </div>

    <p>Accuracy is not a claim here, it is a run. Forty questions with known answers are
    put through the live system; every answer and every verdict is written to a CSV file
    that can be read afterwards. A run with even one <em>technical</em> error is thrown
    away rather than scored — a crash must never be allowed to look like a wrong
    answer, or like a polite refusal.</p>

    ${table(['What is checked', 'How', 'Result'], [
      ['<b>Did it find the right document?</b>',
       'For each of 36 questions, the passage that must appear in the top 5 is named in ' +
       'advance.', 'All 36'],
      ['<b>Is the answer right?</b>',
       'All 40 answers graded against the expected facts, then audited by hand against ' +
       'the CSV.', '39 of 40'],
      ['<b>Does it refuse when it should?</b>',
       'Questions the documents cannot answer, including one where the documents list ' +
       'some options but not the one asked about.', 'All pass'],
      ['<b>Does it stay fixed?</b>',
       '225 automated tests covering every trap, the streaming format and the failure ' +
       'paths.', 'All pass'],
    ])}

    <div class="callout">
      <p><b>The one known failure, stated plainly.</b> Asked in Bangla what canteen
      lunch costs, the assistant gives the staff price but leaves out the guest price.
      Both are in the passage it retrieved, so this is a writing mistake, not a search
      one.</p>
      <p>It has been left unfixed on purpose. The obvious patch — an instruction
      about canteen prices — would raise the score to 40/40 without making the
      system any better at the next question it has not seen.</p>
    </div>

    <p class="fnote">The automatic grader is itself a 4-billion-parameter local model,
    and it has been caught marking wrongly in <em>both</em> directions — failing
    two correct answers in one run, and passing the canteen answer in another while its
    own written reason said the price for guests was missing. That is why every verdict
    is written out and audited, and why the figure quoted here is the audited one.</p>

    <h3><span class="num">8</span>The traps in the documents</h3>

    <p>The document set is deliberately booby-trapped. Each trap is a realistic failure
    of this kind of system, and each has a specific mechanism and a test that keeps it
    fixed.</p>

    ${table(['Trap', 'How it is handled'], [
      ['<b>An outdated policy that contradicts the current one</b><br>' +
       '<small>An old document says 15 days of leave; the handbook says 18</small>',
       'The document’s status is read from both its filename and its header, and a ' +
       'warning is written <em>into</em> the passage the search engine sees. Its score ' +
       'is then halved. It is demoted, never deleted — the correct answer cites ' +
       '<em>both</em> and says which figure is current.'],
      ['<b>A hidden instruction attack</b><br>' +
       '<small>A vendor notice contains text telling the assistant to lie about pricing</small>',
       'The payload is stripped when the document is indexed, and the strip is logged. ' +
       'The rest of the notice stays indexed, because a genuine question about printer ' +
       'maintenance must still be answerable. Retrieved text is also fenced off in the ' +
       'prompt as untrusted, and asking the assistant to reveal its instructions is ' +
       'refused before any search runs.'],
      ['<b>Exact codes</b><br><small>Ordinary tokenisers shred <code>ND-1042</code> ' +
       'into “nd” and “1042”</small>',
       'A custom tokeniser keeps the whole code <em>and</em> its pieces. A literal match ' +
       'is boosted before re-ranking, and is guaranteed a place on the shortlist so it ' +
       'cannot be voted off by near-misses.'],
      ['<b>Tables, including one trapped in a PDF</b>',
       'A passage holds a whole table or none of it, always with its heading. The PDF’s ' +
       'gridlines are filled rectangles rather than lines, which the parser resolves into ' +
       'real rows.'],
      ['<b>Bangla, and questions asked across languages</b>',
       'Multilingual embeddings carry an English question to a Bangla-only document. ' +
       'Bengali digits are folded to 0-9 for keyword search <em>only</em>, so answers ' +
       'keep their original script on screen.'],
      ['<b>Questions the documents simply cannot answer</b>',
       'Four different refusals. The hardest: when the documents list some options but ' +
       'not the one you asked about, the assistant says it is not mentioned and stops. ' +
       'It never reads silence as a “no”.'],
    ])}

    <h3><span class="num">9</span>What runs where</h3>

    ${table(['Part', 'Choice', 'Why'], [
      ['Model runtime', '<code>Ollama</code>, native chat API',
       'Three model slots so the writer, the embedder and the rewriter all stay in ' +
       'memory. With only two, every follow-up evicted and reloaded a model twice — ' +
       '20–40 wasted seconds per answer, measured.'],
      ['Meaning index', 'Qdrant, embedded in the process',
       'The same API as the server version, saved to a folder, with no Docker and no ' +
       'second process to run.'],
      ['Word index', '<code>rank_bm25</code>',
       'No server needed for 50 passages, and complete control of the tokeniser — ' +
       'which is the part that actually decides whether <code>ND-1042</code> is found.'],
      ['Web server', 'FastAPI with server-sent events',
       'Streams words as they are produced. One HTML page, no build step, no framework.'],
      ['Start-up', 'Eight pre-flight checks',
       'The server refuses to start if a model is missing or an index is stale, and ' +
       'prints the exact command that fixes it — rather than failing on the first ' +
       'question.'],
    ])}

    <p class="fnote">A technical failure and an honest “I don’t know” are
    kept strictly apart, everywhere in the system. They travel through different
    channels, they look different on screen, and the evaluation scores them as different
    outcomes. A broken pipeline that quietly says “I could not find that” would
    be the worst possible failure for a tool whose whole purpose is to be trusted.</p>
  `;
})();
