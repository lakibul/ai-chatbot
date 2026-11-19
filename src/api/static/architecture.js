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

  /* ------------------------------------------- LangChain diagrams ------ */
  /* Drawn as SVG so they scale and follow the light/dark colour tokens. The
     classes (own / lc / ext / warn) are styled in index.html. Three colours
     mean three things: blue = LangChain, plain = this project's own code,
     dashed = a separate program. */
  const LC_MAP = `
  <svg class="lcsvg" viewBox="0 0 840 596" role="img" aria-labelledby="lcm-t lcm-d">
    <title id="lcm-t">Where LangChain sits in the system</title>
    <desc id="lcm-d">The browser talks to the web server, which runs the engine. The engine does
      its own searching, and sends its two model calls through a switch that chooses either the
      native client or LangChain. Both reach the local Ollama server.</desc>
    <defs><marker id="lca1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
      orient="auto"><path class="head" d="M0 0L10 5L0 10z"/></marker></defs>

    <rect class="own" x="320" y="8" width="200" height="40" rx="6"/>
    <text class="t" x="420" y="33" text-anchor="middle">Browser · chat page</text>
    <path class="edge" d="M420 48V84" marker-end="url(#lca1)"/>
    <text class="s" x="432" y="70">events: stage · trace · token · done</text>
    <rect class="own" x="320" y="84" width="200" height="40" rx="6"/>
    <text class="t" x="420" y="109" text-anchor="middle">FastAPI · app.py</text>
    <path class="edge" d="M420 124V158" marker-end="url(#lca1)"/>
    <rect class="own" x="40" y="158" width="760" height="46" rx="6"/>
    <text class="t" x="420" y="178" text-anchor="middle">engine.py · the conductor</text>
    <text class="s" x="420" y="195" text-anchor="middle">runs the 17 steps and sends one event for each</text>

    <path class="edge" d="M220 204V240" marker-end="url(#lca1)"/>
    <text class="s" x="232" y="226">search</text>
    <rect class="grp" x="40" y="240" width="360" height="304" rx="8"/>
    <text class="t" x="56" y="263">Your own code · no LangChain</text>
    <rect class="own" x="56" y="276" width="328" height="34" rx="5"/>
    <text class="s2" x="220" y="298" text-anchor="middle">greeting check · injection guard · memory</text>
    <rect class="own" x="56" y="318" width="328" height="34" rx="5"/>
    <text class="s2" x="220" y="340" text-anchor="middle">BM25 words  +  Qdrant meaning (bge-m3)</text>
    <rect class="own" x="56" y="360" width="328" height="34" rx="5"/>
    <text class="s2" x="220" y="382" text-anchor="middle">merge (RRF) · exact-code boost · shortlist of 12</text>
    <rect class="own" x="56" y="402" width="328" height="34" rx="5"/>
    <text class="s2" x="220" y="424" text-anchor="middle">Jina re-ranker · outdated policies pushed down</text>
    <rect class="own" x="56" y="444" width="328" height="34" rx="5"/>
    <text class="s2" x="220" y="466" text-anchor="middle">relevance floor 0.088 · says “I don’t know”</text>
    <text class="s" x="56" y="498">Why not LangChain here: the Pipeline page also shows the</text>
    <text class="s" x="56" y="513">query tokens, shortlist rules, the floor verdict and the time</text>
    <text class="s" x="56" y="528">of each step. None of that belongs to one passage, so a</text>
    <text class="s" x="56" y="543" dy="0">retriever call has nowhere to report it.</text>

    <path class="edge" d="M620 204V240" marker-end="url(#lca1)"/>
    <text class="s" x="632" y="226">2 model calls: rewrite + answer</text>
    <rect class="own" x="440" y="240" width="360" height="42" rx="6"/>
    <text class="t" x="620" y="258" text-anchor="middle">prompts.py · builds the messages</text>
    <text class="s" x="620" y="274" text-anchor="middle">unchanged — the same text goes out either way</text>
    <path class="edge" d="M620 282V306" marker-end="url(#lca1)"/>
    <rect class="key" x="440" y="306" width="360" height="50" rx="6"/>
    <text class="t" x="620" y="326" text-anchor="middle">backend.py · the switch</text>
    <text class="s" x="620" y="344" text-anchor="middle">USE_LANGCHAIN in .env · read on every call</text>
    <path class="edge" d="M525 356V398" marker-end="url(#lca1)"/>
    <path class="edge" d="M715 356V398" marker-end="url(#lca1)"/>
    <text class="s" x="505" y="382" text-anchor="end">off</text>
    <text class="s" x="727" y="382">on</text>

    <rect class="own" x="440" y="398" width="170" height="72" rx="6"/>
    <text class="t" x="525" y="421" text-anchor="middle">llm.py</text>
    <text class="s" x="525" y="440" text-anchor="middle">native httpx client</text>
    <text class="s" x="525" y="456" text-anchor="middle">the default</text>
    <rect class="lc" x="630" y="398" width="170" height="72" rx="6"/>
    <text class="t" x="715" y="421" text-anchor="middle">lc.py · LangChain</text>
    <text class="s" x="715" y="440" text-anchor="middle">ChatPromptTemplate</text>
    <text class="s" x="715" y="456" text-anchor="middle">| ChatOllama</text>
    <path class="edge" d="M525 470V508" marker-end="url(#lca1)"/>
    <path class="edge" d="M715 470V508" marker-end="url(#lca1)"/>
    <rect class="ext" x="440" y="508" width="360" height="46" rx="6"/>
    <text class="t" x="620" y="528" text-anchor="middle">Ollama server · runs on this computer</text>
    <text class="s" x="620" y="545" text-anchor="middle">qwen3:4b writes · qwen3:1.7b rewrites</text>

    <rect class="lc" x="40" y="566" width="14" height="14" rx="3"/>
    <text class="s" x="60" y="578">LangChain</text>
    <rect class="own" x="150" y="566" width="14" height="14" rx="3"/>
    <text class="s" x="170" y="578">this project’s own code</text>
    <rect class="ext" x="340" y="566" width="14" height="14" rx="3"/>
    <text class="s" x="360" y="578">a separate program</text>
  </svg>`;

  const LC_INSIDE = `
  <svg class="lcsvg" viewBox="0 0 840 356" role="img" aria-labelledby="lci-t lci-d">
    <title id="lci-t">What happens inside lc.py</title>
    <desc id="lci-d">The messages are converted to LangChain message objects, passed through a prompt
      template and into ChatOllama, which streams from Ollama. The text chunks become token events
      and the final chunk becomes the done event.</desc>
    <defs><marker id="lca2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
      orient="auto"><path class="head" d="M0 0L10 5L0 10z"/></marker></defs>

    <rect class="own" x="8" y="18" width="180" height="82" rx="6"/>
    <text class="t" x="98" y="42" text-anchor="middle">① messages</text>
    <text class="s" x="98" y="62" text-anchor="middle">from prompts.py:</text>
    <text class="s" x="98" y="78" text-anchor="middle">a list of {role, content}</text>
    <path class="edge" d="M188 59H224" marker-end="url(#lca2)"/>
    <rect class="lc" x="226" y="18" width="180" height="82" rx="6"/>
    <text class="t" x="316" y="42" text-anchor="middle">② to_messages()</text>
    <text class="s" x="316" y="62" text-anchor="middle">dicts become System /</text>
    <text class="s" x="316" y="78" text-anchor="middle">Human / AI message objects</text>
    <path class="edge" d="M406 59H442" marker-end="url(#lca2)"/>
    <rect class="lc" x="444" y="18" width="180" height="82" rx="6"/>
    <text class="t" x="534" y="42" text-anchor="middle">③ ChatPromptTemplate</text>
    <text class="s" x="534" y="62" text-anchor="middle">one MessagesPlaceholder:</text>
    <text class="s" x="534" y="78" text-anchor="middle">passes them through untouched</text>
    <path class="edge" d="M624 59H660" marker-end="url(#lca2)"/>
    <rect class="lc" x="662" y="18" width="170" height="82" rx="6"/>
    <text class="t" x="747" y="40" text-anchor="middle">④ ChatOllama</text>
    <text class="s" x="747" y="58" text-anchor="middle">num_gpu=0 · num_ctx</text>
    <text class="s" x="747" y="73" text-anchor="middle">num_thread · temperature</text>
    <text class="s" x="747" y="88" text-anchor="middle">reasoning=False · keep_alive</text>

    <path class="edge" d="M747 100V148" marker-end="url(#lca2)"/>
    <text class="s" x="759" y="130">HTTP, streaming</text>
    <rect class="ext" x="662" y="148" width="170" height="70" rx="6"/>
    <text class="t" x="747" y="174" text-anchor="middle">⑤ Ollama /api/chat</text>
    <text class="s" x="747" y="194" text-anchor="middle">sends the text in pieces</text>
    <path class="edge" d="M662 183H628" marker-end="url(#lca2)"/>
    <rect class="lc" x="444" y="148" width="182" height="70" rx="6"/>
    <text class="t" x="535" y="172" text-anchor="middle">⑥ text pieces</text>
    <text class="s" x="535" y="192" text-anchor="middle">each one becomes</text>
    <text class="s" x="535" y="207" text-anchor="middle">a (“token”, text) frame</text>
    <path class="edge" d="M444 183H410" marker-end="url(#lca2)"/>
    <rect class="lc" x="226" y="148" width="182" height="70" rx="6"/>
    <text class="t" x="317" y="172" text-anchor="middle">⑦ last piece</text>
    <text class="s" x="317" y="192" text-anchor="middle">done=true + timings</text>
    <text class="s" x="317" y="207" text-anchor="middle">→ (“done”, metrics)</text>
    <path class="edge" d="M226 183H192" marker-end="url(#lca2)"/>
    <rect class="own" x="8" y="148" width="182" height="70" rx="6"/>
    <text class="t" x="99" y="172" text-anchor="middle">⑧ engine.py</text>
    <text class="s" x="99" y="192" text-anchor="middle">receives the same frames</text>
    <text class="s" x="99" y="207" text-anchor="middle">llm.py would have sent</text>

    <rect class="warn" x="8" y="252" width="268" height="88" rx="6"/>
    <text class="t" x="142" y="276" text-anchor="middle">Errors are translated</text>
    <text class="s" x="142" y="296" text-anchor="middle">library errors become LLMTimeout,</text>
    <text class="s" x="142" y="311" text-anchor="middle">LLMUnavailable, LLMBadResponse,</text>
    <text class="s" x="142" y="326" text-anchor="middle">so the engine needs no change</text>
    <rect class="warn" x="286" y="252" width="268" height="88" rx="6"/>
    <text class="t" x="420" y="276" text-anchor="middle">Cloud tracing forced off</text>
    <text class="s" x="420" y="296" text-anchor="middle">LangSmith would send every question</text>
    <text class="s" x="420" y="311" text-anchor="middle">and passage away. Off even if the</text>
    <text class="s" x="420" y="326" text-anchor="middle">shell says “true” (a test pins it)</text>
    <rect class="own" x="564" y="252" width="268" height="88" rx="6"/>
    <text class="t" x="698" y="276" text-anchor="middle">Stopping really stops</text>
    <text class="s" x="698" y="296" text-anchor="middle">closing the stream frees the model:</text>
    <text class="s" x="698" y="311" text-anchor="middle">the next request took 0.4 s</text>
    <text class="s" x="698" y="326" text-anchor="middle">(measured, not assumed)</text>
  </svg>`;

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

    <h3><span class="num">10</span>LangChain: where it fits</h3>

    <p><b>LangChain</b> is a free toolkit of ready-made parts for programs built around a
    language model: talking to a model, building prompts, chaining steps together. This
    project was first written without it. It has now been added in <b>one narrow place</b>,
    as a switch that can be turned on and off.</p>

    <div class="callout">
      <p><b>Off by default.</b> Put <code>USE_LANGCHAIN=1</code> in <code>.env</code> and
      restart: the two model calls (rewriting a follow-up, and writing the answer) then go
      through LangChain. <code>/health</code> reports <code>"backend": "langchain"</code>, and
      the Pipeline page says so under every message. Everything else — the prompt text, the
      search, the guardrails, the events your browser receives — is the same code either way.</p>
      <p>It changes <em>how the model is called</em>, not <em>what the model says</em>. It is
      not an accuracy improvement, and nothing on this page claims it is.</p>
    </div>

    <div class="flane">The whole system · which part is LangChain</div>
    <p class="lchint">Swipe sideways to see the whole diagram →</p>
    <div class="lcwrap">${LC_MAP}</div>

    ${table(['Part', 'Written with', 'Why'], [
      ['<b>The two model calls</b><br><small>rewrite, answer</small>',
       'LangChain when the switch is on; <code>llm.py</code> when off',
       'This is the only place a toolkit adds anything useful: one standard way to call a ' +
       'model, so trying another model provider later is a one-line change.'],
      ['<b>Prompt text</b>', '<code>prompts.py</code>, unchanged',
       'Every sentence in it was added for a measured reason. LangChain only <em>carries</em> ' +
       'the finished messages; it must not re-interpret them.'],
      ['<b>Search, merge, re-rank, floor</b>', 'This project’s own code',
       'A LangChain retriever returns only a list of passages. Per-passage scores would fit ' +
       'in each passage’s metadata, but the Pipeline page also shows things that belong to no ' +
       'single passage — the query tokens, the shortlist rules, the floor verdict, the time of ' +
       'every step — and a retriever call has no place to report them.'],
      ['<b>Guardrails</b>', 'This project’s own code',
       'They run before and after the model, in plain code you can read and test.'],
      ['<b>Cloud tracing (LangSmith)</b>', 'Forced off',
       'It would send every question and retrieved passage to a third party. This system’s ' +
       'promise is that nothing leaves the machine.'],
    ])}

    <h3><span class="num">11</span>Inside the LangChain part</h3>

    <p>When the switch is on, <code>lc.py</code> does five small things, in this order.
    The file is about 180 lines, and the numbers below match the diagram.</p>

    <p class="lchint">Swipe sideways to see the whole diagram →</p>
    <div class="lcwrap">${LC_INSIDE}</div>

    <ol class="plain">
      <li><b>① ② Turn the messages into objects.</b> <code>prompts.py</code> produces a
      plain list of <code>{role, content}</code>. LangChain wants typed message objects, so
      each one becomes a <code>SystemMessage</code>, <code>HumanMessage</code> or
      <code>AIMessage</code>, in the same order.</li>
      <li><b>③ Pass them through a prompt template.</b> The template holds a single
      placeholder that carries the messages through untouched. This matters: the retrieved
      passages contain <code>{ }</code> and <code>&lt;document&gt;</code>, and a template that tried to
      interpret them would crash or silently change what the model reads. A test pins this.</li>
      <li><b>④ Call the model with the same settings.</b> <code>ChatOllama</code> is given the
      options the native client always sent. <code>num_gpu=0</code> is not cosmetic: without it
      Ollama tries the unusable graphics card. <code>reasoning=False</code> is Ollama’s
      <code>think: false</code>: without it the small rewriter spends its whole budget “thinking”
      and returns nothing. A test checks these reach the actual HTTP request, not just the object.</li>
      <li><b>⑤ ⑥ ⑦ Stream the reply back.</b> Each text piece becomes a <code>token</code>
      frame. The last piece carries Ollama’s <code>done</code> flag and the timings, and becomes
      the <code>done</code> frame. A stream that ends without that flag sends no <code>done</code>
      at all, and the engine treats that as a failure — never as a finished answer.</li>
      <li><b>⑧ The engine cannot tell the difference.</b> It receives exactly the frames the
      native client would have sent, which is why the Pipeline, the Context view and the saved
      conversations needed no change.</li>
    </ol>

    <h3><span class="num">12</span>What you see in the Pipeline after every message</h3>

    <p>The two steps that call a model now say which code made the call, for that message.</p>

    ${table(['Where', 'What it shows', 'When'], [
      ['<b>Rewrite the follow-up</b> and <b>Write the answer</b> — tag',
       '<code>LangChain · ChatOllama</code> or plain <code>Ollama</code>',
       'On every message that ran a model'],
      ['The same two steps, opened — a row',
       '<b>Called through</b>: <code>LangChain (ChatPromptTemplate | ChatOllama)</code> or ' +
       '<code>Native client (httpx)</code>',
       'On every message; absent on old saved messages that predate it'],
      ['A first question has no rewrite step',
       'No tag claiming a backend — no model ran, so nothing is claimed',
       'First message of a conversation'],
    ])}

    <h3><span class="num">13</span>What was checked, and what was not</h3>

    <ul class="plain">
      <li><b>Same request.</b> On identical messages the model reported the same prompt size
      through both paths (313 and 34 tokens) and wrote the same text.</li>
      <li><b>Same events.</b> For a first question and a follow-up, both paths produced the same
      events in the same order, the same 17 Pipeline steps, and word-for-word identical answers.</li>
      <li><b>Stopping works.</b> Abandoning an answer through LangChain freed the model at once;
      the next request took 0.4 s instead of queueing behind it.</li>
      <li><b>Cloud tracing cannot be switched on by accident</b> — tested with the shell
      variable set to “true”.</li>
      <li><b>The full 40-question exam, with LangChain on:</b> 40 of 40 correct, no technical
      errors. The one question the automatic grader has wrongly passed before (the canteen
      price) was checked by hand: the answer contains both the staff and the guest price. The
      run started a few minutes before the Pipeline tag was added; that edit only adds trace
      data, and two questions re-checked on the final code gave identical events and answers on
      both paths.</li>
      <li><b>Not measured:</b> any speed difference. The two paths ran one after the other, and
      the second always found the model’s cache already warm, so a timing comparison would be
      unfair. No speed claim is made.</li>
      <li><b>The cost:</b> three extra packages (<code>langchain-core</code>,
      <code>langchain-ollama</code>, <code>ollama</code>), plus LangChain’s cloud-tracing client
      that arrives with them and is never used. They are loaded only when the switch is on.</li>
    </ul>
  `;
})();
