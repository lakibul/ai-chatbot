'use strict';
/* The per-message pipeline.
 *
 * One pipeline per answer, drawn as a flow: a strip of the five stages along
 * the top, and below it every step on a rail that fills in as the run
 * advances. For each step it shows
 *   - its status (tick = done, clock = waiting, spinner = running,
 *     dash = not needed for this question),
 *   - whether it ran a MODEL, and which one (steps with no model chip are
 *     plain code),
 *   - how long it took,
 *   - the RAW data that went in and came out, kept with the message so a
 *     reopened conversation shows exactly what it showed live.
 *
 * Data comes from `trace` events (engine._node), each
 *   {node, state: 'done'|'skipped'|'error', ms?, model?, data}
 * and is stored with the assistant turn. Nothing is invented: a step with no
 * event is drawn as "not reached" (for example the search steps after a
 * greeting was answered with a fixed reply).
 *
 * All raw text is written with textContent. It contains user input and
 * document text, and a document is untrusted by design. Icons are static
 * constants parsed once; no data ever reaches them.
 */
(function () {
  const SVGNS = 'http://www.w3.org/2000/svg';

  const h = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  };

  /* ------------------------------------------------------------- icons --- */
  /* 24x24 line icons. Each value is a list of [tag, attrs] drawn in order. */
  const ICONS = {
    tick: [['path', { d: 'M5 12.5l4.5 4.5L19 7.5' }]],
    clock: [['circle', { cx: 12, cy: 12, r: 8.5 }], ['path', { d: 'M12 7.5V12l3 2' }]],
    dash: [['path', { d: 'M7.5 12h9' }]],
    bang: [['path', { d: 'M12 7v6.5M12 17h.01' }]],
    fork: [['path', { d: 'M12 4v6M12 10L6.5 18M12 10l5.5 8' }]],
    input: [['path', { d: 'M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z' }]],
    smalltalk: [['circle', { cx: 12, cy: 12, r: 8.5 }], ['path', { d: 'M8.5 14.2a4.2 4.2 0 0 0 7 0M9.2 9.6h.01M14.8 9.6h.01' }]],
    guard: [['path', { d: 'M12 3l7 3v5c0 4.5-3 8-7 10-4-2-7-5.5-7-10V6z' }]],
    memory: [['path', { d: 'M12 3l9 5-9 5-9-5z' }], ['path', { d: 'M3 12.5l9 5 9-5M3 17l9 5 9-5' }]],
    rewrite: [['path', { d: 'M4 20l1-4L16 5l3 3L8 19z' }], ['path', { d: 'M14 7l3 3' }]],
    bm25: [['circle', { cx: 11, cy: 11, r: 6 }], ['path', { d: 'M20 20l-4.5-4.5' }]],
    vector: [['circle', { cx: 6, cy: 7, r: 2.2 }], ['circle', { cx: 18, cy: 9, r: 2.2 }], ['circle', { cx: 9, cy: 18, r: 2.2 }],
             ['path', { d: 'M8 8l8 .8M7 9l1.6 7M16.6 11l-5.6 6' }]],
    fuse: [['path', { d: 'M5 4l7 8 7-8M12 12v8' }]],
    boost: [['path', { d: 'M5 9h14M5 15h14M10 4L8 20M16 4l-2 16' }]],
    window: [['path', { d: 'M4 5h16l-6 8v6l-4-2v-4z' }]],
    rerank: [['path', { d: 'M7 4v16M7 4L4 7M7 4l3 3M17 20V4M17 20l-3-3M17 20l3-3' }]],
    floor: [['path', { d: 'M3 18h18M6 18v-6M10 18V8M14 18v-4M18 18V6' }]],
    prompt: [['path', { d: 'M7 3h7l5 5v13H7z' }], ['path', { d: 'M14 3v5h5M10 13h6M10 17h6' }]],
    llm: [['rect', { x: 7, y: 7, width: 10, height: 10, rx: 1.6 }],
          ['path', { d: 'M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4' }]],
    cite: [['path', { d: 'M6 3h12v18l-6-4-6 4z' }]],
    save: [['ellipse', { cx: 12, cy: 6, rx: 7, ry: 3 }],
           ['path', { d: 'M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3' }]],
    outcome: [['path', { d: 'M5 21V4M5 4h11l-2 4 2 4H5' }]],
    stage1: [['path', { d: 'M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z' }]],
  };

  function icon(name, size) {
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', size || 16);
    svg.setAttribute('height', size || 16);
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.8');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    (ICONS[name] || []).forEach((p) => {
      const e = document.createElementNS(SVGNS, p[0]);
      Object.keys(p[1]).forEach((k) => e.setAttribute(k, p[1][k]));
      svg.appendChild(e);
    });
    return svg;
  }

  const fmtMs = (ms) => {
    if (ms === undefined || ms === null || isNaN(ms)) return '';
    if (ms < 1) return '<1 ms';
    return ms < 950 ? Math.round(ms) + ' ms' : (ms / 1000).toFixed(1) + ' s';
  };
  const short = (m) => (m && m.indexOf('/') >= 0) ? m.slice(m.lastIndexOf('/') + 1) : m;
  const num = (v, d) => (v === undefined || v === null || isNaN(v)) ? '-' : Number(v).toFixed(d === undefined ? 2 : d);
  const plural = (c, one, many) => c + ' ' + (c === 1 ? one : (many || one + 's'));

  /* ------------------------------------------------------------ nodes --- */
  /* role: which kind of model this step runs, or null for plain code. */
  const ROLE = { rewrite: 'LLM', llm: 'LLM', embed: 'Embedding', reranker: 'Reranker' };

  const NODES = {
    input: {
      title: 'Your question', icon: 'input', role: null, tools: ['engine.py'],
      sum: (d) => plural(d.chars, 'character') + (d.truncated ? ' (cut from ' + d.original_chars + ')' : ''),
      raw: (d, c) => [c.kv([['Characters', d.chars], ['Received', d.original_chars], ['Limit', d.limit],
                            ['Cut short', d.truncated ? 'yes' : 'no']]),
                      c.block('The question', d.question)],
    },
    smalltalk: {
      title: 'Greeting shortcut', icon: 'smalltalk', role: null, tools: ['smalltalk.py'],
      sum: (d) => d.matched ? 'A greeting: answered with a fixed reply' : 'Not a greeting, carry on',
      raw: (d, c) => [c.kv([['Whole message is a pleasantry', d.matched ? 'yes' : 'no']]),
                      d.reply ? c.block('Fixed reply', d.reply) : null],
    },
    guard: {
      title: 'Instruction-leak guard', icon: 'guard', role: null, safety: true, tools: ['sanitise.py'],
      sum: (d) => d.refused ? 'Asked for the hidden instructions: declined'
        : 'Checked against ' + (d.patterns || 0) + ' attack patterns, no match',
      raw: (d, c) => [c.kv([['Attack patterns checked', d.patterns],
                            ['Matched', (d.matched || []).length ? d.matched.join(', ') : 'none'],
                            ['Declined', d.refused ? 'yes' : 'no']]),
        c.note('The same patterns stripped the attack planted in the documents, but that ran once at build time ' +
               '(see storage/sanitised.log). This check is the one that ran for this question.')],
    },
    memory: {
      title: 'Recent turns', icon: 'memory', role: null, tools: ['memory.py'],
      sum: (d, f) => f.state === 'skipped' ? 'First question, nothing to remember'
        : plural(d.turns, 'earlier turn') + ' sent along (limit ' + d.limit + ')',
      raw: (d, c) => (d.messages || []).length
        ? (d.messages || []).map((m) => c.block(m.role === 'user' ? 'You said' : 'Assistant said', m.content))
        : [c.note('No earlier turns in this conversation.')],
    },
    rewrite: {
      title: 'Rewrite the follow-up', icon: 'rewrite', role: 'rewrite', tools: ['Ollama'],
      sum: (d, f) => f.state === 'skipped' ? 'Not needed: ' + (d.reason || 'no earlier turns')
        : d.error ? 'Rewriter failed, used the question as asked'
        : d.changed ? 'Search query: “' + d.search_query + '”'
        : 'Already stands alone, used as asked',
      raw: (d, c) => {
        if (d.skipped) return [c.note('No model call was made. ' + (d.reason || ''))];
        const out = [c.kv([['Model', d.model], ['Tokens read', d.prompt_tokens], ['Tokens written', d.output_tokens],
                           ['Seconds', d.seconds]])];
        if (d.system_prompt) out.push(c.block('System prompt (shown because TRACE_SHOW_SYSTEM_PROMPT is on)', d.system_prompt));
        else out.push(c.note('The rewriter’s instructions are withheld, like the main assistant’s.'));
        out.push(c.block('Sent to the model', d.input));
        if (d.error) out.push(c.block('Error', d.error));
        else {
          out.push(c.block('Raw model output', d.raw_output));
          out.push(c.block('Search query used', d.search_query));
        }
        return out;
      },
    },
    bm25: {
      title: 'Keyword search', icon: 'bm25', role: null, tools: ['rank_bm25'],
      sum: (d) => plural((d.hits || []).length, 'passage') + ' matched on words',
      raw: (d, c) => [c.chips('Words searched for', d.query_tokens || []),
        c.table(['#', 'Passage', 'BM25 score'],
          (d.hits || []).map((x, i) => [i + 1, x.id, c.bar(x.score, (d.hits[0] || {}).score, 2)]))],
    },
    vector: {
      title: 'Meaning search', icon: 'vector', role: 'embed', tools: ['Qdrant'],
      sum: (d) => plural((d.hits || []).length, 'passage') + ' matched on meaning' +
        (d.dimension ? ' (' + d.dimension + ' numbers)' : ''),
      raw: (d, c) => [
        d.dimension ? c.kv([['Vector length', d.dimension],
                            ['First 8 numbers', '[' + (d.preview || []).join(', ') + ', ...]'],
                            ['Vector size (norm)', d.norm]]) : null,
        c.table(['#', 'Passage', 'Cosine similarity'],
          (d.hits || []).map((x, i) => [i + 1, x.id, c.bar(x.score, 1, 3)]))],
    },
    fuse: {
      title: 'Merge the two rankings', icon: 'fuse', role: null, tools: ['rrf.py'],
      sum: (d) => plural(d.unique, 'unique passage') + ' after merging (k = ' + d.k + ')',
      raw: (d, c) => [c.table(['#', 'Passage', 'RRF score', 'Keyword rank', 'Meaning rank'],
        (d.hits || []).map((x, i) => [i + 1, x.id, num(x.score, 4),
          (x.ranks || {}).bm25 || '-', (x.ranks || {}).vector || '-']))],
    },
    boost: {
      title: 'Protect exact identifiers', icon: 'boost', role: null, tools: ['code_boost.py'],
      sum: (d, f) => f.state === 'skipped' ? 'No exact code in this question'
        : plural((d.matched || []).length, 'passage') + ' lifted for ' + (d.codes_in_question || []).join(', '),
      raw: (d, c) => [c.chips('Codes found in the question', d.codes_in_question || []),
        (d.matched || []).length
          ? c.table(['Passage', 'Matched', 'Score after boost'],
              d.matched.map((x) => [x.id, (x.matched || []).join(', '), num(x.score, 1)]))
          : c.note('Nothing to boost.')],
    },
    window: {
      title: 'Choose who gets re-read', icon: 'window', role: null, tools: ['pipeline.py'],
      sum: (d) => plural((d.candidates || []).length, 'candidate') + ' shortlisted (limit ' + d.limit + ')',
      raw: (d, c) => [c.table(['#', 'Passage', 'Why it is here'],
        (d.candidates || []).map((x, i) => [i + 1, x.id, x.why]))],
    },
    rerank: {
      title: 'Re-read each candidate', icon: 'rerank', role: 'reranker', tools: ['ONNX Runtime'],
      sum: (d) => !d.available ? 'Reranker unavailable, kept the merge order'
        : plural((d.candidates || []).length, 'candidate') + ' scored, best kept',
      raw: (d, c) => !d.available ? [c.note('Reranker could not run: ' + (d.reason || ''))] : [
        c.kv([['Old-policy multiplier', d.superseded_multiplier]]),
        c.table(['Passage', 'Raw score', '0-1 score', 'Final', 'Kept'],
          (d.candidates || []).map((x) => [x.id, x.logit, num(x.probability, 4),
            c.bar(x.score, 1, 4), (x.kept ? 'kept' : '') + (x.demoted ? ' (old policy, halved)' : '')]))],
    },
    floor: {
      title: 'Is anything relevant?', icon: 'floor', role: null, safety: true, tools: ['engine.py'],
      sum: (d) => !d.applies ? 'Skipped (reranker was not used)'
        : 'Best ' + num(d.top_score, 3) + ' against a floor of ' + num(d.floor, 3) + (d.passed ? ': passed' : ': below, so refuse'),
      raw: (d, c) => [c.kv([['Best passage score', d.top_score], ['Floor', d.floor], ['Ranking used', d.ranking],
                            ['Applies', d.applies ? 'yes' : 'no'], ['Passed', d.passed ? 'yes' : 'no']])],
    },
    prompt: {
      title: 'Build the prompt', icon: 'prompt', role: null, safety: true, tools: ['prompts.py'],
      sum: (d) => plural((d.documents || []).length, 'passage') + ' + question + ' +
        plural(d.history_messages || 0, 'earlier message') +
        (d.fenced ? ' \u00b7 ' + plural(d.fenced, 'passage') + ' fenced as untrusted' : ''),
      raw: (d, c) => [
        c.kv([['Passages fenced as untrusted', d.fenced],
              ['Marked out of date', d.superseded_marked],
              ['Earlier messages included', d.history_messages]]),
        (d.rules || []).length ? c.list('Standing rules in force', d.rules) : null,
        d.system_prompt ? c.block('System prompt (shown because TRACE_SHOW_SYSTEM_PROMPT is on)', d.system_prompt)
          : c.note('The instructions themselves are withheld, because the assistant refuses to reveal them. The rules above say what they enforce.'),
        c.table(['Passage', 'Section', 'Status'], (d.documents || []).map((x) => [x.id, x.section, x.status])),
        c.block('Exactly what is sent as the user message', d.user_message)],
    },
    llm: {
      title: 'Write the answer', icon: 'llm', role: 'llm', tools: ['Ollama'],
      sum: (d) => plural(d.prompt_tokens || 0, 'token') + ' read in ' + num(d.prefill_s, 1) + ' s, ' +
        plural(d.output_tokens || 0, 'token') + ' written in ' + num(d.generate_s, 1) + ' s',
      raw: (d, c) => [
        c.kv([['Tokens read', d.prompt_tokens], ['Read speed (tokens/s)', d.read_tokens_per_s],
              ['Tokens written', d.output_tokens], ['Write speed (tokens/s)', d.write_tokens_per_s],
              ['Model load (s)', d.load_s]]),
        c.kv(Object.keys(d.options || {}).map((k) => ['option ' + k, d.options[k]])),
        c.block('Raw model output', d.answer_raw)],
    },
    cite: {
      title: 'Pick the sources to show', icon: 'cite', role: null, tools: ['engine.py'],
      sum: (d) => plural((d.shown || []).length, 'source') + ' shown of ' + (d.retrieved || []).length + ' read',
      raw: (d, c) => [c.kv([['Cited by the model', (d.named_in_answer || []).join(', ') || 'none'],
                            ['Read', (d.retrieved || []).join(', ')],
                            ['Shown', (d.shown || []).join(', ')],
                            ['No citation found, so all shown', d.fallback_all ? 'yes' : 'no']])],
    },
    save: {
      title: 'Save to history', icon: 'save', role: null, tools: ['memory.py'],
      sum: (d) => d.saved ? 'Saved with this conversation' : 'Not saved',
      raw: () => [],
    },
    outcome: {
      title: 'Result', icon: 'outcome', role: null, tools: [],
      sum: (d) => d.text || d.kind,
      raw: (d, c) => [c.kv([['Outcome', d.kind]])],
    },
  };

  /* Five stages. A step id inside an array of arrays is a side-by-side pair. */
  const STAGES = [
    { key: 'understand', label: 'Understand', title: 'Understand the question',
      rows: [['input'], ['smalltalk'], ['guard'], ['memory'], ['rewrite']] },
    { key: 'search', label: 'Search', title: 'Search the documents',
      rows: [['bm25', 'vector'], ['fuse'], ['boost'], ['window'], ['rerank']] },
    { key: 'decide', label: 'Decide', title: 'Decide what to use',
      rows: [['floor'], ['prompt']] },
    { key: 'write', label: 'Write', title: 'Write the answer',
      rows: [['llm']] },
    { key: 'deliver', label: 'Deliver', title: 'Deliver and save',
      rows: [['cite'], ['save'], ['outcome']] },
  ];
  /* The algorithm or library behind each step, shown as an outlined tag. A step
     that also runs a model gets a filled tag with the model's own name. */
  const TECH = {
    input: 'engine.py',
    smalltalk: 'phrase list',
    guard: 'regex patterns',
    memory: 'memory.py',
    rewrite: 'Ollama',
    bm25: 'BM25 · rank_bm25',
    vector: 'Qdrant · cosine',
    fuse: (f) => 'RRF' + (f && f.data && f.data.k ? ' · k=' + f.data.k : ' · rank fusion'),
    boost: 'regex code match',
    window: 'shortlist rules',
    rerank: 'cross-encoder · ONNX Runtime',
    floor: 'score threshold',
    prompt: 'prompts.py',
    llm: 'Ollama',
    cite: 'citation parser',
    save: 'JSON file',
    outcome: '',
  };
  /* What a step is known to have done when only the progress events arrived
     (no raw data), so a step the stream has moved past still reads sensibly. */
  const IMPLIED = {
    input: 'Question received',
    smalltalk: 'Not a greeting, carry on',
    guard: 'No attempt to extract instructions',
    memory: 'Earlier turns checked',
  };
  const ORDER = [];
  STAGES.forEach((s) => s.rows.forEach((r) => r.forEach((id) => ORDER.push(id))));
  const STAGE_OF = {};
  STAGES.forEach((s) => s.rows.forEach((r) => r.forEach((id) => { STAGE_OF[id] = s.key; })));

  /* --------------------------------------------------- raw-data helpers --- */
  const helpers = {
    kv(pairs) {
      const t = h('dl', 'pl-kv');
      pairs.forEach((p) => {
        if (p[1] === undefined || p[1] === null || p[1] === '') return;
        t.appendChild(h('dt', null, p[0]));
        t.appendChild(h('dd', null, String(p[1])));
      });
      return t;
    },
    block(title, text) {
      const w = h('div', 'pl-block');
      w.appendChild(h('div', 'pl-bt', title));
      w.appendChild(h('pre', 'pl-pre', text === undefined || text === null ? '' : String(text)));
      return w;
    },
    note(text) { return h('p', 'pl-note', text); },
    list(title, items) {
      const w = h('div', 'pl-block');
      w.appendChild(h('div', 'pl-bt', title));
      const ul = h('ul', 'pl-list');
      items.forEach((t) => ul.appendChild(h('li', null, t)));
      w.appendChild(ul);
      return w;
    },
    chips(title, list) {
      const w = h('div', 'pl-block');
      w.appendChild(h('div', 'pl-bt', title));
      const row = h('div', 'pl-chips');
      if (!list.length) row.appendChild(h('span', 'pl-note', 'none'));
      list.forEach((t) => row.appendChild(h('code', 'pl-chip', t)));
      w.appendChild(row);
      return w;
    },
    bar(value, max, digits) {
      const cell = h('span', 'pl-barcell');
      const v = Number(value);
      cell.appendChild(h('span', 'pl-num', num(v, digits === undefined ? 3 : digits)));
      const track = h('span', 'pl-bar');
      const i = h('i');
      i.style.width = Math.round((max > 0 ? Math.max(0, Math.min(1, v / max)) : 0) * 100) + '%';
      track.appendChild(i);
      cell.appendChild(track);
      return cell;
    },
    table(heads, rows) {
      const wrap = h('div', 'pl-scroll');
      const t = h('table', 'pl-table');
      const hd = h('tr');
      heads.forEach((x) => hd.appendChild(h('th', null, x)));
      const th = h('thead'); th.appendChild(hd); t.appendChild(th);
      const tb = h('tbody');
      const idCol = heads.indexOf('Passage');
      rows.forEach((r) => {
        const tr = h('tr');
        r.forEach((cell, i) => {
          const td = h('td', i === idCol ? 'pl-mono' : '');
          if (cell instanceof Node) td.appendChild(cell);
          else td.textContent = String(cell === undefined || cell === null ? '' : cell);
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      t.appendChild(tb);
      wrap.appendChild(t);
      return wrap;
    },
  };

  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch (e) { /* fall through */ }
    try {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      const ok = document.execCommand('copy');
      ta.remove();
      return ok;
    } catch (e) { return false; }
  }

  /* ----------------------------------------------------------- status --- */
  const STATUS_ICON = { done: 'tick', pending: 'clock', skipped: 'dash', error: 'bang', unreached: null, live: null };
  function setDot(dot, state, size) {
    dot.className = dot.className.replace(/\bpl-s-\w+\b/g, '').trim() + ' pl-s-' + state;
    dot.textContent = '';
    const name = STATUS_ICON[state];
    if (name) dot.appendChild(icon(name, size || 13));
  }

  /* ------------------------------------------------------------ create --- */
  function create(container, opts) {
    const live = !(opts && opts.replay);
    /* Configured model names by role (llm, rewrite, embed, reranker), so a
       step that has not run yet already names the model it will use. */
    const models = (opts && opts.models) || (() => ({}));
    let lastReading = null, lastWriting = null;
    const frames = {};
    const parts = {};              // node id -> {el, row, sum, ms, tag, raw, def}
    const steps = [];              // timeline entries in order: {el, dot, ids, kind}
    const stageEls = {};           // stage key -> {item, dot, time, head}
    let finished = false;

    /* ----- top bar: legend + open all ----- */
    const top = h('div', 'pl-top');
    const legend = h('div', 'pl-legend');
    /* The key names the models themselves, not just "a model runs here":
       filled tags are the models of this pipeline, outlined tags are the
       algorithm or library behind a step that runs no model. */
    const legendModels = h('span', 'pl-key pl-models');
    const kTech = h('span', 'pl-key');
    kTech.appendChild(h('span', 'pl-tag pl-tag-tech', 'name'));
    kTech.appendChild(document.createTextNode('algorithm or library, no model'));
    const kSafe = h('span', 'pl-key');
    kSafe.appendChild(h('span', 'pl-tag pl-tag-safe', 'Guardrail'));
    kSafe.appendChild(document.createTextNode('a safety check'));
    legend.appendChild(legendModels); legend.appendChild(kTech); legend.appendChild(kSafe);
    top.appendChild(legend);
    const openAll = h('button', 'pl-btn', 'Open all raw data');
    openAll.type = 'button';
    top.appendChild(openAll);
    container.appendChild(top);
    const unavailable = h('p', 'pl-note pl-unavail',
      'Steps marked "no raw data" were reported by the server as progress only. This answer was produced by a ' +
      'server that did not record raw data (restart it to record it for new answers), so model names on those ' +
      'steps are the currently configured ones.');
    unavailable.hidden = true;
    container.appendChild(unavailable);

    /* ----- stage strip ----- */
    const strip = h('ol', 'pl-strip');
    STAGES.forEach((s, i) => {
      const item = h('li', 'pl-stage pl-s-pending');
      const btn = h('button', 'pl-stagebtn');
      btn.type = 'button';
      btn.title = s.title;
      const dot = h('span', 'pl-sdot');
      setDot(dot, 'pending', 14);
      const label = h('span', 'pl-slabel', s.label);
      const time = h('span', 'pl-stime');
      btn.appendChild(dot); btn.appendChild(label); btn.appendChild(time);
      btn.addEventListener('click', () => {
        const hd = stageEls[s.key].head;
        if (hd && hd.scrollIntoView) hd.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
      item.appendChild(btn);
      strip.appendChild(item);
      stageEls[s.key] = { item, dot, time, btn, idx: i };
    });
    container.appendChild(strip);

    /* ----- the rail ----- */
    const rail = h('ol', 'pl-rail');
    const progress = h('div', 'pl-progress');
    rail.appendChild(progress);
    container.appendChild(rail);

    function makeRow(id, boxed) {
      const def = NODES[id];
      const row = h('div', 'pl-row');
      const head = h('button', 'pl-head');
      head.type = 'button';
      head.setAttribute('aria-expanded', 'false');
      head.appendChild(h('span', 'pl-ico')).appendChild(icon(def.icon, 16));
      head.appendChild(h('span', 'pl-title', def.title));
      const ms = h('span', 'pl-ms');
      const chev = h('span', 'pl-chev');
      head.appendChild(h('span', 'pl-grow'));
      head.appendChild(ms); head.appendChild(chev);
      const tags = h('div', 'pl-tags');
      const sum = h('div', 'pl-sum', '');
      row.appendChild(head); row.appendChild(tags); row.appendChild(sum);
      const raw = h('div', 'pl-raw');
      raw.hidden = true;
      head.addEventListener('click', () => toggleRaw(id));
      parts[id] = { row, head, sum, ms, tags, raw, def, boxed };
      return { row, raw };
    }

    function toggleRaw(id, force) {
      const p = parts[id];
      if (!frames[id] || frames[id].synthetic) return;
      const open = force === undefined ? p.raw.hidden : force;
      p.raw.hidden = !open;
      p.head.setAttribute('aria-expanded', String(open));
      if (open && !p.raw.dataset.built) fillRaw(id);
      syncProgress();
    }

    function fillRaw(id) {
      const p = parts[id], f = frames[id];
      p.raw.textContent = '';
      p.raw.dataset.built = '1';
      const tools = h('div', 'pl-tools');
      if (f.model) tools.appendChild(h('span', 'pl-tool pl-toolmodel', short(f.model)));
      (p.def.tools || []).forEach((t) => tools.appendChild(h('span', 'pl-tool', t)));
      p.raw.appendChild(tools);
      let out = [];
      try { out = p.def.raw(f.data || {}, helpers, f) || []; } catch (e) { out = [helpers.note('Could not render this step: ' + e.message)]; }
      out.filter(Boolean).forEach((x) => p.raw.appendChild(x));
      const copy = h('button', 'pl-btn pl-copy', 'Copy raw JSON');
      copy.type = 'button';
      copy.addEventListener('click', async () => {
        const ok = await copyText(JSON.stringify(f.data || {}, null, 2));
        copy.textContent = ok ? 'Copied' : 'Copy failed';
        setTimeout(() => { copy.textContent = 'Copy raw JSON'; }, 1400);
      });
      p.raw.appendChild(copy);
    }

    STAGES.forEach((s) => {
      const head = h('li', 'pl-stagehead');
      head.appendChild(h('span', 'pl-sn', String(stageEls[s.key].idx + 1)));
      head.appendChild(h('span', 'pl-st', s.title));
      const total = h('span', 'pl-stotal');
      head.appendChild(h('span', 'pl-grow'));
      head.appendChild(total);
      rail.appendChild(head);
      stageEls[s.key].head = head;
      stageEls[s.key].total = total;

      s.rows.forEach((ids) => {
        const li = h('li', 'pl-step pl-s-pending');
        const dot = h('span', 'pl-dot');
        setDot(dot, 'pending');
        li.appendChild(dot);
        const main = h('div', 'pl-main');
        li.appendChild(main);
        if (ids.length === 1) {
          const r = makeRow(ids[0]);
          main.appendChild(r.row);
          main.appendChild(r.raw);
          steps.push({ el: li, dot, ids, stage: s.key });
        } else {
          /* A real fork: two branches that run side by side, then merge in
             the next step. Their raw data opens full width underneath, since
             a table does not fit in half a column. */
          main.appendChild(h('div', 'pl-forklabel')).appendChild(icon('fork', 13));
          main.firstChild.appendChild(document.createTextNode(' Two searches run side by side'));
          const grid = h('div', 'pl-branches');
          const raws = [];
          ids.forEach((id) => {
            const r = makeRow(id, true);
            r.row.classList.add('pl-branch');
            grid.appendChild(r.row);
            raws.push(r.raw);
          });
          raws.forEach((r) => { r.classList.add('pl-rawwide'); grid.appendChild(r); });
          main.appendChild(grid);
          steps.push({ el: li, dot, ids, stage: s.key, fork: true });
        }
        rail.appendChild(li);
      });
    });

    openAll.addEventListener('click', () => {
      const real = (id) => frames[id] && !frames[id].synthetic;
      const anyClosed = ORDER.some((id) => real(id) && parts[id].raw.hidden);
      ORDER.forEach((id) => { if (real(id)) toggleRaw(id, anyClosed); });
      openAll.textContent = anyClosed ? 'Close all raw data' : 'Open all raw data';
    });

    /* ------------------------------------------------------ painting --- */
    function stateOf(id) {
      const f = frames[id];
      if (f) return f.state === 'skipped' ? 'skipped' : f.state === 'error' ? 'error' : 'done';
      if (finished) return 'unreached';
      return parts[id].row.classList.contains('pl-live') ? 'live' : 'pending';
    }

    function paintStep(st) {
      const states = st.ids.map(stateOf);
      let s;
      if (states.every((x) => x === 'unreached')) s = 'unreached';
      else if (states.some((x) => x === 'error')) s = 'error';
      else if (states.some((x) => x === 'live')) s = 'live';
      else if (states.every((x) => x === 'done' || x === 'skipped' || x === 'unreached')) {
        s = states.every((x) => x === 'skipped') ? 'skipped' : 'done';
      } else s = 'pending';
      st.el.className = 'pl-step pl-s-' + s;
      setDot(st.dot, s);
      st.state = s;
    }

    function addTag(p, cls, text, title) {
      const t = h('span', cls, text);
      if (title) t.title = title;
      p.tags.appendChild(t);
    }

    /* Tags name what runs the step: a filled tag for the model, an outlined
       tag for the algorithm or library. They show before the step runs too. */
    function paintTags(id) {
      const p = parts[id], f = frames[id], role = ROLE[p.def.role];
      p.tags.textContent = '';
      if (role) {
        if (f && f.state === 'skipped') {
          /* A model step that was not needed says so, and shows no model name,
             which would suggest it ran. */
          addTag(p, 'pl-tag pl-tag-off', role + ' · not used');
        } else {
          const name = (f && f.model) || models()[p.def.role];
          addTag(p, 'pl-tag', role + (name ? ' · ' + short(name) : ''), name || '');
        }
      }
      /* A step that exists to stop something bad, not to find an answer. */
      if (p.def.safety) addTag(p, 'pl-tag pl-tag-safe', 'Guardrail');
      const tech = typeof TECH[id] === 'function' ? TECH[id](f) : TECH[id];
      if (tech) addTag(p, 'pl-tag pl-tag-tech', tech);
    }

    function paintNode(id) {
      const p = parts[id], f = frames[id];
      const s = stateOf(id);
      p.row.classList.remove('pl-live');
      p.row.dataset.state = s;
      /* A step known only from progress events has nothing to open. */
      p.head.classList.toggle('pl-has', !!f && !f.synthetic);
      paintTags(id);
      if (f) {
        let text = '';
        if (f.synthetic) text = f.text || '';
        else { try { text = p.def.sum(f.data || {}, f); } catch (e) { text = ''; } }
        p.sum.textContent = text;
        p.ms.textContent = f.state === 'skipped' ? '' : fmtMs(f.ms);
        if (f.synthetic && finished) p.sum.textContent = (text ? text + ' ' : '') + '(no raw data)';
        if (p.raw.dataset.built && !f.synthetic) fillRaw(id);
      } else if (s === 'unreached') {
        p.sum.textContent = 'Not reached: an earlier step ended this question';
      }
    }

    function stageState(key) {
      const ids = ORDER.filter((id) => STAGE_OF[id] === key);
      const have = ids.filter((id) => frames[id]);
      if (finished) return have.length ? (have.some((id) => frames[id].state === 'error') ? 'error' : 'done') : 'unreached';
      if (have.length === ids.length) return 'done';
      const next = ORDER.find((id) => !frames[id]);
      if (live && STAGE_OF[next] === key) return 'live';
      return 'pending';
    }

    function paintStages() {
      STAGES.forEach((s, i) => {
        const e = stageEls[s.key];
        const st = stageState(s.key);
        e.item.className = 'pl-stage pl-s-' + st;
        setDot(e.dot, st, 14);
        const ids = ORDER.filter((id) => STAGE_OF[id] === s.key);
        const ms = ids.reduce((a, id) => a + ((frames[id] && frames[id].ms) || 0), 0);
        e.time.textContent = ms > 0 ? fmtMs(ms) : '';
        e.total.textContent = ms > 0 ? fmtMs(ms) : '';
        const prev = i ? stageState(STAGES[i - 1].key) : 'done';
        e.item.classList.toggle('pl-linked', prev === 'done');
      });
    }

    /* Which step is "working now": the first with no data yet (both halves of
       a fork together). Meaningless once the run is over. */
    function paintLive() {
      ORDER.forEach((id) => parts[id].row.classList.remove('pl-live'));
      if (live && !finished) {
        const next = ORDER.find((id) => !frames[id]);
        if (next) {
          const st = steps.find((x) => x.ids.indexOf(next) >= 0);
          st.ids.forEach((id) => { if (!frames[id]) parts[id].row.classList.add('pl-live'); });
        }
      }
      steps.forEach(paintStep);
      paintStages();
      paintLegend();
      syncProgress();
    }

    /* The models of this pipeline, by name, in the order they run. A name comes
       from the data when the step has run, else from the configured models. A
       model that was not needed for this question is shown as such. */
    function paintLegend() {
      legendModels.textContent = '';
      legendModels.appendChild(h('span', 'pl-keylabel', 'Models'));
      ['rewrite', 'embed', 'reranker', 'llm'].forEach((role) => {
        const id = ORDER.find((x) => NODES[x].role === role);
        const f = frames[id];
        const name = (f && f.model) || models()[role];
        const label = ROLE[role];
        if (f && f.state === 'skipped') {
          const t = h('span', 'pl-tag pl-tag-off', label + ' · ' + (name ? short(name) : '') + ' (not used)');
          if (name) t.title = name;
          legendModels.appendChild(t);
        } else {
          const t = h('span', 'pl-tag', label + (name ? ' · ' + short(name) : ''));
          if (name) t.title = name;
          legendModels.appendChild(t);
        }
      });
    }

    /* The green rail: filled down to the last step that has finished. */
    function syncProgress() {
      let last = null;
      steps.forEach((st) => { if (st.state === 'done' || st.state === 'skipped' || st.state === 'error') last = st; });
      if (!last) { progress.style.height = '0px'; return; }
      progress.style.height = (last.el.offsetTop + 11) + 'px';
    }

    ORDER.forEach((id) => { paintNode(id); });
    paintLive();

    /* ---- progress-only frames -------------------------------------------
       The step list is driven by `stage` events. The pipeline follows the same
       events, so it can never stall on step one just because raw data (the
       `trace` events) did not arrive, for example from a server that has not
       been restarted. Raw data, when it comes, replaces these frames. */
    function synth(id, state, text, ms, model) {
      if (frames[id] && !frames[id].synthetic) return;     // real data wins
      frames[id] = { node: id, state: state, ms: ms, data: {}, text: text || '',
                     synthetic: true, model: model || undefined };
      paintNode(id);
    }
    function implyBefore(id) {
      ORDER.slice(0, ORDER.indexOf(id)).forEach((x) => { if (!frames[x]) synth(x, 'done', IMPLIED[x] || ''); });
    }

    return {
      /* One trace event. Unknown node ids (a newer server) are ignored. */
      add(frame) {
        if (!frame || !NODES[frame.node]) {
          if (frame && frame.node === 'error') {
            const d = frame.data || {};
            container.appendChild(h('div', 'pl-fail', 'Stopped in ' + (d.where || 'the pipeline') + ': ' + (d.message || '')));
          }
          return;
        }
        frames[frame.node] = frame;
        paintNode(frame.node);
        paintLive();
      },
      /* A progress event, exactly as the step list receives it. */
      stage(p) {
        if (!p) return;
        const ms = p.ms || {};
        if (p.stage === 'rewriting') {
          implyBefore('memory');
        } else if (p.stage === 'rewritten') {
          implyBefore('rewrite');
          synth('rewrite', 'done', 'Search query: \u201c' + p.query + '\u201d');
        } else if (p.stage === 'searching') {
          implyBefore('rewrite');
          if (!frames.rewrite) synth('rewrite', 'skipped', 'Used exactly as asked');
        } else if (p.stage === 'searched') {
          implyBefore('bm25');
          synth('bm25', 'done', plural(p.lexical || 0, 'passage') + ' matched on words', ms.bm25);
          synth('vector', 'done', plural(p.vector || 0, 'passage') + ' matched on meaning', ms.vector);
          synth('fuse', 'done', plural(p.fused || 0, 'unique passage') + ' after merging', ms.fuse);
          synth('boost', p.boosted ? 'done' : 'skipped',
            p.boosted ? plural(p.boosted, 'passage') + ' lifted for an exact code' : 'No exact code in this question');
          synth('window', 'done', plural(p.candidates || 0, 'candidate') + ' shortlisted');
          if (p.ranking === 'rerank') synth('rerank', 'done', plural(p.candidates || 0, 'candidate') + ' scored', ms.rerank);
          else synth('rerank', 'skipped', 'Reranker unavailable, kept the merge order');
        } else if (p.stage === 'reading') {
          lastReading = p;
          implyBefore('floor');
          synth('floor', 'done', 'Best passage clears the relevance floor');
        } else if (p.stage === 'writing') {
          implyBefore('prompt');
          synth('prompt', 'done', (lastReading && lastReading.sources ? plural(lastReading.sources, 'passage') + ' + question' : 'Prompt built'));
          if (p.model && frames.llm) frames.llm.model = p.model;
          lastWriting = p;
        } else {
          return;
        }
        paintLive();
      },
      /* The server's own timings, which arrive with the final `done` event. */
      timings(t) {
        if (!t) return;
        implyBefore('llm');
        const ms = ((t.prefill_s || 0) + (t.generate_s || 0)) * 1000;
        synth('llm', 'done', plural(t.prompt_tokens || 0, 'token') + ' read in ' + num(t.prefill_s, 1) + ' s, ' +
          plural(t.output_tokens || 0, 'token') + ' written in ' + num(t.generate_s, 1) + ' s', ms,
          lastWriting && lastWriting.model);
        synth('cite', 'done', 'Sources listed');
        synth('save', 'done', 'Saved with this conversation');
        synth('outcome', 'done', 'Answer streamed to the browser.');
        paintLive();
      },
      /* The run is over: whatever never reported is shown as "not reached". */
      finish() {
        finished = true;
        ORDER.forEach(paintNode);
        unavailable.hidden = !ORDER.some((id) => frames[id] && frames[id].synthetic);
        paintLive();
      },
      fail() { this.finish(); },
      count() { return Object.keys(frames).length; },
      start() { paintLive(); },
      /* The panel may be shown after being built hidden; offsets are only
         valid once it is visible. */
      relayout() { syncProgress(); },
    };
  }

  window.NodiPipeline = { create: create, NODES: NODES, STAGES: STAGES, ICONS: ICONS };
})();
