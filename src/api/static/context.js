'use strict';
/* The per-message Context view.
 *
 * A model reads and writes inside a fixed-size "context window", measured in
 * tokens. Everything shares it: the standing instructions, the earlier turns,
 * the retrieved passages, the question, and the answer as it is written. This
 * panel shows, for each model call behind one answer, how much of that window
 * was used and by what.
 *
 * Where the numbers come from (engine._context_report):
 *   - token counts  — measured by the model itself, then subtracted, so the
 *                     parts always add up to the total Ollama reported;
 *   - character counts — the exact length of the strings that were sent.
 * Nothing here is estimated. If the split could not be measured, the panel
 * says so and shows totals only.
 *
 * Everything is written with textContent; no data reaches innerHTML.
 */
(function () {
  const h = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  };
  const nf = (v) => Number(v || 0).toLocaleString('en-US');
  const pct = (part, whole) => (whole > 0 ? (part / whole) * 100 : 0);
  const pctText = (part, whole) => {
    const p = pct(part, whole);
    return (p > 0 && p < 1 ? '<1' : Math.round(p)) + '%';
  };
  const short = (m) => (m && m.indexOf('/') >= 0) ? m.slice(m.lastIndexOf('/') + 1) : m;

  /* What each part of the window actually is, in plain words. */
  const WHAT = {
    instructions: 'the standing rules, sent again with every question',
    history: 'the earlier turns of this chat, sent again because the model remembers nothing',
    conversation: 'your earlier turns and this follow-up',
    passages: 'the document passages the search picked',
    question: 'your question, plus the line telling it to answer completely',
    output: 'what the model wrote back',
    free: 'never used',
  };
  const OUTPUT_LABEL = 'The answer it wrote';

  function used(p) { return (p.prompt_tokens || 0) + (p.output_tokens || 0); }

  /* The figure on the button: the answer call's share of the window. */
  function percent(ctx) {
    if (!ctx || !ctx.processes || !ctx.processes.length) return null;
    return Math.round(pct(used(ctx.processes[ctx.processes.length - 1]), ctx.num_ctx));
  }

  /* One row: name and numbers on the first line, what it is on the second. */
  function row(list, key, label, tokens, chars, win) {
    const li = h('li', 'cx-row');
    li.appendChild(h('span', 'cx-sw cx-k-' + key));
    const text = h('span', 'cx-label');
    text.appendChild(h('span', 'cx-name', label));
    const bits = [];
    if (WHAT[key]) bits.push(WHAT[key]);
    if (chars) bits.push(nf(chars) + ' characters of text');
    if (bits.length) text.appendChild(h('span', 'cx-desc', bits.join(' · ')));
    li.appendChild(text);
    li.appendChild(h('span', 'cx-grow'));
    li.appendChild(h('span', 'cx-tokens', nf(tokens)));
    li.appendChild(h('span', 'cx-share', pctText(tokens, win)));
    list.appendChild(li);
  }

  function section(p, index, total, ctx) {
    const win = ctx.num_ctx || 1;
    const box = h('section', 'cx-proc');

    const head = h('div', 'cx-head');
    head.appendChild(h('span', 'cx-step', total > 1 ? (index + 1) + ' of ' + total : ''));
    head.appendChild(h('span', 'cx-title', p.title));
    if (p.model) {
      const tag = h('span', 'cx-model', short(p.model));
      tag.title = p.model;
      head.appendChild(tag);
    }
    head.appendChild(h('span', 'cx-grow'));
    head.appendChild(h('span', 'cx-total',
      nf(used(p)) + ' of ' + nf(win) + ' tokens used · ' + pctText(used(p), win)));
    box.appendChild(head);

    /* What filled the window, in the order it was sent, then what it wrote. */
    const parts = (p.segments || []).map((s) => ({ key: s.key, label: s.label, tokens: s.tokens, chars: s.chars }));
    const measured = parts.length > 0;
    if (!measured) {
      parts.push({ key: 'instructions', label: 'Everything that was sent', tokens: p.prompt_tokens });
    }
    parts.push({ key: 'output', label: OUTPUT_LABEL, tokens: p.output_tokens || 0, chars: p.output_chars });
    const free = Math.max(win - used(p), 0);

    const bar = h('div', 'cx-bar');
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', p.title + ': ' + nf(used(p)) + ' of ' + nf(win) + ' tokens used');
    parts.forEach((s) => {
      if (!(s.tokens > 0)) return;
      const seg = h('span', 'cx-seg cx-k-' + s.key);
      seg.style.width = pct(s.tokens, win) + '%';
      seg.title = s.label + ': ' + nf(s.tokens) + ' tokens';
      bar.appendChild(seg);
    });
    box.appendChild(bar);

    const list = h('ul', 'cx-legend');
    parts.forEach((s) => row(list, s.key, s.label, s.tokens, s.chars, win));
    row(list, 'free', 'Room left over', free, 0, win);
    box.appendChild(list);

    if (!measured) {
      box.appendChild(h('p', 'cx-note', 'The breakdown could not be measured for this call, so only the totals are shown.'));
    }

    /* Would it still have fitted if the answer had run to its full length? */
    const reserve = p.max_output || 0;
    const spare = win - (p.prompt_tokens || 0) - reserve;
    let verdict;
    if (used(p) > win) {
      verdict = h('p', 'cx-verdict cx-bad',
        'Over the limit. When that happens the oldest text is dropped silently, with no error, so this answer may rest on incomplete input.');
    } else if (spare < 0) {
      verdict = h('p', 'cx-verdict cx-warn',
        'It fitted this time, but only because the answer was short. A full-length answer of ' + nf(reserve) +
        ' tokens would have needed ' + nf(-spare) + ' more than the window holds.');
    } else {
      verdict = h('p', 'cx-verdict',
        'Comfortable: ' + nf(spare) + ' tokens would still have been spare even if the answer had run to its full ' +
        nf(reserve) + '-token limit.');
    }
    box.appendChild(verdict);
    return box;
  }

  /* Teach what a token is using THIS message's own measured numbers. */
  function tokenLine(ctx) {
    for (const p of ctx.processes) {
      for (const s of (p.segments || [])) {
        if (s.key === 'passages' && s.tokens > 0 && s.chars > 0) {
          return 'In this answer, one token was worth about ' + (s.chars / s.tokens).toFixed(1) +
            ' characters of the document text. That ratio is not fixed: Bangla needs far more tokens ' +
            'than English for the same number of characters, so a Bangla answer fills the window faster.';
        }
      }
    }
    return null;
  }

  function render(container, ctx) {
    container.textContent = '';
    if (!ctx || !ctx.processes || !ctx.processes.length) {
      container.appendChild(h('p', 'cx-note', 'No context data was recorded for this answer.'));
      return;
    }
    const n = ctx.processes.length;

    container.appendChild(h('p', 'cx-intro',
      'A model has no memory. Everything it needs must be sent again, every time, and it all has to fit in one ' +
      'fixed space called the context window — ' + nf(ctx.num_ctx) + ' tokens here. The answer it writes ' +
      'takes room in that same space.'));
    const tl = tokenLine(ctx);
    if (tl) container.appendChild(h('p', 'cx-intro', tl));
    container.appendChild(h('p', 'cx-intro',
      n > 1
        ? 'This answer took ' + n + ' model calls. Each one got its OWN empty window — they do not share it, ' +
          'and nothing carries over except what we chose to send again.'
        : 'This answer took one model call, in its own empty window.'));

    ctx.processes.forEach((p, i) => container.appendChild(section(p, i, n, ctx)));

    container.appendChild(h('p', 'cx-note',
      'Why ' + nf(ctx.num_ctx) + ' and not more? It is a project setting (NUM_CTX), not a limit of the models. ' +
      'A bigger window reserves memory for every token whether used or not, and makes each answer slower.'));
  }

  window.NodiContext = { render: render, percent: percent, WHAT: WHAT };
})();
