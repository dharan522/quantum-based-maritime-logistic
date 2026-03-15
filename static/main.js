let pollInterval = null;
let logCount = 0;

function startPipeline() {
  const scenario = document.querySelector('input[name="scenario"]:checked').value;
  const layers   = document.getElementById('layers').value;
  const vessels  = document.getElementById('vessels').value;

  document.getElementById('run-btn').disabled = true;
  document.getElementById('run-btn').textContent = '⏳  RUNNING...';
  document.getElementById('terminal').innerHTML = '';
  document.getElementById('kpi-row').style.display = 'none';
  document.getElementById('results').style.display = 'none';
  logCount = 0;

  fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario, layers: parseInt(layers), vessels: parseInt(vessels) })
  });

  pollInterval = setInterval(pollStatus, 600);
}

function pollStatus() {
  fetch('/status').then(r => r.json()).then(data => {
    updateTerminal(data.logs);
    updateProgress(data.progress);
    if (data.done) {
      clearInterval(pollInterval);
      document.getElementById('run-btn').disabled = false;
      document.getElementById('run-btn').textContent = '▶  RUN QAOA';
      if (data.error) {
        document.getElementById('progress-text').textContent = 'Error — see log';
      } else if (data.report) {
        showResults(data.report);
      }
    }
  });
}

function updateTerminal(logs) {
  const term = document.getElementById('terminal');
  logs.slice(logCount).forEach(entry => {
    const div = document.createElement('div');
    div.className = 'log-' + entry.level;
    div.innerHTML = `<span class="log-ts">[${entry.ts}]</span>${esc(entry.msg)}`;
    term.appendChild(div);
  });
  logCount = logs.length;
  term.scrollTop = term.scrollHeight;
}

function updateProgress(pct) {
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
  document.getElementById('progress-text').textContent =
    pct < 100 ? 'Running... ' + pct + '%' : '✅ Complete';
}

function showResults(report) {
  const kpiRow = document.getElementById('kpi-row');
  kpiRow.style.display = 'flex';
  const q = report.qaoa;
  kpiRow.innerHTML =
    kpi('QUBO Energy',    q.qubo_energy.toFixed(4),                          'var(--cyan)')   +
    kpi('Route Cost',     '$' + fmt(q.total_cost),                           'var(--amber)')  +
    kpi('Vessels',        q.n_assigned + '/' + report.n_vessels,             'var(--green)')  +
    kpi('Constraints',    q.constraints_satisfied ? '✅ OK' : '❌ Fail',
                          q.constraints_satisfied ? 'var(--green)' : 'var(--coral)')          +
    kpi('Layers',         report.p_layers,                                   'var(--purple)') +
    kpi('Scenario',       report.scenario.toUpperCase(),                     'var(--teal)')   +
    kpi('Fuel',           '$' + report.fuel_last + '/t',                     'var(--amber)')  +
    kpi('Events',         report.n_events,
                          report.n_events > 3 ? 'var(--coral)' : 'var(--green)');

  const ts = Date.now();
  document.getElementById('img-conv').src  = '/outputs/fig_convergence_landscape.png?'  + ts;
  document.getElementById('img-dist').src  = '/outputs/fig_distribution_benchmark.png?' + ts;
  document.getElementById('img-twin').src  = '/outputs/fig_digital_twin.png?'           + ts;
  document.getElementById('img-map').src   = '/outputs/fig_port_network.png?'           + ts;
  document.getElementById('img-ops').src   = '/outputs/fig_twin_operations.png?'        + ts;
  document.getElementById('img-qubo').src  = '/outputs/fig_qubo_topk.png?'              + ts;

  buildRoutesTable(report.routes, report.qaoa.bitstring, report.qubo_meta.var_labels);

  if (report.events && report.events.length > 0) {
    document.getElementById('events-card').style.display = 'block';
    buildEventsTable(report.events);
  }

  buildTopKTable(report.top_k);
  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function buildRoutesTable(routes, bestBs, varLabels) {
  const tbl  = document.getElementById('routes-table');
  const cols = ['vessel', 'route', 'distance_nm', 'days', 'cost_usd', 'var'];
  let html   = '<tr>' + cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('') + '<th>selected</th></tr>';
  routes.forEach(r => {
    const qi  = varLabels ? varLabels.indexOf(r.var || '') : -1;
    const sel = qi >= 0 && qi < bestBs.length && bestBs[qi] === '1';
    html += `<tr class="${sel ? 'selected-row' : ''}">`;
    cols.forEach(c => {
      let v = r[c] != null ? r[c] : '';
      if (c === 'cost_usd' || c === 'distance_nm') v = Number(v).toLocaleString();
      if (c === 'days') v = Number(v).toFixed(2);
      html += `<td>${v}</td>`;
    });
    html += `<td>${sel ? '<span class="tag tag-ok">✅ SELECTED</span>' : ''}</td></tr>`;
  });
  tbl.innerHTML = html;
}

function buildEventsTable(events) {
  const tbl  = document.getElementById('events-table');
  const cols = ['timestamp', 'port', 'event_type', 'severity', 'duration_hr', 'delay_hours', 'throughput%', 'cost_mult'];
  let html   = '<tr>' + cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('') + '</tr>';
  events.forEach(ev => {
    const sev = parseFloat(ev.severity || 0);
    const cls = sev > 0.7 ? 'tag-err' : sev > 0.4 ? 'tag-warn' : 'tag-ok';
    html += '<tr>';
    cols.forEach(c => {
      let v = ev[c] != null ? ev[c] : '';
      if (c === 'severity') v = `<span class="tag ${cls}">${parseFloat(v).toFixed(2)}</span>`;
      html += `<td>${v}</td>`;
    });
    html += '</tr>';
  });
  tbl.innerHTML = html;
}

function buildTopKTable(topk) {
  if (!topk || !topk.length) return;
  const tbl = document.getElementById('topk-table');
  let html  = '<tr><th>#</th><th>Bitstring</th><th>Probability</th><th>Energy</th><th>Count</th></tr>';
  topk.forEach((s, i) => {
    html += `<tr ${i === 0 ? 'class="selected-row"' : ''}>
      <td>${i + 1}</td>
      <td style="font-size:10px">${s.bitstring}</td>
      <td>${(s.probability * 100).toFixed(2)}%</td>
      <td>${s.energy.toFixed(4)}</td>
      <td>${s.count}</td>
    </tr>`;
  });
  tbl.innerHTML = html;
}

function kpi(label, value, color) {
  return `<div class="kpi" style="border-left-color:${color}">
    <div class="kpi-label">${label}</div>
    <div class="kpi-value" style="color:${color}">${value}</div>
  </div>`;
}

function fmt(n)  { return Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
function esc(s)  { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
```

---

### Final folder structure
```
