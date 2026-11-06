/* Exercise the production filters and job selection without a browser dependency. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

function explorer() {
  const selects = new Map();
  const element = id => {
    if (!selects.has(id)) selects.set(id, {
      items: [], textContent: '', current: '',
      get value() { return this.current; },
      get selectedOptions() { return this.items.filter(i => i.value === this.current); },
      set value(value) { this.current = this.items.some(i => i.value === value) ? value : ''; },
      replaceChildren(...items) { this.items = items; this.current = items[0]?.value ?? ''; },
      focus() {}, scrollIntoView() {},
    });
    return selects.get(id);
  };
  const fixture = [
    { backend: 'ibm_fez', algorithm: 'SHOR', config: 'M3', family: 'a=7', width: 7,
      campaign: 'full', window: 1, ref: 'old', rep: 1, finishedAt: '2026-11-04T13:00:00Z' },
    { backend: 'ibm_fez', algorithm: 'SHOR', config: 'M3', family: 'a=7', width: 7,
      campaign: 'growth', window: 2, ref: 'new', rep: 1, finishedAt: '2026-11-06T00:00:00Z' },
    { backend: 'ibm_fez', algorithm: 'SHOR', config: 'M4', family: 'a=7', width: 8,
      campaign: 'growth', window: 2, ref: 'wider', rep: 1, finishedAt: '2026-11-06T00:01:00Z' },
    { backend: 'ibm_fez', algorithm: 'QFT', config: 'SP3', family: 'SP', width: 3,
      campaign: 'full', window: 1, ref: 'qft', rep: 1 },
    { backend: 'ibm_kingston', algorithm: 'SHOR', config: 'M3', family: 'a=7', width: 7,
      campaign: 'full', window: 1, ref: 'other-chip', rep: 1 },
  ];
  const context = vm.createContext({
    document: { getElementById: element },
    snapshot: { jobs: fixture },
    matchMedia: () => ({ matches: false }),
    Option: function(text, value) { this.text = text; this.value = value; },
    window: { scrollY: 0 },
  });
  const source = fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8');
  vm.runInContext(source.slice(0, source.lastIndexOf('try {\n  init();')), context);
  vm.runInContext(`
    renderSweep = () => {};
    renderRun = () => {};
    globalThis.api = {
      scope, sweep, openJob, repetitions, jobKey, selectedJob,
      get jobs() { return jobs; },
      get configs() { return configs; },
    };
    options('backend', ['ibm_fez', 'ibm_kingston'], 'ibm_fez');
    options('algorithm', ['SHOR', 'QFT'], 'SHOR');
  `, context);
  return { api: context.api, element, fixture };
}

test('all periods is the default and individual collections remain selectable', () => {
  const { api, element } = explorer();
  api.scope();
  assert.equal(element('window').value, '');
  assert.equal(element('window').items[0].text, 'All periods');
  assert.deepEqual(Array.from(api.jobs, j => j.ref), ['old', 'new', 'wider']);
  const period = element('window').items.find(i => i.value.includes('growth')).value;
  element('window').value = period;
  api.scope();
  assert.deepEqual(Array.from(api.jobs, j => j.ref), ['new', 'wider']);
  element('window').value = '';
  api.scope();
  assert.equal(api.jobs.length, 3);
  element('algorithm').value = 'QFT';
  api.sweep();
  assert.deepEqual(Array.from(api.jobs, j => j.ref), ['qft']);
});

test('same-numbered runs from different periods open the correct histogram job', () => {
  const { api, element, fixture } = explorer();
  api.scope();
  api.openJob(fixture[1], element('plot-title'));
  assert.equal(element('rep').items.length, 2);
  assert.equal(new Set(element('rep').items.map(i => i.value)).size, 2);
  assert.equal(api.selectedJob().ref, 'new');
  assert.match(element('rep').items[1].text, /6 Nov 2026/);
  element('rep').value = api.jobKey(fixture[0]);
  assert.equal(api.selectedJob().ref, 'old');
  api.openJob({ ...fixture[1], value: [7, 0.25] }, element('plot-title'));
  assert.equal(api.selectedJob().ref, 'new');
});

test('selecting another chip excludes the first chip across all periods', () => {
  const { api, element } = explorer();
  api.scope();
  element('backend').value = 'ibm_kingston';
  api.scope();
  assert.deepEqual(Array.from(api.jobs, j => j.ref), ['other-chip']);
});
