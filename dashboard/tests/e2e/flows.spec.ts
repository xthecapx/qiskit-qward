// The browser flows listed in dashboard/README.md, against the production build and real data.
import { expect, test, type Page } from "@playwright/test";

async function openApp(page: Page, hash = "") {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  await page.goto(`./${hash}`);
  await expect(page.locator("#source")).toContainText("jobs");
  return errors;
}

async function openFirstJob(page: Page) {
  await page.getByText("Compare the experiments shown").click();
  await page.locator("#settings-table button.job-value").first().click();
  await expect(page.locator("#histogram-title")).toBeFocused();
}

test("1 · algorithm, question and period update the plot and open a histogram", async ({ page }) => {
  const errors = await openApp(page);
  await expect(page.locator("#plot-title")).toHaveText("Success rate by logical qubits");
  await page.getByLabel("Algorithm").selectOption({ label: "Grover" });
  await page.getByLabel("Question").selectOption({ label: "Did expected peaks dominate competitors?" });
  await expect(page.locator("#plot-title")).toHaveText("Differential Success Rate by logical qubits");
  await expect(page.locator("#coverage")).toContainText("jobs");

  await page.getByRole("button", { name: "Adjust plot" }).click();
  const periods = page.getByLabel("Periods (UTC)");
  await expect(periods.locator("option").first()).toHaveText("All periods");
  const before = await page.locator("#settings-coverage").textContent();
  await periods.selectOption({ index: 1 });
  await expect(page.locator("#settings-coverage")).not.toHaveText(before!);
  await page.getByRole("button", { name: "Done" }).click();
  await expect(page.locator("#plot-settings")).not.toBeVisible();
  await expect(page.getByRole("button", { name: "Adjust plot" })).toBeFocused();

  await openFirstJob(page);
  await expect(page.locator("#hist svg")).toBeVisible();
  await page.getByRole("button", { name: "Operating range" }).click();
  await expect(page.locator("#plot-title")).toBeVisible();
  expect(errors).toEqual([]);
});

test("2 · experiment picker searches, selects by keyboard, and Escape keeps the selection", async ({ page }) => {
  await openApp(page);
  await openFirstJob(page);
  const trigger = page.locator("#choose-config");
  const original = await trigger.textContent();

  await trigger.click();
  const search = page.locator("#experiment-search");
  await expect(search).toBeFocused();
  await search.fill("zzzz-no-match");
  await expect(page.locator("#picker-empty")).toBeVisible();
  await page.getByRole("button", { name: "Clear search" }).click();
  await search.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  const target = (await page.locator("#experiment-results button:focus strong").textContent())!;
  await page.keyboard.press("Enter");
  await expect(page.locator("#experiment-picker")).not.toBeVisible();
  await expect(trigger).toHaveText(target);
  expect(target).not.toBe(original);

  await trigger.click();
  await page.keyboard.press("Escape");
  await expect(page.locator("#experiment-picker")).not.toBeVisible();
  await expect(trigger).toHaveText(target);
  await expect(trigger).toBeFocused();
});

test("3 · chart detail groups bitstrings without changing metrics", async ({ page }) => {
  await openApp(page);
  // Open the job with the most distinct outcomes so grouping is guaranteed.
  const busiest = await page.evaluate(async () => {
    const data = await (await fetch("data/index.json")).json();
    return data.jobs.reduce((a: any, b: any) =>
      Object.keys(b.counts).length > Object.keys(a.counts).length ? b : a,
    );
  });
  await page.goto(`./#a=${busiest.algorithm}&v=job&c=${busiest.config}&r=${busiest.ref}:${busiest.rep}`);
  await expect(page.locator("#histogram-title")).toBeVisible();
  const metrics = await page.locator("#numbers").textContent();
  const rows = await page.locator("#outcomes tr").count();
  await expect(page.locator("#bins-help")).toContainText("grouped as “Other outcomes”");
  await page.getByLabel("Chart detail").selectOption({ label: "Expanded · up to 32 bitstrings" });
  await expect(page.locator("#bins-help")).toContainText(/^(3[0-2]|[12]?\d) individual bitstrings/);
  await expect(page.locator("#numbers")).toHaveText(metrics!);
  expect(await page.locator("#outcomes tr").count()).toBe(rows);
});

test("4 · run calibration round-trip preserves the histogram run and settings", async ({ page }) => {
  await openApp(page);
  await openFirstJob(page);
  await page.getByLabel("Chart detail").selectOption({ label: "Standard · up to 16 bitstrings" });
  const run = await page.getByLabel("Recorded run").inputValue();
  const experiment = await page.locator("#choose-config").textContent();

  await page.getByRole("button", { name: /Inspect run calibration/ }).click();
  await expect(page.locator("#calibration-title")).toHaveText("Calibration for this run");
  await expect(page.getByLabel("Calibration snapshot (UTC)")).toBeDisabled();
  await page.locator("#cal-mode").selectOption({ label: "T1 · relaxation time" });
  await expect(page.locator("#cal-value")).toContainText("μs");
  await page.getByRole("button", { name: "Explore this component’s history" }).click();
  await expect(page.locator("#calibration-title")).toHaveText("Chip calibration history");
  await expect(page.getByLabel("Calibration snapshot (UTC)")).toBeEnabled();

  await page.getByRole("button", { name: "← Back to job histogram" }).click();
  await expect(page.locator("#job-calibration")).toBeFocused();
  await expect(page.getByLabel("Recorded run")).toHaveValue(run);
  await expect(page.locator("#choose-config")).toHaveText(experiment!);
  await expect(page.getByLabel("Chart detail")).toHaveValue("1");
});

test("5 · chip calibration history opens independently and searches components @narrow", async ({ page }) => {
  const errors = await openApp(page);
  await page.getByRole("button", { name: /Chip calibration history/ }).click();
  await expect(page.locator("#calibration-title")).toHaveText("Chip calibration history");
  await expect(page.locator("#cal-map svg")).toBeVisible();
  await page.getByLabel("Find a qubit or connection").fill("Q99999");
  await expect(page.getByText("No matching components.")).toBeVisible();
  await page.getByLabel("Find a qubit or connection").fill("");
  await page.locator("#cal-mode").selectOption({ label: "Two-qubit gate errors" });
  await expect(page.getByLabel("Operation")).toBeVisible();
  await expect(page.locator("#cal-component-title")).toContainText("→");
  const width = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  expect(width).toBeLessThanOrEqual(0);
  await page.getByRole("button", { name: "← Back to operating range" }).click();
  await expect(page.locator("#open-calibration")).toBeFocused();
  expect(errors).toEqual([]);
});

test("deep links restore the histogram view", async ({ page }) => {
  await openApp(page);
  await openFirstJob(page);
  const url = page.url();
  await page.goto("about:blank");
  await page.goto(url);
  await expect(page.locator("#histogram-title")).toBeVisible();
  expect(new URL(url).hash).toContain("v=job");
});

test("compare decay shows matched fits, shares selections and restores range settings @narrow", async ({ page }) => {
  const errors = await openApp(page);
  await page.getByLabel("Algorithm", {exact:true}).selectOption({label:"Grover"});
  await page.getByRole("tab", {name:"Compare decay"}).click();
  const panel = page.getByRole("region", {name:"Decay comparison"});
  await expect(panel.locator(".panels svg").first()).toBeVisible();
  await expect(panel.getByRole("heading", {name:"QFT and Phase estimation"})).toBeVisible();
  expect(await panel.locator(".panels article").count()).toBeGreaterThanOrEqual(3);
  await expect(panel.locator(".stat strong")).toHaveText(/\d+\.\d{2} pp/);
  await panel.getByLabel("Resource",{exact:true}).selectOption({label:"Total logical qubits"});
  await panel.getByRole("button",{name:"Grover / Bernstein–Vazirani",exact:true}).first().click();
  await expect(panel.getByRole("heading",{name:"Grover and Bernstein–Vazirani"})).toBeVisible();
  await panel.getByLabel("Outcome",{exact:true}).selectOption({label:"Hellinger fidelity"});
  await page.reload();
  await expect(panel.getByRole("heading",{name:"Grover and Bernstein–Vazirani"})).toBeVisible();
  await expect(panel.locator(".note").first()).toContainText("Hellinger fidelity");
  await panel.getByLabel("Resource",{exact:true}).selectOption({label:"Logical two-qubit gates"});
  await expect(panel.getByText(/No period has the required/)).toBeVisible();
  await page.getByRole("tab",{name:"Operating range"}).click();
  await expect(page.getByLabel("Algorithm",{exact:true}).locator("option:checked")).toHaveText("Grover");
  expect(errors).toEqual([]);
});

test("direction compares column loss to row loss and preserves the question on reload @narrow", async ({page}) => {
  const errors = await openApp(page, "#section=decay&b=ibm_kingston&resource=a.num_qubits&outcome=success&pair=QFT%2FPHASE-ESTIMATION");
  const panel = page.getByRole("region", {name:"Decay comparison"});
  await expect(panel.locator("#matrix-guide")).toContainText("Which algorithm pairs show similar changes");
  await panel.getByLabel("Matrix view").selectOption({label:"Direction"});
  await expect(panel.locator("#matrix-guide")).toContainText("larger fitted decrease in success rate");
  const forward = panel.getByRole("button", {name:/^QFT to PE:/});
  const reverse = panel.getByRole("button", {name:/^PE to QFT:/});
  const shown = Number(await forward.innerText());
  expect(Number.isFinite(shown)).toBeTruthy();
  expect(Number(await reverse.innerText()) + shown).toBeCloseTo(0);
  await reverse.click();
  await expect(panel.locator(".stat span")).toContainText("QFT relative to PE");
  await expect(panel.locator(".stat strong")).toContainText((await reverse.getAttribute("aria-label"))!.split(": ")[1]);
  await page.reload();
  await expect(reverse).toHaveAttribute("aria-pressed", "true");
  await expect(panel.locator(".panels svg").first()).toBeVisible();
  await panel.getByLabel("Matrix view").selectOption({label:"Similarity"});
  await expect(panel.locator(".stat span")).toHaveText("Median gap in fitted change");
  expect(errors).toEqual([]);
});
