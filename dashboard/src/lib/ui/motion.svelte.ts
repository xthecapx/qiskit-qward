/** Reactive prefers-reduced-motion flag shared by charts and transitions. */
class Motion {
  reduced = $state(false);

  constructor() {
    if (typeof matchMedia === "undefined") return;
    const query = matchMedia("(prefers-reduced-motion: reduce)");
    this.reduced = query.matches;
    query.addEventListener("change", (event) => (this.reduced = event.matches));
  }
}

export const motion = new Motion();
