// A- / A+ in the header (data-text-size): every size in style.css is in rem, so changing the root
// font size scales text, fields, tables and spacing together. Steps of 10 % from 80 % to 150 %,
// remembered in this browser only (a viewer convenience: it falls back to 100 % if storage is
// unavailable, e.g. in a private window).
(() => {
    const KEY = "spectralfinder-text-step";
    const MIN = -2, MAX = 5;
    const apply = (step) => { document.documentElement.style.fontSize = `${100 + 10 * step}%`; };
    let step = 0;
    try {
        step = Math.min(MAX, Math.max(MIN, parseInt(localStorage.getItem(KEY), 10) || 0));
    } catch (e) { /* storage blocked: default size */ }
    apply(step);

    document.addEventListener("click", (event) => {
        const button = event.target.closest("[data-text-size]");
        if (!button) {
            return;
        }
        step = Math.min(MAX, Math.max(MIN, step + parseInt(button.getAttribute("data-text-size"), 10)));
        apply(step);
        try {
            localStorage.setItem(KEY, String(step));
        } catch (e) { /* not remembered, still applied */ }
        window.dispatchEvent(new Event("resize"));  // the plot re-fits its container
    });
})();
