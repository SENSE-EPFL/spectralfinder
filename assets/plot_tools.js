// The zoom / reset / PNG buttons above the spectrum (app.py window_detail, data-plot-tool). Plain
// DOM: the plot is redrawn with every window shown, so the buttons are matched on each click.
// Zooming acts on the wavenumber axis, around the middle of what is shown, and never goes past
// the window's edges (xaxis.minallowed / maxallowed): the spectrum is computed only there.
document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-plot-tool]");
    const plot = document.querySelector("#spectrum .js-plotly-plot");
    if (!button || !plot || !window.Plotly) {
        return;
    }
    const axis = plot.layout.xaxis;
    const lo = axis.minallowed, hi = axis.maxallowed;
    const [a, b] = plot._fullLayout.xaxis.range.map(Number);
    const middle = (a + b) / 2;
    const tool = button.getAttribute("data-plot-tool");
    if (tool === "in" || tool === "out") {
        const half = (b - a) / 2 * (tool === "in" ? 0.5 : 2);
        if (2 * half >= hi - lo) {
            Plotly.relayout(plot, {"xaxis.range": [lo, hi]});
        } else {  // keep the same width inside the window if it would run past an edge
            const start = Math.min(Math.max(middle - half, lo), hi - 2 * half);
            Plotly.relayout(plot, {"xaxis.range": [start, start + 2 * half]});
        }
    } else if (tool === "reset") {
        Plotly.relayout(plot, {"xaxis.range": [lo, hi], "yaxis.autorange": true});
    } else if (tool === "png") {
        Plotly.downloadImage(plot, {format: "png", filename: button.getAttribute("data-filename") || "spectrum",
                                    width: 1400, height: 600});
    }
});
