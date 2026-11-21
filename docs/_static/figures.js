// Plotly figures in a hidden tab are drawn at zero width; redraw when a tab is shown.
document.addEventListener("change", function (e) {
  if (e.target && e.target.matches && e.target.matches(".sd-tab-set > input")) {
    window.dispatchEvent(new Event("resize"));
  }
});
