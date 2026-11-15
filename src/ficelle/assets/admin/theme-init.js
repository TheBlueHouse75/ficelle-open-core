// Runs before first paint so a light-theme user never sees the hardcoded dark default.
// Mirrors applyTheme() in app.js (same storage key, same system fallback); app.js stays
// the owner once it loads.
(function () {
  var pref = "system";
  try { pref = localStorage.getItem("ficelle.theme") || "system"; } catch (e) {}
  var dark = true;
  try { dark = window.matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) {}
  document.documentElement.setAttribute("data-theme", pref === "light" || pref === "dark" ? pref : (dark ? "dark" : "light"));
})();
