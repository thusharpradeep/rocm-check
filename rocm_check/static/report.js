(function () {
  "use strict";

  var searchInput = document.getElementById("finding-search");
  var fileFilter = document.getElementById("file-filter");
  var severityButtons = Array.prototype.slice.call(
    document.querySelectorAll("[data-severity-filter]")
  );
  var visibleCount = document.getElementById("visible-count");
  var toggleDetails = document.getElementById("toggle-details");
  var themeToggle = document.getElementById("theme-toggle");
  var findings = Array.prototype.slice.call(
    document.querySelectorAll(".finding")
  );
  var fileGroups = Array.prototype.slice.call(
    document.querySelectorAll(".file-group")
  );
  var emptyMessage = document.getElementById("empty-filter-message");

  if (!emptyMessage) {
    emptyMessage = document.createElement("div");
    emptyMessage.id = "empty-filter-message";
    emptyMessage.className = "empty-state";
    emptyMessage.hidden = true;
    emptyMessage.innerHTML =
      "<strong>No findings match the current filters.</strong>" +
      "<span>Adjust the search or severity filter.</span>";
    var results = document.getElementById("file-results");
    if (results) results.appendChild(emptyMessage);
  }

  var state = {
    query: "",
    severity: "all",
    file: "all",
  };

  function activeTheme() {
    var explicit = document.documentElement.dataset.theme;
    if (explicit === "dark" || explicit === "light") return explicit;
    return window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function updateThemeLabel() {
    if (!themeToggle) return;
    themeToggle.textContent =
      activeTheme() === "dark" ? "Light mode" : "Dark mode";
  }

  function applyFilters() {
    var query = state.query.trim().toLowerCase();
    var visible = 0;

    findings.forEach(function (finding) {
      var matchesSearch = !query || finding.dataset.search.indexOf(query) !== -1;
      var matchesSeverity =
        state.severity === "all" || finding.dataset.severity === state.severity;
      var matchesFile =
        state.file === "all" || finding.dataset.file === state.file;
      var shown = matchesSearch && matchesSeverity && matchesFile;
      finding.hidden = !shown;
      if (shown) visible += 1;
    });

    fileGroups.forEach(function (group) {
      var anyVisible = group.querySelector(".finding:not([hidden])");
      group.hidden = !anyVisible;
    });

    if (visibleCount) {
      visibleCount.textContent =
        findings.length === 0
          ? "0 findings"
          : visible + " of " + findings.length + " findings";
    }
    emptyMessage.hidden = visible !== 0 || findings.length === 0;
  }

  if (searchInput) {
    searchInput.addEventListener("input", function () {
      state.query = searchInput.value;
      applyFilters();
    });
  }

  if (fileFilter) {
    fileFilter.addEventListener("change", function () {
      state.file = fileFilter.value;
      applyFilters();
    });
  }

  severityButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      state.severity = button.dataset.severityFilter;
      severityButtons.forEach(function (other) {
        other.classList.toggle("active", other === button);
      });
      applyFilters();
    });
  });

  if (toggleDetails) {
    toggleDetails.addEventListener("click", function () {
      var expanding = toggleDetails.textContent === "Expand all";
      findings.forEach(function (finding) {
        finding.open = expanding;
      });
      toggleDetails.textContent = expanding ? "Collapse all" : "Expand all";
    });
  }

  if (themeToggle) {
    updateThemeLabel();
    themeToggle.addEventListener("click", function () {
      var next = activeTheme() === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try {
        localStorage.setItem("rocm-check-theme", next);
      } catch (error) {
        // The theme still applies for this view when storage is blocked.
      }
      updateThemeLabel();
    });
  }

  applyFilters();
})();
