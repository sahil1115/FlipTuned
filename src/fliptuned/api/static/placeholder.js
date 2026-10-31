// Reads the per-launch token from the #token= fragment, keeps it in sessionStorage, removes it
// from the address bar, and shows a short machine summary from /api/doctor.
(function () {
  "use strict";
  var key = "fliptuned-token";
  var match = window.location.hash.match(/token=([^&]+)/);
  if (match) {
    window.sessionStorage.setItem(key, decodeURIComponent(match[1]));
    window.history.replaceState(null, "", window.location.pathname);
  }
  var token = window.sessionStorage.getItem(key);
  var status = document.getElementById("status");
  if (!token) {
    status.textContent = "No access token. Open the link printed by `fliptuned serve`.";
    return;
  }
  fetch("/api/doctor", { headers: { "X-FlipTuned-Token": token } })
    .then(function (response) {
      if (!response.ok) { throw new Error("HTTP " + response.status); }
      return response.json();
    })
    .then(function (report) {
      status.textContent = "Connected.";
      var list = document.getElementById("machine");
      var rows = [
        ["System", report.machine && report.machine.os_version],
        ["Backend", report.backend && report.backend.selected
          ? report.backend.selected.backend + " on " + report.backend.selected.device
          : "none available"]
      ];
      rows.forEach(function (row) {
        var dt = document.createElement("dt");
        dt.textContent = row[0];
        var dd = document.createElement("dd");
        dd.textContent = row[1] || "unknown";
        list.appendChild(dt);
        list.appendChild(dd);
      });
    })
    .catch(function (error) {
      status.textContent = "Could not reach the API (" + error.message + ").";
    });
})();
