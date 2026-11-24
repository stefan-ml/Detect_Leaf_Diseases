// All durations use this browser's monotonic clock. No browser/server clock subtraction.
window.SIR2Timing = {
  enabled: false,
  report(token, value) {
    if (!this.enabled) return;
    fetch('/api/timing/client', {
      method: 'POST',
      headers: {'Content-Type': 'application/json', 'X-SIR2-Token': token},
      body: JSON.stringify(value),
      keepalive: true
    }).catch(() => {});
  }
};
