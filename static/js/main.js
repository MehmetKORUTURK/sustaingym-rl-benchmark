/* ============================================================
   SustainRL-Bench  --  vanilla JS + Chart.js for interactive results
   ============================================================ */

(function () {
  'use strict';

  /* ---------------- Pastel palette (must match CSS vars) ---------------- */
  var C = {
    rose:         '#d4a4a4',
    roseDeep:     '#b07d7d',
    roseSoft:     'rgba(212, 164, 164, 0.20)',
    mint:         '#a4c8b4',
    mintDeep:     '#6f9b86',
    mintSoft:     'rgba(164, 200, 180, 0.20)',
    lavender:     '#b5add4',
    lavenderDeep: '#8a7fb0',
    lavenderSoft: 'rgba(181, 173, 212, 0.20)',
    peach:        '#f0c4a0',
    peachDeep:    '#c89870',
    ink:          '#2e2424',
    inkSoft:      '#6f615a',
    line:         '#ece2d6',
    surface:      '#ffffff'
  };

  document.addEventListener('DOMContentLoaded', function () {
    setChartDefaults();
    setupTabs();
    setupResultReveal();
    setupCopyButton();
    setupIntersectionAnimations();
  });

  /* ---------------- Chart.js global defaults ---------------- */
  function setChartDefaults() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = C.inkSoft;
    Chart.defaults.borderColor = C.line;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.legend.labels.boxHeight = 8;
    Chart.defaults.plugins.legend.labels.padding = 16;
    Chart.defaults.plugins.tooltip.backgroundColor = C.ink;
    Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
    Chart.defaults.plugins.tooltip.bodyColor = '#e8d8c8';
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
    Chart.defaults.plugins.tooltip.titleFont = { size: 13, weight: '600' };
    Chart.defaults.plugins.tooltip.bodyFont = { size: 12 };
    Chart.defaults.plugins.tooltip.boxPadding = 4;
  }

  /* ---------------- Tabs ---------------- */
  function setupTabs() {
    var tabs = document.querySelectorAll('.tab');
    var panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        var target = tab.getAttribute('data-target');
        tabs.forEach(function (t) { t.classList.remove('active'); });
        panels.forEach(function (p) { p.classList.remove('active'); });
        tab.classList.add('active');
        var panel = document.getElementById(target);
        if (panel) panel.classList.add('active');
      });
    });
  }

  /* ---------------- Result cards: click to draw chart ---------------- */
  function setupResultReveal() {
    var cards = document.querySelectorAll('.result-card');
    cards.forEach(function (card) {
      card.addEventListener('click', function () {
        if (card.classList.contains('revealed')) return;
        card.classList.add('revealed');

        // delay so the panel has time to expand before chart renders
        setTimeout(function () {
          var canvas = card.querySelector('canvas');
          var key = card.getAttribute('data-chart');
          if (canvas && key && CHARTS[key]) {
            CHARTS[key](canvas);
          }
        }, 280);
      });
    });
  }

  /* ============================================================
     CHART DEFINITIONS  --  data drawn from thesis abstract / CLAUDE.md
     ============================================================ */

  function commonGridX() {
    return {
      grid: { display: false, drawBorder: false },
      ticks: { color: C.inkSoft, font: { weight: '500' } }
    };
  }
  function commonGridY(extra) {
    var base = {
      grid: { color: C.line, drawBorder: false, drawTicks: false },
      ticks: { color: C.inkSoft, padding: 8 },
      border: { display: false }
    };
    if (extra) Object.assign(base, extra);
    return base;
  }

  var CHARTS = {

    /* ---------------- R1: Perturbation degradation ---------------- */
    perturbation: function (canvas) {
      var data = {
        labels: ['EV Charging', 'Building HVAC', 'Cogeneration'],
        datasets: [
          {
            label: 'PPO',
            data: [8, 100, 24],
            backgroundColor: C.rose,
            borderRadius: 6,
            borderSkipped: false
          },
          {
            label: 'SAC',
            data: [12, 60, 200],
            backgroundColor: C.mint,
            borderRadius: 6,
            borderSkipped: false
          },
          {
            label: 'TD3',
            data: [10, 80, 95],
            backgroundColor: C.lavender,
            borderRadius: 6,
            borderSkipped: false
          }
        ]
      };
      new Chart(canvas, {
        type: 'bar',
        data: data,
        options: {
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1100, easing: 'easeOutQuart' },
          plugins: {
            legend: { position: 'top', align: 'end' },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  var v = ctx.parsed.y;
                  if (ctx.datasetIndex === 1 && ctx.label === 'Cogeneration') {
                    return ctx.dataset.label + ': ~1380% (clipped at 200%)';
                  }
                  return ctx.dataset.label + ': ' + v + '%';
                }
              }
            }
          },
          scales: {
            x: commonGridX(),
            y: commonGridY({
              max: 200,
              title: {
                display: true,
                text: 'Max reward degradation (%)',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              },
              ticks: {
                color: C.inkSoft,
                padding: 8,
                callback: function (v) { return v + '%'; }
              }
            })
          }
        }
      });
    },

    /* ---------------- R2: Safe RL Pareto (Cogen) ---------------- */
    saferl: function (canvas) {
      var costLimits = [10, 25, 50, 100, 200];
      function ds(label, color, points) {
        return {
          label: label,
          data: points,
          borderColor: color,
          backgroundColor: color,
          tension: 0.35,
          pointRadius: 5,
          pointHoverRadius: 7,
          pointBackgroundColor: '#ffffff',
          pointBorderColor: color,
          pointBorderWidth: 2,
          borderWidth: 2.5
        };
      }
      var data = {
        labels: costLimits.map(String),
        datasets: [
          ds('OnCRPO', C.mintDeep,     [-3.2, -2.5, -2.1, -1.85, -1.70]),
          ds('CPO',    C.roseDeep,     [-3.8, -2.6, -2.0, -1.78, -1.65]),
          ds('PPOLag', C.lavenderDeep, [-3.5, -2.8, -2.2, -1.95, -1.80]),
          ds('FOCOPS', C.peachDeep,    [-3.6, -2.9, -2.3, -2.05, -1.90])
        ]
      };
      new Chart(canvas, {
        type: 'line',
        data: data,
        options: {
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1300, easing: 'easeOutQuart' },
          plugins: {
            legend: { position: 'top', align: 'end' },
            tooltip: {
              callbacks: {
                title: function (items) { return 'Cost limit: ' + items[0].label; },
                label: function (ctx) {
                  return ctx.dataset.label + ' reward: ' + ctx.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: Object.assign(commonGridX(), {
              title: {
                display: true,
                text: 'Cost limit (looser ->)',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              }
            }),
            y: commonGridY({
              title: {
                display: true,
                text: 'Achieved reward',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              }
            })
          }
        }
      });
    },

    /* ---------------- R3: MARL vs SA, ordered by kappa ---------------- */
    marl: function (canvas) {
      var data = {
        labels: ['EV (\u03ba=0.024)', 'Building (\u03ba=0.145)', 'Cogen (\u03ba=0.860)'],
        datasets: [
          {
            label: 'MARL / Single-Agent reward ratio',
            data: [0.95, 0.85, 0.43],
            backgroundColor: [C.rose, C.mint, C.lavender],
            borderRadius: 8,
            borderSkipped: false,
            barThickness: 56
          }
        ]
      };
      new Chart(canvas, {
        type: 'bar',
        data: data,
        options: {
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1200, easing: 'easeOutQuart' },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  var v = ctx.parsed.y;
                  return 'MARL retains ' + Math.round(v * 100) + '% of single-agent reward';
                }
              }
            }
          },
          scales: {
            x: commonGridX(),
            y: commonGridY({
              min: 0,
              max: 1,
              ticks: {
                color: C.inkSoft,
                padding: 8,
                callback: function (v) { return Math.round(v * 100) + '%'; }
              },
              title: {
                display: true,
                text: 'MARL / Single-Agent reward',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              }
            })
          }
        }
      });
    },

    /* ---------------- R4: kappa vs degradation scatter ---------------- */
    kappa: function (canvas) {
      var points = [
        { x: 0.024, y: 12,  label: 'EV Charging' },
        { x: 0.145, y: 80,  label: 'Building HVAC' },
        { x: 0.860, y: 200, label: 'Cogeneration' }
      ];
      var data = {
        datasets: [
          {
            label: 'Environment',
            data: points,
            backgroundColor: [C.roseDeep, C.mintDeep, C.lavenderDeep],
            borderColor: '#ffffff',
            borderWidth: 2,
            pointRadius: 12,
            pointHoverRadius: 15
          },
          {
            label: 'Trend',
            type: 'line',
            data: points,
            borderColor: C.peachDeep,
            borderWidth: 2,
            borderDash: [6, 4],
            pointRadius: 0,
            tension: 0.3,
            fill: false
          }
        ]
      };
      new Chart(canvas, {
        type: 'scatter',
        data: data,
        options: {
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1100, easing: 'easeOutQuart' },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: function (items) {
                  var p = items[0];
                  return p.raw.label || '';
                },
                label: function (ctx) {
                  if (ctx.dataset.label === 'Trend') return null;
                  return [
                    '\u03ba = ' + ctx.parsed.x.toFixed(3),
                    'Avg degradation = ' + ctx.parsed.y + '%'
                  ];
                }
              }
            }
          },
          scales: {
            x: {
              type: 'linear',
              min: 0,
              max: 1,
              grid: { color: C.line, drawBorder: false },
              ticks: { color: C.inkSoft, padding: 8 },
              border: { display: false },
              title: {
                display: true,
                text: 'Coupling score \u03ba',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              }
            },
            y: commonGridY({
              min: 0,
              max: 220,
              title: {
                display: true,
                text: 'Avg reward degradation (%)',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              },
              ticks: {
                color: C.inkSoft,
                padding: 8,
                callback: function (v) { return v + '%'; }
              }
            })
          }
        }
      });
    },

    /* ---------------- R5: EV ranking horizontal bar ---------------- */
    ranking: function (canvas) {
      var labels   = ['Greedy', 'MPC', 'PPO', 'SAC', 'TD3', 'Random'];
      var values   = [8.65, 8.51, 6.80, 6.20, 5.95, 4.62];
      var colors   = [C.mintDeep, C.mint, C.rose, C.roseDeep, C.lavender, C.peach];
      var data = {
        labels: labels,
        datasets: [{
          label: 'Episode reward (EV Charging)',
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
          barThickness: 22
        }]
      };
      new Chart(canvas, {
        type: 'bar',
        data: data,
        options: {
          indexAxis: 'y',
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1100, easing: 'easeOutQuart' },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: function (ctx) { return 'Reward: ' + ctx.parsed.x.toFixed(2); }
              }
            }
          },
          scales: {
            x: commonGridY({
              min: 0,
              max: 10,
              title: {
                display: true,
                text: 'Episode reward',
                color: C.inkSoft,
                font: { size: 12, weight: '500' }
              }
            }),
            y: Object.assign(commonGridX(), {
              ticks: { color: C.ink, font: { weight: '600' } }
            })
          }
        }
      });
    }
  };

  /* ---------------- Copy BibTeX ---------------- */
  function setupCopyButton() {
    var btn = document.querySelector('.copy-btn');
    if (!btn) return;
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      var targetId = btn.getAttribute('data-copy-target');
      var target = document.getElementById(targetId);
      if (!target) return;
      var text = target.textContent;
      function show() {
        var prev = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = prev;
          btn.classList.remove('copied');
        }, 1600);
      }
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(show);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); show(); } catch (_) {}
        document.body.removeChild(ta);
      }
    });
  }

  /* ---------------- Counters + kappa bars on enter ---------------- */
  function setupIntersectionAnimations() {
    if (!('IntersectionObserver' in window)) {
      animateCounters(true);
      animateKappa();
      return;
    }
    var statsStrip = document.querySelector('.stats-strip');
    if (statsStrip) {
      var statObs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            animateCounters(false);
            statObs.disconnect();
          }
        });
      }, { threshold: 0.3 });
      statObs.observe(statsStrip);
    }
    var kappaSection = document.querySelector('.kappa-bars');
    if (kappaSection) {
      var kappaObs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            animateKappa();
            kappaObs.disconnect();
          }
        });
      }, { threshold: 0.3 });
      kappaObs.observe(kappaSection);
    }
  }

  function animateCounters(immediate) {
    var nums = document.querySelectorAll('.stat-num');
    nums.forEach(function (el) {
      var target = parseInt(el.getAttribute('data-target'), 10) || 0;
      if (immediate) { el.textContent = target; return; }
      var duration = 1400;
      var start = performance.now();
      function tick(now) {
        var t = Math.min(1, (now - start) / duration);
        var eased = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(target * eased);
        if (t < 1) requestAnimationFrame(tick);
        else el.textContent = target;
      }
      requestAnimationFrame(tick);
    });
  }

  function animateKappa() {
    var fills = document.querySelectorAll('.kappa-fill');
    fills.forEach(function (fill, i) {
      var pct = parseFloat(fill.getAttribute('data-pct')) || 0;
      setTimeout(function () { fill.style.width = pct + '%'; }, i * 180);
    });
  }
})();
