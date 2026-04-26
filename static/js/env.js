/* ============================================================
   SustainRL-Bench  --  per-environment deep-dive page
   Reads ?env=<id> from URL, fetches static/data/site_data.json,
   renders hero, hyperparameter cards, and 5 interactive charts.
   ============================================================ */

(function () {
  'use strict';

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
    inkLight:     '#9b8d86',
    line:         '#ece2d6',
    surface:      '#ffffff'
  };

  // Algorithm grouping for display
  var ALGO_GROUPS = [
    {
      title: 'Standard RL',
      framework: 'Stable-Baselines3',
      color: 'rose',
      keys: ['PPO', 'SAC', 'TD3']
    },
    {
      title: 'Safe RL (CMDP)',
      framework: 'OmniSafe',
      color: 'mint',
      keys: ['PPOLag', 'CPO', 'OnCRPO', 'FOCOPS']
    },
    {
      title: 'Multi-Agent RL',
      framework: 'Ray RLlib + PettingZoo',
      color: 'lavender',
      keys: ['MARL_PPO', 'MARL_SAC', 'MARL_APPO', 'MARL_IMPALA']
    }
  ];

  // Color cycle for line charts
  var ALGO_COLORS = {
    PPO:      C.roseDeep,
    SAC:      C.mintDeep,
    TD3:      C.lavenderDeep,
    APPO:     C.peachDeep,
    IMPALA:   '#9b6b9b',
    PPOLag:   C.roseDeep,
    CPO:      C.mintDeep,
    OnCRPO:   C.lavenderDeep,
    FOCOPS:   C.peachDeep,
    SACLag:   '#9b6b9b'
  };

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    setChartDefaults();

    var envId = (new URLSearchParams(window.location.search)).get('env') || 'evcharging';
    highlightTopnavTab(envId);

    fetch('static/data/site_data.json')
      .then(function (r) { return r.json(); })
      .then(function (allData) {
        var data = allData[envId];
        if (!data) {
          document.body.innerHTML = '<div style="padding:80px;text-align:center"><h1>Unknown environment: ' + envId + '</h1><p><a href="index.html">Back to overview</a></p></div>';
          return;
        }
        renderHero(envId, data);
        renderAlgorithms(data);
        renderPerturbationChart(envId, data);
        renderMarlChart(envId, data);
        renderSaferlCharts(envId, data);
        renderBaselineChart(envId, data);
      })
      .catch(function (err) {
        console.error(err);
        document.body.innerHTML = '<div style="padding:80px;text-align:center"><h1>Could not load data</h1><pre>' + err + '</pre></div>';
      });
  }

  function setChartDefaults() {
    if (typeof Chart === 'undefined') return;
    Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = C.inkSoft;
    Chart.defaults.borderColor = C.line;
    Chart.defaults.plugins.legend.labels.usePointStyle = true;
    Chart.defaults.plugins.legend.labels.boxWidth = 8;
    Chart.defaults.plugins.legend.labels.boxHeight = 8;
    Chart.defaults.plugins.legend.labels.padding = 14;
    Chart.defaults.plugins.tooltip.backgroundColor = C.ink;
    Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
    Chart.defaults.plugins.tooltip.bodyColor = '#e8d8c8';
    Chart.defaults.plugins.tooltip.padding = 12;
    Chart.defaults.plugins.tooltip.cornerRadius = 8;
  }

  function highlightTopnavTab(envId) {
    document.querySelectorAll('.topnav-tab').forEach(function (a) {
      if (a.getAttribute('data-env-link') === envId) a.classList.add('active');
    });
  }

  /* ============================================================
     HERO
     ============================================================ */
  function renderHero(envId, data) {
    var meta = data.meta;
    document.title = 'SustainRL-Bench  --  ' + meta.label;
    document.body.setAttribute('data-env', envId);

    document.getElementById('envLabel').textContent = meta.label;
    document.getElementById('envLabel').classList.add(meta.color);
    document.getElementById('envTitle').textContent = meta.label;
    document.getElementById('envSub').innerHTML = meta.data_source +
      ' &middot; ' + meta.steps_per_episode + ' steps/episode (&Delta;t&nbsp;=&nbsp;' +
      meta.delta_t_minutes + '&nbsp;min)';

    var stats = [
      ['Coupling \u03ba', meta.kappa],
      ['MARL agents', meta.marl_agents],
      ['Action space', meta.action_space],
      ['CMDP cost', meta.cmdp_cost]
    ];
    var html = stats.map(function (s) {
      return '<div class="env-hero-stat"><span class="khv">' + s[0] + '</span><span class="khn">' + s[1] + '</span></div>';
    }).join('');
    document.getElementById('envHeroStats').innerHTML = html;
  }

  /* ============================================================
     ALGORITHMS
     ============================================================ */
  function renderAlgorithms(data) {
    var hp = data.hyperparams || {};
    var html = '';

    ALGO_GROUPS.forEach(function (group) {
      var present = group.keys.filter(function (k) { return hp[k]; });
      if (!present.length) return;

      html += '<div class="algo-group-block">';
      html += '<div class="algo-group-block-head ' + group.color + '">';
      html += '<h3>' + group.title + '</h3>';
      html += '<span>' + group.framework + '</span>';
      html += '</div>';
      html += '<div class="algo-cards">';
      present.forEach(function (k) {
        var cfg = hp[k];
        var displayName = k.replace(/^MARL_/, '');
        html += '<div class="algo-card">';
        html += '<div class="algo-card-name">' + displayName + '</div>';
        html += '<div class="algo-card-grid">';
        Object.keys(cfg).forEach(function (param) {
          if (param === 'note' || param === 'framework') return;
          html += '<div class="algo-param">' +
                  '<span class="algo-param-key">' + formatKey(param) + '</span>' +
                  '<span class="algo-param-val">' + formatVal(cfg[param]) + '</span>' +
                  '</div>';
        });
        html += '</div>';
        if (cfg.note) {
          html += '<div class="algo-card-note">' + cfg.note + '</div>';
        }
        html += '</div>';
      });
      html += '</div></div>';
    });

    document.getElementById('algoSection').innerHTML = html;
  }

  function formatKey(k) {
    var map = {
      'lr': 'lr',
      'n_steps': 'n_steps',
      'batch_size': 'batch',
      'n_epochs': 'epochs',
      'gamma': 'gamma',
      'tau': 'tau',
      'buffer_size': 'buffer',
      'total_steps': 'total steps',
      'policy': 'policy',
      'cost_limits': 'cost limits',
      'cost_scale': 'cost scale',
      'steps_per_epoch': 'steps/epoch',
      'network': 'net',
      'workers': 'workers',
      'iterations': 'iters',
      'train_batch': 'train batch',
      'rollout_fragment': 'rollout',
      'grad_clip': 'grad clip',
      'reward_beta': 'reward beta',
      'config': 'config'
    };
    return map[k] || k;
  }
  function formatVal(v) {
    if (Array.isArray(v)) return '[' + v.join(', ') + ']';
    if (typeof v === 'number') {
      if (v >= 1000000) return (v / 1000000) + 'M';
      if (v >= 1000) return (v / 1000) + 'K';
      if (v < 0.01 && v > 0) return v.toExponential(0).replace('e-0', 'e-');
      return v.toString();
    }
    return v;
  }

  /* ============================================================
     CHART HELPERS
     ============================================================ */
  function gridX(extra) {
    var b = {
      grid: { display: false, drawBorder: false },
      ticks: { color: C.inkSoft, font: { weight: '500' } },
      border: { display: false }
    };
    if (extra) Object.assign(b, extra);
    return b;
  }
  function gridY(extra) {
    var b = {
      grid: { color: C.line, drawBorder: false, drawTicks: false },
      ticks: { color: C.inkSoft, padding: 8 },
      border: { display: false }
    };
    if (extra) Object.assign(b, extra);
    return b;
  }

  /* ============================================================
     CHART 1: PERTURBATION
     ============================================================ */
  function renderPerturbationChart(envId, data) {
    var p = data.perturbation || {};
    var algos = Object.keys(p);
    if (!algos.length) {
      document.getElementById('perturbCard').style.display = 'none';
      return;
    }

    // Collect all level labels (PS=0.05, PA=0.15, etc.) and order them
    var levelSet = {};
    algos.forEach(function (a) {
      Object.keys(p[a]).forEach(function (l) { levelSet[l] = true; });
    });
    var levels = Object.keys(levelSet);
    // sort so baseline comes first, then PS, PA, PD groups by intensity
    levels.sort(function (a, b) {
      if (a === 'baseline') return -1;
      if (b === 'baseline') return  1;
      var ka = a.split('=')[0], kb = b.split('=')[0];
      if (ka !== kb) {
        var order = { PS: 1, PA: 2, PD: 3 };
        return (order[ka] || 9) - (order[kb] || 9);
      }
      return parseFloat(a.split('=')[1]) - parseFloat(b.split('=')[1]);
    });

    var palette = [C.roseDeep, C.mintDeep, C.lavenderDeep, C.peachDeep];
    var datasets = algos.map(function (algo, i) {
      var col = palette[i % palette.length];
      return {
        label: algo,
        data: levels.map(function (l) {
          var v = p[algo][l];
          return v ? v.mean : null;
        }),
        backgroundColor: col,
        borderRadius: 5,
        borderSkipped: false,
        // store std for tooltip access
        _std: levels.map(function (l) {
          var v = p[algo][l];
          return v ? v.std : null;
        }),
        _n: levels.map(function (l) {
          var v = p[algo][l];
          return v ? v.n : null;
        })
      };
    });

    new Chart(document.getElementById('perturbChart'), {
      type: 'bar',
      data: { labels: levels, datasets: datasets },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        animation: { duration: 1100, easing: 'easeOutQuart' },
        plugins: {
          legend: { position: 'top', align: 'end' },
          tooltip: {
            callbacks: {
              title: function (items) { return 'Perturbation: ' + items[0].label; },
              label: function (ctx) {
                var ds = ctx.dataset;
                var i = ctx.dataIndex;
                var mean = ctx.parsed.y;
                var std = ds._std ? ds._std[i] : null;
                var n   = ds._n   ? ds._n[i]   : null;
                if (mean === null) return ds.label + ': n/a';
                return ds.label + ': ' + mean.toFixed(3) +
                  (std !== null ? ' \u00b1 ' + std.toFixed(3) : '') +
                  (n   !== null ? '  (n=' + n + ')' : '');
              }
            }
          }
        },
        scales: {
          x: gridX({
            title: { display: true, text: 'Perturbation channel & intensity',
                     color: C.inkSoft, font: { size: 12, weight: '500' } }
          }),
          y: gridY({
            title: { display: true, text: 'Mean reward (multi-seed)',
                     color: C.inkSoft, font: { size: 12, weight: '500' } }
          })
        }
      }
    });

    var n = (datasets[0]._n || []).find(function (x) { return x; }) || 5;
    document.getElementById('perturbCaption').innerHTML =
      'Multi-seed test (n=' + n + ' per cell). Channel codes: <strong>PS</strong> = state, <strong>PA</strong> = action, <strong>PD</strong> = dynamics.';
  }

  /* ============================================================
     CHART 2: MARL CURVES
     ============================================================ */
  function renderMarlChart(envId, data) {
    var curves = data.marl_curves || {};
    var algos = Object.keys(curves);
    if (!algos.length) return;

    var palette = [C.roseDeep, C.mintDeep, C.lavenderDeep, C.peachDeep, '#9b6b9b'];
    var datasets = algos.map(function (a, i) {
      var col = ALGO_COLORS[a] || palette[i % palette.length];
      return {
        label: a,
        data: curves[a].map(function (p) { return { x: p.x, y: p.y }; }),
        borderColor: col,
        backgroundColor: col,
        tension: 0.25,
        pointRadius: 0,
        pointHoverRadius: 5,
        borderWidth: 2.2
      };
    });

    new Chart(document.getElementById('marlChart'), {
      type: 'line',
      data: { datasets: datasets },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        animation: { duration: 1300, easing: 'easeOutQuart' },
        interaction: { intersect: false, mode: 'nearest', axis: 'x' },
        plugins: {
          legend: { position: 'top', align: 'end' },
          tooltip: {
            callbacks: {
              title: function (items) { return 'Iteration ' + Math.round(items[0].parsed.x); },
              label: function (ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3); }
            }
          }
        },
        scales: {
          x: Object.assign(gridY(), {
            type: 'linear',
            title: { display: true, text: 'Training iteration', color: C.inkSoft, font: { weight: '500' } }
          }),
          y: gridY({
            title: { display: true, text: 'Mean episode reward',
                     color: C.inkSoft, font: { size: 12, weight: '500' } }
          })
        }
      }
    });

    document.getElementById('marlCaption').textContent =
      'Each curve is one independent-policy MARL run (no policy sharing). Curves downsampled to ~80 points for browser performance.';
  }

  /* ============================================================
     CHARTS 3 & 4: SAFE RL REWARD + COST
     ============================================================ */
  function renderSaferlCharts(envId, data) {
    var curves = data.saferl_curves || {};
    var keys = Object.keys(curves);
    if (!keys.length) return;

    // Group by algo so we can colour by algo, and dash by cost limit
    var byAlgo = {};
    keys.forEach(function (k) {
      var parts = k.split('_CL');
      var algo = parts[0];
      var cl   = parts[1];
      if (!byAlgo[algo]) byAlgo[algo] = [];
      byAlgo[algo].push({ key: k, cl: parseFloat(cl) });
    });
    Object.keys(byAlgo).forEach(function (a) {
      byAlgo[a].sort(function (x, y) { return x.cl - y.cl; });
    });

    var dashStyles = [[], [4,4], [8,4], [2,3], [10,2,2,2]];

    function buildDatasets(field) {
      var datasets = [];
      Object.keys(byAlgo).forEach(function (algo, ai) {
        var col = ALGO_COLORS[algo] || [C.roseDeep, C.mintDeep, C.lavenderDeep, C.peachDeep][ai % 4];
        byAlgo[algo].forEach(function (entry, di) {
          var pts = curves[entry.key];
          datasets.push({
            label: algo + ' CL=' + entry.cl,
            data: pts.map(function (p) { return { x: p.x, y: p[field] }; }),
            borderColor: col,
            backgroundColor: col,
            borderDash: dashStyles[di % dashStyles.length],
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 1.6,
            tension: 0.25
          });
        });
      });
      return datasets;
    }

    function makeChart(canvasId, field, yTitle) {
      new Chart(document.getElementById(canvasId), {
        type: 'line',
        data: { datasets: buildDatasets(field) },
        options: {
          maintainAspectRatio: false,
          responsive: true,
          animation: { duration: 1100, easing: 'easeOutQuart' },
          interaction: { intersect: false, mode: 'nearest', axis: 'x' },
          plugins: {
            legend: { position: 'top', align: 'end',
                      labels: { boxWidth: 16, font: { size: 11 }, padding: 10 } },
            tooltip: {
              callbacks: {
                label: function (ctx) {
                  return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3);
                }
              }
            }
          },
          scales: {
            x: Object.assign(gridY(), {
              type: 'linear',
              title: { display: true, text: 'Training step', color: C.inkSoft, font: { weight: '500' } }
            }),
            y: gridY({
              title: { display: true, text: yTitle, color: C.inkSoft, font: { weight: '500' } }
            })
          }
        }
      });
    }

    makeChart('saferlRewardChart', 'r', 'Episode reward');
    makeChart('saferlCostChart',   'c', 'Episode cost');
  }

  /* ============================================================
     CHART 5: BASELINES
     ============================================================ */
  function renderBaselineChart(envId, data) {
    var baselines = data.baselines || {};
    var rl        = data.clean_rl || {};

    var entries = [];
    Object.keys(baselines).forEach(function (k) {
      var v = baselines[k];
      entries.push({ name: k, mean: v.mean, std: v.std, kind: 'baseline' });
    });
    Object.keys(rl).forEach(function (k) {
      var v = rl[k];
      entries.push({ name: k + ' (RL)', mean: v.mean, std: v.std, kind: 'rl' });
    });
    if (!entries.length) return;
    entries.sort(function (a, b) { return b.mean - a.mean; });

    var labels = entries.map(function (e) { return e.name; });
    var means  = entries.map(function (e) { return e.mean; });
    var stds   = entries.map(function (e) { return e.std; });
    var colors = entries.map(function (e) {
      return e.kind === 'rl' ? C.lavenderDeep : C.mintDeep;
    });

    new Chart(document.getElementById('baselineChart'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          data: means,
          backgroundColor: colors,
          borderRadius: 5,
          borderSkipped: false,
          barThickness: 22,
          _stds: stds
        }]
      },
      options: {
        indexAxis: 'y',
        maintainAspectRatio: false,
        responsive: true,
        animation: { duration: 1100, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: function (items) { return items[0].label; },
              label: function (ctx) {
                var s = ctx.dataset._stds ? ctx.dataset._stds[ctx.dataIndex] : null;
                return 'Reward: ' + ctx.parsed.x.toFixed(3) +
                  (s !== null ? ' \u00b1 ' + s.toFixed(3) : '');
              }
            }
          }
        },
        scales: {
          x: gridY({
            title: { display: true, text: 'Mean episode reward',
                     color: C.inkSoft, font: { weight: '500' } }
          }),
          y: gridX({ ticks: { color: C.ink, font: { weight: '600' } } })
        }
      }
    });

    document.getElementById('baselineCaption').textContent =
      'Mean reward across all evaluation episodes (n typically 100). Lavender = clean RL test (multi-seed); mint = non-RL baselines.';
  }

})();
