/* ============================================================
   SustainRL-Bench  --  vanilla JS for tabs, counters, reveal
   ============================================================ */

(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', init);

  function init() {
    setupTabs();
    setupResultReveal();
    setupCopyButton();
    setupIntersectionAnimations();
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

  /* ---------------- Result cards: click to reveal ---------------- */
  function setupResultReveal() {
    var cards = document.querySelectorAll('.result-card');

    cards.forEach(function (card) {
      card.addEventListener('click', function () {
        if (card.classList.contains('revealed')) return;

        var img = card.querySelector('img[data-src]');
        if (img && !img.src) {
          img.src = img.getAttribute('data-src');
        }

        // small delay so the wipe animation reads after expansion starts
        requestAnimationFrame(function () {
          card.classList.add('revealed');
        });
      });
    });
  }

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
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(showCopied);
      } else {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); showCopied(); } catch (_) {}
        document.body.removeChild(ta);
      }

      function showCopied() {
        var prev = btn.textContent;
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = prev;
          btn.classList.remove('copied');
        }, 1600);
      }
    });
  }

  /* ---------------- Stats counter + kappa bars on enter ---------------- */
  function setupIntersectionAnimations() {
    if (!('IntersectionObserver' in window)) {
      // Fallback: just set final values
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
        var eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
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
      setTimeout(function () {
        fill.style.width = pct + '%';
      }, i * 180);
    });
  }
})();
