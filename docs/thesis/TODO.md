# Thesis TODO List

## 🔴 KALAN İŞLER (Figür Yeniden Üretme)

Aşağıdaki iki item plot script düzeltmeleri yapılmış ama figürler henüz yeniden üretilmemiş.
ARC'ta çalıştırılması gerekiyor.

### A. Figürleri Yeniden Üret (v0 kaldırma + Safe RL boyut büyütme)

**Neden**: `style.py` ve `omni_plot.py` güncellendi — figür title'larında "EVCharging-v0" → "EV Charging", Safe RL figsize büyütüldü.

**Komutlar** (ARC'ta çalıştır):
```bash
# 1. Training curves — her env × algo × noise type
for env in evcharging building cogen; do
  for algo in PPO SAC TD3; do
    for dt in DS DA DE; do
      python scripts/plot/stdrl_plot.py --env $env --algo $algo --dt $dt --auto_ylim
    done
  done
done

# 2. Baseline comparison
for env in evcharging building cogen; do
  python scripts/plot/baseline_plot.py --env $env --plot all
done

# 3. Safe RL plots (büyütülmüş figsize ile)
for env in evcharging building cogen; do
  python scripts/plot/omni_plot.py --env $env --t_steps 20000 --w_size 500
done

# 4. Post-test plots (violin, panel, etc.)
for env in evcharging building cogen; do
  for algo in PPO SAC; do
    python scripts/plot/stdrl_post_plot.py --env $env --algo $algo --plot all
  done
done
```

**Sonra**: Yeni PNG'leri `docs/thesis/figures/` altına kopyala.

### B. LaTeX Derleme ve Kontrol

- [ ] TikZ figürlerin (Figure 4.1 framework, Figure 4.3 workflow) doğru render olduğunu kontrol et
- [ ] Subfigure'lerin (0.32\textwidth) okunabilir olduğunu kontrol et
- [ ] Safe RL figürlerin (1.0\textwidth) legend'lerinin okunabilir olduğunu kontrol et
- [ ] `\today` → sabit savunma tarihi (tarih belirlendikten sonra)

### C. MARL Results (Chapter 5)

MARL deneyleri tamamlandığında eklenecek:
- [ ] EVCharging MARL sonuçları (54 agent, APPO/PPO/SAC/IMPALA)
- [ ] Building MARL sonuçları (AC-enabled zones, SAC/PPO)
- [ ] Cogen MARL sonuçları (4 agent, PPO/APPO/IMPALA)
- [ ] Single-agent vs multi-agent karşılaştırma tablosu
- [ ] MARL findings (RQ4 cevapla)
- **Konum**: chapter5.tex MARL section (şu an "in progress" placeholder)

---

## ✅ TAMAMLANAN İŞLER (22/22)

## Priority Legend
- 🔴 CRITICAL — thesis savunulamaz bunlar olmadan
- 🟡 IMPORTANT — kalite/tutarlılık için gerekli
- 🟢 NICE-TO-HAVE — iyileştirme

---

## 🔴 CRITICAL

### ~~1. Chapter 6: Conclusion and Future Work YAZ~~ ✅ TAMAMLANDI
- [x] `chapter6.tex` oluşturuldu
- [x] `main.tex`'e `\input{chapter6}` eklendi
- [x] Key findings özeti (RQ1-RQ6 cevapları)
- [x] Limitations section (6 madde)
- [x] Future work (6 subsection: noise-aware training, transfer learning, improved safe RL, scalable MARL, real-world deployment, extended benchmark)
- [x] chapter1.tex'teki hardcoded "Chapter 6" → `\ref{ch:conclusion}` düzeltildi

### 2. MARL Results Ekle (Chapter 5)
- [ ] EVCharging MARL sonuçları (54 agent, APPO/PPO/SAC/IMPALA)
- [ ] Building MARL sonuçları (AC-enabled zones, SAC/PPO)
- [ ] Cogen MARL sonuçları (4 agent, PPO/APPO/IMPALA)
- [ ] Single-agent vs multi-agent karşılaştırma tablosu
- [ ] MARL findings (RQ5 cevapla)
- **Konum**: chapter5.tex:742-744 (şu an "in progress" placeholder)

### ~~3. TD3 Noise Results Ekle (Chapter 5)~~ ✅ TAMAMLANDI
- [x] EV Charging TD3 subsection — monotonic decline, no noise experiments (negative finding)
- [x] Building TD3 subsection — catastrophic degradation -50→-150, no noise experiments
- [x] Cogen TD3 subsection — mid-training crash + noise-as-regularization finding
- [x] Finding 5.4 (TD3 Training Instability) eklendi
- [x] Cogen intro text güncellendi (TD3 mention eklendi)
- **Bulgu**: TD3 tüm env'lerde unstable; Cogen'de noise-trained TD3 daha stabil (implicit regularization)

---

## 🟡 IMPORTANT — Tutarsızlıklar

### ~~4. EV Charging Safe RL Cost Limits Senkronize Et~~ ✅ TAMAMLANDI
- [x] Chapter 4 tablo: `{1, 5, 25, 1000}` → `{3, 5, 25, 1000, 10000}` düzeltildi
- [x] Chapter 4 paragraf: env-specific limit açıklamaları eklendi
- [x] Chapter 5 intro: `{3, 5, 25}` → `{3, 5, 25, 1000, 10000}` düzeltildi
- [x] Chapter 5 tablo: d=1000 ve d=10000 satırları eklendi (log verilerinden)
- [x] Chapter 5 algo text: CPO, OnCRPO, FOCOPS değerleri güncellendi
- [x] Chapter 5 cross-algo: CL5 ve CL1000 figürleri eklendi (mevcut figürlerden)

### ~~5. Building Safe RL Cost Limits & Cost Function Senkronize Et~~ ✅ TAMAMLANDI
- [x] Chapter 4 tablo: `{1, 5, 25, 1000}` → `{25, 50, 100, 200}` düzeltildi (figürlerden)
- [x] Chapter 4 paragraf: Building limits açıklaması güncellendi
- [x] Chapter 5 tablo: `d1, d2, d3, d4` → `d=25, d=50, d=100, d=200` düzeltildi
- [x] Chapter 5 CPO text: `d=200, 500` → `d=100, 200` düzeltildi

### ~~6. Experiment Matrix Güncelle~~ ✅ TAMAMLANDI
- [x] Safe RL: "OnCRPO" → "PPOLag, CPO, OnCRPO, FOCOPS" düzeltildi
- [x] StdRL: Her env için obs/act/env level sayıları doğru yazıldı (loglardan sayıldı)
- [x] StdRL: TD3 baseline-only notu eklendi (noise experiments hariç)
- [x] MARL: Tüm algoritmalar eklendi (EV/BU: 4 algo, CO: 3 algo)
- [x] Total: ~444 → 212 unique configs düzeltildi
- [x] Paragraf: "735 runs" → "over 300 runs" düzeltildi

### ~~7. TCTN/TNTC Scope'tan Çıkarıldı~~ ✅ TAMAMLANDI
- [x] Chapter 1: RQ3 (Distribution Shift) kaldırıldı, RQ4→RQ3, RQ5→RQ4, RQ6→RQ5 renumber
- [x] Chapter 2: TCTN/TNTC paragrafı güncellendi (general approach anlatılıyor)
- [x] Chapter 3: Distribution Shift Protocols subsection tamamen kaldırıldı
- [x] Chapter 5: RQ referansları güncellendi (RQ4→RQ3, RQ6→RQ5)
- [x] Chapter 6: RQ referansları güncellendi
- [x] main.tex: TCTN/TNTC nomenclature'dan kaldırıldı

### ~~8. IQM vs Mean Tutarsızlığı~~ ✅ TAMAMLANDI
- [x] Chapter 4: IQM+5seeds methodology → "mean with bootstrap CI + violin distributional analysis"
- [x] Chapter 4: Evaluation Layer description güncellendi
- [x] Chapter 2: "Our statistical methodology" paragraph güncellendi (IQM future work olarak acknowledge)
- [x] IQM definition Chapter 2'de kalıyor (literature review olarak)
- [x] main.tex nomenclature'da IQM kalıyor (tanım hala geçerli)
- **Multi-seed IQM analizi**: SmartGridComm paper'a taşındı → `docs/SGC/TODO.md`

### ~~9. Non-RL Baseline Karşılaştırması~~ ✅ TAMAMLANDI (GERİ EKLENDİ)
- [x] Chapter 4: Baseline tanımları kaldırıldı (methodology'de gereksizdi)
- [x] Chapter 5: Non-RL baseline karşılaştırma section eklendi (tablo + 3 compare figür)
- [x] Baselines: Random, DoNothing, Greedy, MPC, OfflineOptimal (env-specific)
- [x] Key finding: EV'de RL heuristik'lerden kötü, Building'de PPO açık ara iyi, Cogen'de SAC/TD3 hakim

### ~~10. Abstract'teki "700 runs" vs Experiment Matrix "444 configs"~~ ✅ TAMAMLANDI (#6 ile birlikte)
- [x] main.tex abstract: "over 700" → "over 200 unique configs / over 300 runs"
- [x] chapter4.tex tablo: ~444 → 212, paragraf: 735 → "over 300"
- [x] Tutarlı: abstract ↔ tablo ↔ paragraf hepsi senkron

---

## 🟡 IMPORTANT — Eksik Figürler & Placeholderlar

### ~~11. Placeholder Figürleri Oluştur veya Kaldır~~ ✅ TAMAMLANDI
- [x] Figure 4.1: TikZ framework architecture diagram oluşturuldu (4 katman, bileşenler, oklar)
- [x] main.tex: `tikz` + `arrows.meta` paketleri eklendi
- [x] Figure 4.2 (RC thermal): placeholder kaldırıldı, equations yeterli
- [x] Figure 4.3 (workflow): TikZ workflow diagram oluşturuldu (Configure → SLURM → Train → Save → Evaluate → Analysis)
- **Not**: LaTeX derlemeden TikZ çıktısı kontrol edilemez — derleme sonrası kontrol gerekli

### ~~12. EV Charging Safe RL Cross-Algo Karşılaştırma Eksik~~ ✅ TAMAMLANDI (#4 ile birlikte)
- [x] CL5 ve CL1000 cross-algo figürleri eklendi
- [x] EV artık 4 cross-algo figür: CL3, CL5, CL25, CL1000

### ~~13. Cogen SAC Action Noise/Violin Eksik~~ ✅ TAMAMLANDI
- [x] action_violin.png figürü zaten mevcutmuş — chapter5'e eklendi
- [x] Text güncellendi: action noise degradation detayları ve figür referansı eklendi
- [x] Obs + action + env noise 3'ü de artık hem figür hem text'te ele alınıyor

---

## 🟡 IMPORTANT — Formatlama & Görsel

### ~~14. Figür Sayısını Azalt / Subfigure Birleştirme~~ ✅ TAMAMLANDI
- [x] 12 training curve triplet (DS/DA/DE) → 12 subfigure (3×0.32\textwidth) = 24 figür azaldı
- [x] 10 violin triplet (obs/act/env) → 10 subfigure = 20 figür azaldı
- [x] Toplam: ~44 ayrı figür → ~22 combined figür (net ~22 float azaldı)
- [x] Panel figürler + TD3 single figures olduğu gibi bırakıldı
- [x] Tüm eski label'lar korundu (cross-ref break yok)

### 15. Figür Title'larından "v0" Kaldır — SCRIPT'LER DÜZELTİLDİ
- [x] `style.py`: ENV_TITLES güncellendi (EVCharging-v0 → EV Charging, Building-v0 → Building, Cogen-v0 → Cogeneration)
- [x] `stdrl_plot.py`: ENV_DEFAULTS title'ları güncellendi
- [x] `baseline_plot.py`, `stdrl_post_plot.py`: zaten doğruydu
- [x] `omni_plot.py`: style.py'dan import, otomatik düzeldi
- [ ] **FIGÜRLER YENİDEN ÜRETİLMELİ** — ARC'ta aşağıdaki komutları çalıştır:
  ```
  # Training curves (her env × her algo × her noise type)
  python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt DS
  python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt DA
  python scripts/plot/stdrl_plot.py --env evcharging --algo PPO --dt DE
  # ... (tüm kombinasyonlar)

  # Safe RL plots
  python scripts/plot/omni_plot.py --env evcharging --t_steps 20000 --w_size 500
  python scripts/plot/omni_plot.py --env building --t_steps 20000 --w_size 500
  python scripts/plot/omni_plot.py --env cogen --t_steps 20000 --w_size 500

  # Baseline comparison
  python scripts/plot/baseline_plot.py --env evcharging
  # ...
  ```
- [ ] Yeni PNG'leri `docs/thesis/figures/` altına kopyala

### ~~16. X-axis Tutarsızlığı~~ ✅ TAMAMLANDI
- [x] Chapter 5 intro'ya "A note on figure axes" paragrafı eklendi
- [x] Training Episode (SB3) vs Training Epoch (OmniSafe) vs Noise Level (test) farkı açıklandı
- [x] Farklı framework'lerin farklı birim kullanması doğal — sorun değil

### ~~17. Building PPO Panel Plot Env Noise X-axis~~ ✅ SORUN YOK
- [x] Kontrol edildi: Env noise gerçekten 0.1-3.0 arası test edilmiş (loglardan doğrulandı)
- [x] Obs/Act noise 0-0.40 arası, ama Env noise farklı scale (dict-based: out_temp=1C, ghi=50W/m² per unit)
- [x] Panel figürü doğru — farklı noise channel'ların farklı scale kullanması normal

### ~~18. Safe RL Dual-Panel Figür Boyutu~~ ✅ SCRIPT + TEX DÜZELTİLDİ
- [x] `omni_plot.py`: figsize (14,5) → (16,5.5) büyütüldü
- [x] chapter5.tex: 20 Safe RL figür `width=0.9\textwidth` → `width=1.0\textwidth`
- [ ] **FIGÜRLER YENİDEN ÜRETİLMELİ** — #15 ile birlikte ARC'ta çalıştırılacak

---

## 🟢 NICE-TO-HAVE

### ~~19. Title Page Tamamla~~ ✅ TAMAMLANDI
- [x] Committee members eklendi: Ali Mehrizi-Sani, Alkan Soysal
- [ ] `\today` → sabit tarih (savunma tarihi belirlendikten sonra)

### ~~20. Acknowledgments Yaz~~ ✅ TAMAMLANDI
- [x] Advisor (Dr. Ming Jin), aile, Blacksburg arkadaşlar, MEB YLSY bursu teşekkürü yazıldı

### ~~21. Noise Sensitivity Coefficient (κ) Kaldırıldı~~ ✅ TAMAMLANDI
- [x] Chapter 4: κ tanımı Derived Metrics'ten kaldırıldı (normalized drop + Pareto metric kaldı)
- [x] κ hesaplama SGC paper TODO'ya taşındı → `docs/SGC/TODO.md` #9

### ~~22. Baseline Tablo vs Text: "profit ratio" Kaynağı~~ ✅ TAMAMLANDI
- [x] Profit ratio claim kaldırıldı (tabloda yok, kaynağı belirsiz)
- [x] EV Charging paragrafı sadece reward karşılaştırmasına odaklanıyor
- [ ] Ya tabloya ekle ya da referans ver

### 23. ~~REFERANS SORUNLARI~~ ✅ TAMAMLANDI
- [x] Orphan citations düzeltildi (9 adet)
- [x] Missing bib entries eklendi (10 adet)
- [x] Bib key mismatch düzeltildi (lee2021→lee2019, ray2019benchmarking)
- [x] Volume/pages eklendi (20+ entry)

---

## İlerleme Notu
- Referans sorunları 2026-03-18 tarihinde çözüldü
- Sıradaki: Kritik items (1-3) → Tutarsızlıklar (4-10) → Figürler (11-18) → Nice-to-have (19-22)
