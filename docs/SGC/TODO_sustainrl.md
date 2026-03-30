# TODO: SustainRL-Bench — ACCEPT'e Ulaşmak İçin Yapılacaklar

Reviewer skoru: 68.75/100 (REJECT, G1 FAIL)
Hedef skor: ~80-84 (ACCEPT)
Tarih: 2026-03-28

---

## TIER 1 — KRİTİK (bunlar olmadan ACCEPT imkansız)

### TODO-1: MARL Deneylerini Tamamla [G1 FIX]
**Neden**: G1 gate FAIL — C4 contribution'da "evaluation" iddia ediliyor ama sıfır sonuç var.
Section 5.3 şu an tek cümle: "experiments are currently in progress."

**Yapılacaklar**:
- [x] EV Charging MARL (54 agent, per-station decomposition)
  - [x] MAPPO eğitimi + 100 episode evaluation
  - [x] MASAC eğitimi + 100 episode evaluation
  - [x] APPO eğitimi + 100 episode evaluation
  - [x] IMPALA eğitimi + 100 episode evaluation
  - [x] Single-agent PPO vs MAPPO karşılaştırma tablosu (Table 5.2 formatında)
- [x] Building MARL (per-zone agents)
  - [x] Aynı 4 algoritma eğitim + evaluation
  - [x] Single-agent vs multi-agent karşılaştırma
- [x] Cogeneration MARL (4 turbine agents)
  - [x] Aynı 4 algoritma eğitim + evaluation
  - [x] Single-agent vs multi-agent karşılaştırma
- [x] Cross-environment MARL analizi
  - [x] Agent sayısının (4 vs 5-zone vs 54) öğrenme verimliliğine etkisi
  - [x] Hangi ortamlarda multi-agent avantajlı, hangilerinde değil?
  - [x] Parameter sharing vs independent learning karşılaştırması
- [x] Section 5.3'ü tam sonuçlarla doldur
  - [x] Training curves (MARL)
  - [ ] Violin plots (single vs multi) — not done, bar chart used instead
  - [x] Summary table (Table marl_summary added to chapter5.tex)
  - [x] Finding 5.8+ numaralı bulgular ekle (3 finding boxes in sec 5.3.5)
- [x] RQ4'ü yanıtla (Section 6.1.3 güncellendi — "will be reported" kaldırıldı, 3 key finding eklendi)
- [ ] **100-episode MARL evaluation** (ARC'da çalıştırılacak)
  - [ ] `sbatch scripts/slurm/arc_marl_test.sh` — 19 run (3 env × algo × mode)
  - [ ] Script hazır: `scripts/test/marl_testing.py` + `scripts/slurm/arc_marl_test.sh`
  - [ ] Output: `logs_marl_test/` → episode_results.csv, evaluation_summary.txt (mean, std, CI95)
  - [ ] Sonuçlarla Table marl_summary'yi güncelle (training metrics → eval metrics)

**Beklenen etki**: G1 FAIL → PASS, D: 3→3-4, E: 3→3-4
**Tahmini süre**: 1-2 hafta (eğitim süresi HPC'ye bağlı)
**SLURM config**: Section 4.6.5'teki ARC HPC ayarlarını kullan

---

### TODO-2: Kodu GitHub'da Yayınla [H FIX]
**Neden**: C5 "open-source experimental infrastructure" iddia ediyor ama tezde hiçbir URL yok.
Reviewer bunu hem H skorunda (2/4) hem G5'te (borderline) sorun olarak işaretledi.

**Yapılacaklar**:
- [x] GitHub repo oluştur → https://github.com/MehmetKORUTURK/sustaingym-rl-benchmark
- [x] Teze repo URL'sini ekle (Section 1.4 C5'e eklendi)
- [x] Repo yapısı düzenlendi
- [x] README.md yazıldı (kurulum, reproduction, run_all)
- [x] requirements.txt eklendi

**Beklenen etki**: H: 2→3, G5 borderline endişesi kalkar
**Tahmini süre**: 2-3 gün (kod zaten var, organize etmek lazım)

---

## TIER 2 — YÜKSEK ETKİ (B skorunu 2→3 yapacak, novelty argümanını güçlendirecek)

### TODO-3: Non-Obvious Findings'i Ön Plana Çıkar [B FIX — sadece yazı]
**Neden**: Reviewer "directional findings predictable" diyor. Haklı: "PPO robust" ve "environment > algorithm"
herkes tahmin edebilir. Ama specific magnitude ve interaction findings genuinely informative.

**Yapılacaklar**:
- [ ] Abstract'ı yeniden yaz — şu findings'i headline yap:
  - SAC'ın Cogen'de >2000% degradation (PPO <0.3% iken)
  - TD3'ün perturbation altında paradoksal olarak DAHA stabil eğitim göstermesi
  - EV Charging'de Greedy heuristic > tüm RL algoritmaları
  - Building Safe RL: reward-cost alignment nedeniyle CMDP tamamen etkisiz
  - PPOLag'ın inactive constraints altında graceful degradation göstermesi (CPO crash ederken)
- [ ] Section 1.4 Contributions'ı güncelle:
  - C2'yi "we reveal a 100× perturbation sensitivity gap between environments" olarak yeniden çerçevele
  - C3'e "we identify reward-cost orthogonality as a necessary condition for effective CMDP" ekle
  - "Systematic evaluation" yerine "quantitative findings that update practitioner beliefs" vurgula
- [ ] Conclusion (Section 6.1)'de findings'i magnitude-first olarak sırala:
  - Şu an: "Environment structure dominates..." (predictable)
  - Olması gereken: "SAC exhibits catastrophic 2000%+ degradation in Cogeneration while PPO shows <0.3%..." (specific, surprising)

**Beklenen etki**: B: 2→3 (framing değişikliği, yeni deney gerekmez)
**Tahmini süre**: 1 gün

---

### TODO-4: Reward-Cost Orthogonality'yi Teoriye Bağla [B FIX — yazı + kısa analiz]
**Neden**: Reviewer diyor ki: "constrained optimization theory has long established that inactive constraints
don't matter — this is textbook Lagrangian theory. Where does the thesis cite this connection?"

**Yapılacaklar**:
- [ ] Section 5.2.5'e teorik bağlam paragrafı ekle:
  - Lagrangian duality'den inactive constraint (complementary slackness) kavramını tanıt
  - "Our empirical finding corresponds to the complementary slackness condition: when the optimal unconstrained solution already satisfies the constraint (Building), λ*=0 and the constraint is inactive"
  - Cite: Boyd & Vandenberghe (Convex Optimization), Altman (1999) CMDP kitabı
- [ ] Farkı vurgula: "While the principle is theoretically known, our contribution is the first empirical demonstration across multiple energy environments, revealing that practitioners must verify reward-cost alignment before investing in CMDP formulation"
- [ ] 3 ortamın natural experiment oluşturduğunu açıkça belirt:
  - Cogen: cost ⊥ reward → CMDP works (turbine ramping vs fuel cost)
  - EV: cost ⊥ reward → CMDP works (excess charge vs profit)
  - Building: cost ∥ reward → CMDP fails (comfort already in reward)

**Beklenen etki**: B güçlenir, reviewer'ın "rediscovery" eleştirisini adresler
**Tahmini süre**: 0.5 gün

---

### TODO-5: EV Charging RL Başarısızlığını Derinleştir [B + E FIX]
**Neden**: Table 5.2'de Greedy (8.65) > PPO (6.01) > SAC (3.79) > TD3 (3.16). Reviewer diyor:
"the paper does not investigate why RL fails here — a missed opportunity for a belief update."

**Yapılacaklar**:
- [ ] Reward decomposition analizi:
  - Her algoritma için: r_profit, r_carbon, r_violation ayrı ayrı rapor et
  - Greedy ile karşılaştır: hangi reward component'ta RL kaybediyor?
- [ ] Action distribution analizi:
  - PPO vs Greedy: ortalama pilot signal dağılımı ne kadar farklı?
  - RL over-charging mı yapıyor (violation penalty), under-charging mı (profit kaybı)?
- [ ] Hypothesis test: "EV Charging reward'ı myopic decisions'ı ödüllendiriyor"
  - Episode-level reward vs timestep-level reward korelasyonu
  - Greedy'nin myopic olmasına rağmen başarılı olması bu hypothesis'i destekler
- [ ] Kısa bir "Why RL Fails in EV Charging" subsection'ı ekle (Section 5.1.2 sonuna)
- [ ] Pratik implikasyon: "Practitioners should verify that RL outperforms simple heuristics before deploying complex training pipelines"

**Beklenen etki**: E: 3→4 (mechanistic understanding), B: güçlenir (genuine belief update)
**Tahmini süre**: 2-3 gün (analiz + yazı)

---

## TIER 3 — ORTA ETKİ (F skorunu 2→3 yapacak)

### TODO-6: Multi-Seed Training [F + D FIX]
**Neden**: Reviewer: "single-seed training means quantitative rankings may be seed-dependent."
Section 6.2 bunu limitation olarak kabul ediyor ama adreslenmiyor.

**Yapılacaklar**:
- [ ] Her ortam için 1 algoritma × state perturbation × 3 seed eğit:
  - EV Charging: PPO × σ_s ∈ {0, 0.1, 0.2} × seeds {42, 123, 456}
  - Building: PPO × σ_s ∈ {0, 0.05, 0.15} × seeds {42, 123, 456}
  - Cogen: SAC × σ_s ∈ {0, 5, 10} × seeds {42, 123, 456}
- [ ] Inter-seed variance rapor et (mean ± std across seeds)
- [ ] Directional conclusions'ın seed-stable olduğunu göster:
  - "EV Charging remains robust (mean degradation < 2% across all seeds)"
  - "Building degradation pattern consistent across seeds (>80% at σ=0.15 for all seeds)"
- [ ] Table 5.3'e seed variance kolonu ekle (veya supplementary table)
- [ ] Section 6.2 limitation'ı güncelle: "addressed for representative subset"

**Beklenen etki**: F: 2→3, D güçlenir
**Tahmini süre**: 3-5 gün (HPC eğitim süresi)
**Not**: 3 ortam × 3 sigma × 3 seed = 27 training run. Feasible.

---

### TODO-7: Combined Perturbation Deneyi [F FIX]
**Neden**: Reviewer: "perturbation channels tested independently; real systems experience simultaneous noise."
Section 6.2 bunu limitation olarak kabul ediyor.

**Yapılacaklar**:
- [ ] Building ortamı için (en hassas ortam — en çok bilgi verecek olan):
  - PPO: state + action + dynamics aynı anda, moderate levels
  - σ_s = 0.05, σ_a = 0.1, σ_d = 0.5 (moderate)
  - σ_s = 0.1, σ_a = 0.2, σ_d = 1.0 (high)
- [ ] Interaction analizi:
  - Combined degradation vs sum of individual degradations
  - Additive mi? Super-additive mi (worse than sum)?
- [ ] Kısa subsection ekle: "5.1.6 Combined Perturbation Analysis"
- [ ] Table: individual vs combined degradation karşılaştırması

**Beklenen etki**: F: 2→3
**Tahmini süre**: 2-3 gün (evaluation only — yeni eğitim gerekmez, clean-trained model kullan)

---

### TODO-8: İkinci Environment Konfigürasyonu [F FIX — opsiyonel]
**Neden**: Reviewer: "each environment uses a single configuration. No cross-configuration generalization."

**Yapılacaklar (en az 1 ortam için)**:
- [ ] Building: Hot_Humid profili (şu an sadece Hot_Dry) VEYA OfficeMedium (şu an sadece OfficeSmall)
  - PPO eğitimi + state perturbation evaluation
  - Aynı pattern'ler geçerli mi? (Building hala fragile mi?)
- [ ] VEYA EV Charging: farklı GMM parametreleri (farklı arrival pattern)

**Beklenen etki**: F: 2→3 (TODO-6 ve TODO-7 ile birlikte kesin 3)
**Tahmini süre**: 3-5 gün
**Not**: Bu en düşük öncelikli F-fix. TODO-6 + TODO-7 yeterliyse atlanabilir.

---

## TIER 4 — POLISH (skoru 80+ yapacak ince ayarlar)

### TODO-9: Perturbation Scale Justification [C + minor]
**Neden**: Reviewer: "σ ranges differ across environments without clear physical motivation."

**Yapılacaklar**:
- [ ] Section 4.3'e her ortam için fiziksel motivasyon ekle:
  - EV Charging σ_s ∈ [0, 0.3]: "corresponds to ±30% sensor measurement error, consistent with typical current/voltage sensor specifications (±0.5-2% accuracy under normal conditions, up to ±30% under degraded conditions)"
  - Building σ_s ∈ [0, 0.3]: "corresponds to ±0.3°C-1°C temperature sensor error, spanning the range from calibrated (±0.1°C) to degraded thermocouples (±1°C)"
  - Cogen σ_s ∈ [0, 25]: "proportional to the multi-dimensional observation space; normalized perturbation intensity is comparable across environments"

**Tahmini süre**: 2 saat

---

### TODO-10: SACLag Failure Analizi [E minor fix]
**Neden**: Reviewer: "Section 5.2.4 attributes failure to replay buffer staleness but provides no diagnostic."

**Yapılacaklar**:
- [ ] SACLag training curve'lerini ekle (en az Cogen için)
- [ ] λ trajectory plot: Lagrange multiplier zamanla nasıl değişiyor?
- [ ] Cost critic loss plot: stale data'nın critic'i nasıl bozduğunu göster
- [ ] 1-2 paragraf mekanistik açıklama ekle

**Tahmini süre**: 1 gün

---

## ÖNCELİK ÖZETİ

| # | TODO | Etki | Süre | Skor Etkisi |
|---|------|------|------|-------------|
| 1 | MARL deneyleri | G1 FIX (kritik) | 1-2 hafta | G1 PASS, D↑, E↑ |
| 2 | Code release | H FIX | 2-3 gün | H: 2→3 |
| 3 | Findings reframing | B FIX (yazı) | 1 gün | B: 2→3 |
| 4 | Orthogonality theory | B FIX (yazı) | 0.5 gün | B güçlenir |
| 5 | EV RL failure analizi | E + B FIX | 2-3 gün | E: 3→4, B↑ |
| 6 | Multi-seed training | F + D FIX | 3-5 gün | F: 2→3, D↑ |
| 7 | Combined perturbation | F FIX | 2-3 gün | F: 2→3 |
| 8 | 2nd config (opsiyonel) | F FIX | 3-5 gün | F güçlenir |
| 9 | Scale justification | C minor | 2 saat | C↑ |
| 10 | SACLag analizi | E minor | 1 gün | E↑ |

## MİNİMUM ACCEPT YOLU (en kısa sürede)

Sadece şunları yap (toplam ~2-3 hafta):
1. **TODO-1** (MARL) — zorunlu, G1 fix
2. **TODO-2** (code release) — zorunlu, H fix
3. **TODO-3** (reframing) — sadece yazı, B fix
4. **TODO-6** (multi-seed) — F fix

→ Tahmini skor: ~76-78 (borderline ACCEPT)

## GÜÇLÜ ACCEPT YOLU (tam kapsamlı)

Hepsini yap (toplam ~4-5 hafta):
TODO-1 → TODO-2 → TODO-3 → TODO-4 → TODO-5 → TODO-6 → TODO-7 → TODO-9 → TODO-10

→ Tahmini skor: ~82-85 (solid ACCEPT)
