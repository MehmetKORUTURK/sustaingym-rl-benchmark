# Cok-Ajanli Pekistirmeli Ogrenme (MARL) Egitim Scriptleri - Teknik Rapor

**Proje:** RL Benchmark on SustainGym Environments
**Yazar:** Mehmet Koruturk, Virginia Tech, RoleLab
**Tarih:** Mart 2026
**Framework:** Ray RLlib + PettingZoo

---

## Icerik

1. [Genel Bakis](#1-genel-bakis)
2. [Algoritmalar](#2-algoritmalar)
3. [Policy Modlari](#3-policy-modlari)
4. [Ortam Detaylari](#4-ortam-detaylari)
5. [Hyperparametreler](#5-hyperparametreler)
6. [CPU vs GPU](#6-cpu-vs-gpu)
7. [Output Yapisi](#7-output-yapisi)
8. [CLI Kullanimi](#8-cli-kullanimi)
9. [Deneysel Plan](#9-deneysel-plan)

---

## 1. Genel Bakis

### 1.1 MARL Nedir ve Neden Kullaniyoruz?

Multi-Agent Reinforcement Learning (MARL), birden fazla ajanin ayni ortamda es zamanli olarak karar aldigi ve ogrendigi bir pekistirmeli ogrenme paradigmasidir. Geleneksel single-agent RL'de tek bir karar verici tum aksiyonlari belirlerken, MARL'de her ajan kendi observation'ini alir, kendi policy'si uzerinden aksiyon uretir ve ortak bir reward sinyalinden ogrenme gerceklestirir.

Bu projede MARL kullanmamizin temel nedenleri:

1. **Dogal problem yapisi**: Uc ortam da coklu kontrol birimlerine sahiptir:
   - EVCharging: 54 sarj istasyonu, her biri bagimsiz bir pilot sinyal ureticisi
   - Building: 5 HVAC zonesi, her biri bagimsiz bir isi/sogutma kontroloru
   - Cogen: 4 birim (3 gaz turbini + 1 buhar turbini), her biri farkli alt-aksiyonlar kontrol eder

2. **Olceklenebilirlik analizi**: Single-agent yaklasimda tum aksiyonlar tek bir policy tarafindan uretilir (ornegin EVCharging icin 54 boyutlu aksiyon vektoru). MARL yaklasiminda her ajan sadece kendi alt-aksiyonundan sorumludur, bu da policy networklerinin boyutunu onemli olcude azaltir.

3. **Kiyaslama potansiyeli**: Tez calismamizin temel katkilarindan biri, single-agent vs multi-agent yaklasimlarinin performans, ogrenme hizi ve olceklenebilirlik acisindan karsilastirilmasidir.

4. **Gercekci modelleme**: Gercek dunyada bu sistemler genellikle dagitik (distributed) kontrol altindadir. MARL bu dagitik yapiya daha yakin bir modelleme sunar.

### 1.2 Ray RLlib + PettingZoo Framework'u

Projemizde MARL icin iki temel kutuphaneden yararlaniyoruz:

**PettingZoo**, multi-agent ortamlar icin standart bir Python API'sidir. Gymnasium'un single-agent API'sinin multi-agent versiyonu olarak dusunulebilir. Iki temel arayuz sunar:
- `AECEnv`: Ajanlar sirayla hareket eder (turn-based)
- `ParallelEnv`: Tum ajanlar es zamanli hareket eder (simultaneous)

Projemizde tum ortamlar **`ParallelEnv`** arayuzunu kullanir, cunku uc ortamda da ajanlar her timestep'te es zamanli karar verir.

**Ray RLlib**, dagitik pekistirmeli ogrenme icin en kapsamli kutuphanelerden biridir. Temel avantajlari:
- Coklu rollout worker ile paralel veri toplama
- PPO, SAC, APPO, IMPALA, TD3 gibi algoritmalarin hazir implementasyonlari
- Multi-agent native destegi: farkli policy'ler, policy mapping, parameter sharing
- GPU/CPU esnek kaynak yonetimi
- Checkpoint/resume mekanizmasi

### 1.3 ParallelPettingZooEnv Wrapper'i

Ray RLlib, PettingZoo ortamlarini dogrudan desteklemez. Bunun icin `ParallelPettingZooEnv` wrapper sinifini kullaniyoruz:

```python
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
```

Bu wrapper su islevleri gerceklestirir:

1. **API Cevirisi**: PettingZoo'nun `step(actions_dict)` -> `(obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict)` formatini RLlib'in beklentilerine uygun hale getirir.

2. **Ajan Yonetimi**: Episod bittiginde ajanlarin listesini bosaltir (`self.agents = []`), RLlib'e episode'un bittigini bildirir.

3. **Space Dogrulama**: Her ajanin observation ve action space'ini RLlib policy tanimlarina iletir.

Kullanim ornegi (EVCharging):
```python
def env_creator(env_config):
    trace_gen = GMMsTraceGenerator('caltech', 'Summer 2019')
    env = MultiAgentEVChargingEnv(trace_gen, project_action_in_env=True)
    return ParallelPettingZooEnv(env)

register_env("marl_env", env_creator)
```

`register_env` fonksiyonu, ortam yaratma fonksiyonunu RLlib'e kaydeder. Boylece her rollout worker, `env_creator` fonksiyonunu cagirarak kendi ortam instance'ini olusturabilir.

---

## 2. Algoritmalar

### 2.1 PPO (Proximal Policy Optimization)

**Temel Calisma Prensibi:**
PPO, on-policy bir policy gradient algoritmasidir. Her iterasyonda:
1. Rollout worker'lar ortamdan deneyim toplar
2. Toplanan verilerle policy ve value function guncellenir
3. Policy guncelleme sirasinda "clipping" mekanizmasiyla buyuk policy degisiklikleri engellenir

**Clipping Mekanizmasi:**
```
L_clip = min(r(theta) * A, clip(r(theta), 1-eps, 1+eps) * A)
```
Burada `r(theta)` yeni ve eski policy arasindaki olasilik orani, `A` advantage tahmini, `eps` (clip_param) ise kucuk bir esik degeridir (projemizde 0.2).

**MARL'deki Kullanimi:**
- PPO, tum uc ortamda (EVCharging, Building, Cogen) kullanilabilir
- On-policy oldugu icin Dict action space'leri (Cogen) ile uyumludur
- `--shared-policy` ile MAPPO (Multi-Agent PPO) olarak calistirilabilir
- Stabil ogrenme egrileri saglar ancak off-policy algoritmalara gore sample efficiency'si dusuktur

**Ortam Uyumlulugu:**
| Ortam | Destekleniyor | Notlar |
|-------|:---:|-------|
| EVCharging | Evet | 54 agent, on-policy |
| Building | Evet | ~5 agent, on-policy |
| Cogen | Evet | 4 agent, Dict action space uyumlu |

### 2.2 SAC (Soft Actor-Critic)

**Temel Calisma Prensibi:**
SAC, off-policy ve entropy-regularized bir algoritmadır. Maximum entropy framework'u kullanir:
```
J = E[sum(r + alpha * H(pi))]
```
Burada `H(pi)` policy'nin entropisidir ve `alpha` (temperature) otomatik olarak ayarlanir. Bu entropi terimi, policy'nin exploration yapmaya devam etmesini saglar.

**Off-Policy Avantaji:**
SAC, replay buffer kullanarak gecmiste toplanan deneyimleri tekrar tekrar kullanabilir. Bu, sample efficiency'yi onemli olcude arttirir. `n_step=288` ayari ile 288 adimlik (1 episod) n-step return hesaplanir, bu da uzun vadeli reward iliskilerini yakalamaya yardimci olur.

**Neden Cogen'de Kullanilamaz?**
Cogen ortaminda her ajanin action space'i `spaces.Dict` tipindedir ve icinde karisik (mixed) continuous/discrete alt-uzaylar vardir. Ornegin GT1 ajani icin:
```python
{
    'GT1_PWR': Box(continuous),      # Gaz turbini gucu
    'GT1_PAC_FFU': Discrete(2),      # Fan kontrol (on/off)
    'GT1_EVC_FFU': Discrete(2),      # Fan kontrol (on/off)
    'HR1_HPIP_M_PROC': Box(continuous)  # Isitici kontrol
}
```
SAC, sadece continuous action space'leri destekler. Dict icindeki discrete aksiyonlar icin reparameterization trick uygulanamaz, bu nedenle SAC Cogen ortaminda kullanilamaz.

**Ortam Uyumlulugu:**
| Ortam | Destekleniyor | Notlar |
|-------|:---:|-------|
| EVCharging | Evet | Continuous Box(1) per agent |
| Building | Evet | Continuous Box(1) per agent |
| Cogen | Hayir | Dict action space, mixed continuous/discrete |

### 2.3 APPO (Asynchronous PPO)

**Temel Calisma Prensibi:**
APPO, PPO'nun asenkron versiyonudur. Standart PPO'da tum worker'lar ayni policy versiyonuyla veri toplar ve guncelleme yapilir (synchronous). APPO'da ise:

1. Worker'lar bagimsiz olarak veri toplar ve policy'nin farkli versiyonlariyla calisabilir
2. Learner, veri geldiginde hemen guncelleme yapar (worker'larin bitmesini beklemez)
3. Policy lag'i V-trace ile duzeltilir (opsiyonel)

**PPO'dan Farklari:**
| Ozellik | PPO | APPO |
|---------|-----|------|
| Veri toplama | Senkron | Asenkron |
| Worker senkronizasyonu | Tum worker'lar beklenir | Worker beklenmez |
| Policy lag | Yok (hep ayni versiyon) | Var (V-trace ile duzeltilir) |
| Throughput | Worker sayisiyla sinirli | Daha yuksek throughput |
| `sgd_minibatch_size` | Zorunlu | Opsiyonel |

**Neden Kullaniyoruz:**
APPO, ozellikle EVCharging gibi cok ajanli (54 agent) ortamlarda throughput avantaji saglar. 10 rollout worker ile asenkron veri toplama, egitimi onemli olcude hizlandirir. APPO, EVCharging icin varsayilan (default) algoritmamizdir.

**Ortam Uyumlulugu:**
| Ortam | Destekleniyor | Notlar |
|-------|:---:|-------|
| EVCharging | Evet | Varsayilan algoritma, 54 agent |
| Building | Evet | ~5 agent |
| Cogen | Evet | 4 agent, Dict action space uyumlu |

### 2.4 IMPALA (Importance Weighted Actor-Learner Architecture)

**Temel Calisma Prensibi:**
IMPALA, DeepMind tarafindan gelistirilen yuksek throughput'lu bir dagitik RL algoritmasidir. Temel yeniligi **V-trace** importance sampling duzeltmesidir:

```
v_s = V(x_s) + sum_{t=s}^{s+n-1} (gamma * prod_{i=s}^{t-1} c_i) * delta_t_V
```

Burada:
- `c_i = min(c_bar, pi(a_i|x_i) / mu(a_i|x_i))` truncated importance sampling agirligidir
- `delta_t_V = rho_t * (r_t + gamma * V(x_{t+1}) - V(x_t))` temporal difference hatasidir
- `rho_t = min(rho_bar, pi(a_t|x_t) / mu(a_t|x_t))` truncated olasilik orani

**V-trace Parametreleri:**
Projemizde `vtrace_clip_rho_threshold=1.0` ve `vtrace_clip_pg_rho_threshold=1.0` kullaniyoruz. Bu degerler, importance sampling oranlarini 1.0 ile sinirlar ve asiri buyuk duzeltmeleri engeller.

**Neden Eklendi:**
IMPALA, APPO'ya benzer asenkron yapida calisir ancak V-trace duzeltmesi ile daha kararlı ogrenme vaadeder. Ozellikle cok worker'li ortamlarda policy lag problemi ciddi olabilir; V-trace bunu sistematik olarak duzeltir. Bu nedenle APPO ve PPO ile karsilastirilmak uzere deneysel plana dahil edilmistir.

**Ortam Uyumlulugu:**
| Ortam | Destekleniyor | Notlar |
|-------|:---:|-------|
| EVCharging | Evet | V-trace importance sampling |
| Building | Evet | V-trace importance sampling |
| Cogen | Evet | Dict action space uyumlu |

### 2.5 TD3 (Twin Delayed DDPG)

**Temel Calisma Prensibi:**
TD3, DDPG (Deep Deterministic Policy Gradient) algoritmasi uzerine uc temel iyilestirme getirir:

1. **Twin Q-Networks (`twin_q=True`)**: Iki ayri Q-network kullanilir ve minimum deger alinir. Bu, Q-value overestimation problemini azaltir.

2. **Delayed Policy Updates (`policy_delay=2`)**: Policy, Q-network'ten daha az siklıkla guncellenir (her 2 critic guncellemesinde 1 actor guncellemesi). Bu, daha kararli Q-value tahminleri uzerinden policy guncellemesi yapilmasini saglar.

3. **Target Policy Smoothing**: Target policy'nin aksiyonlarina gurultu eklenerek, Q-function'in dar bolgelerden faydalanmasi (exploitation of Q-function errors) engellenir.

**Neden Cogen'de Calismaz:**
TD3, SAC gibi sadece continuous action space'leri destekler. Cogen ortamindaki Dict action space icindeki `Discrete(2)` ve `Discrete(12)` aksiyonlari TD3'un deterministic policy gradient hesaplamasiyla uyumlu degildir. Deterministic policy, surekli uzayda gradient ile optimize edilir; kesikli uzayda bu gradient tanimli degildir.

**Ortam Uyumlulugu:**
| Ortam | Destekleniyor | Notlar |
|-------|:---:|-------|
| EVCharging | Evet | Continuous Box(1) per agent |
| Building | Evet | Continuous Box(1) per agent |
| Cogen | Hayir | Dict action space, discrete aksiyonlar |

### 2.6 Algoritma Ozet Tablosu

| Algoritma | Tip | Action Space | Cogen Uyumu | Parameter Sharing | Replay Buffer |
|-----------|-----|-------------|:-----------:|:-----------------:|:-------------:|
| PPO | On-policy | Any | Evet | Evet | Hayir |
| SAC | Off-policy | Continuous | Hayir | Evet | Evet |
| APPO | On-policy (async) | Any | Evet | Evet | Hayir |
| IMPALA | On-policy (async) | Any | Evet | Evet | Hayir |
| TD3 | Off-policy | Continuous | Hayir | Evet | Evet |

---

## 3. Policy Modlari

### 3.1 Independent Policies (Bagimsiz Policy'ler)

Independent policy modunda, **her ajan icin ayri bir neural network** egitilir. Bu, MARL literaturunde "Independent Learners" olarak da bilinir.

**Nasil Calisiyor:**

```python
# Her ajan icin ayri policy tanimlanir
policies = {
    f"policy_{aid}": (
        None,                                    # Policy sinifi (None = varsayilan)
        sample_env.observation_spaces[aid],       # Ajan'in observation space'i
        sample_env.action_spaces[aid],            # Ajan'in action space'i
        {},                                       # Ek konfigürasyon
    )
    for aid in agent_ids
}

# Her ajan kendi policy'sine yonlendirilir
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_{agent_id}"
```

**Avantajlari:**
- Her ajan kendi ortam ozelliklerine ozgu bir strateji olusturabilir
- Farkli observation/action space'leri olan ajanlar (Cogen) icin tek secenektir
- Non-stationarity problemi azalir (diger ajanlarin policy degisimi daha az etkiler)

**Dezavantajlari:**
- Policy sayisi ajan sayisina esittir (EVCharging'de 54 ayri network)
- Toplam parametre sayisi fazladir; ogrenme daha yavas olabilir
- Transfer learning veya deneyim paylasimi yoktur

**Ortam Bazinda Policy Sayilari (Independent Mod):**
| Ortam | Ajan Sayisi | Policy Sayisi | Aciklama |
|-------|:-----------:|:-------------:|----------|
| EVCharging | 54 | 54 | Her sarj istasyonu icin 1 policy |
| Building | ~5 | ~5 | Her HVAC zonesi icin 1 policy |
| Cogen | 4 | 4 | GT1, GT2, GT3, ST icin 1'er policy |

### 3.2 Shared Policy (Parameter Sharing / MAPPO / MASAC)

Shared policy modunda, **tum ajanlar ayni neural network'u paylastr**. Bu, literaturde "Parameter Sharing" olarak bilinir ve PPO ile birlestirildiginde MAPPO, SAC ile birlestirildiginde MASAC adini alir.

**Nasil Calisiyor:**

```python
# Tek bir policy tanimlaniyor
ref_agent = agent_ids[0]
policies = {
    "shared_policy": (
        None,
        sample_env.observation_spaces[ref_agent],  # Referans ajanin space'i
        sample_env.action_spaces[ref_agent],
        {},
    )
}

# Tum ajanlar ayni policy'ye yonlendirilir
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"
```

**Avantajlari:**
- Cok daha az parametre: 54 yerine tek bir network
- Tum ajanlarin deneyimlerinden ortak ogrenme (sample efficiency yuksek)
- Ozellikle homojen ajanlar icin (EVCharging, Building) cok etkili
- Daha hizli ogrenme ve daha iyi genelleme

**Dezavantajlari:**
- Farkli action space'leri olan ajanlar icin kullanilamaz (Cogen)
- Ajanlar arasi bireysel strateji farkliliklari yakalanamaz
- Tum ajanlarin ayni gozlem ve aksiyon uzayina sahip olmasi gerekir

**Neden Cogen'de Desteklenmiyor:**
Cogen ortaminda GT1, GT2, GT3 ve ST ajanlarinin action space'leri birbirinden farklıdır:
- GT1/GT2/GT3: 4 aksiyon (guc, 2x fan kontrol, isitici) -- 2 continuous + 2 discrete
- ST: 3 aksiyon (guc, buhar vanasi, sogutma kulesi) -- 2 continuous + 1 discrete(12)

Farkli boyut ve tipteki action space'ler, tek bir shared network'le temsil edilemez. Bu nedenle `--shared-policy` flagi Cogen icin bir hata mesaji verir:

```
error: --shared-policy not supported for cogen. Cogen agents have different action spaces.
```

### 3.3 Karsilastirma Tablosu

| Ozellik | Independent | Shared (MAPPO/MASAC) |
|---------|:-----------:|:--------------------:|
| Policy sayisi | N (ajan sayisi) | 1 |
| Parametre toplami | N x P | 1 x P |
| Sample efficiency | Dusuk | Yuksek |
| Ajan heterojenligi | Desteklenir | Desteklenmez |
| Cogen uyumu | Evet | Hayir |
| EVCharging uyumu | Evet | Evet |
| Building uyumu | Evet | Evet |
| Ogrenme hizi | Yavas (cok parametre) | Hizli (paylasiml deneyim) |
| Bireysel strateji | Var | Yok |

---

## 4. Ortam Detaylari

### 4.1 EVCharging (Elektrikli Arac Sarj)

**Kaynak:** ACN-Data / ACN-Sim (Caltech), GMM tabanli iz (trace) uretimi

| Ozellik | Deger |
|---------|-------|
| Ajan sayisi | 54 (her sarj istasyonu bir ajan) |
| Ajan kimlikleri | Caltech kampus sarj istasyonu ID'leri (ornegin `CA-148`, `CA-149`, ...) |
| Episod uzunlugu | 288 timestep (24 saat, 5 dakika cozunurluk) |
| Observation space (per ajan) | Flattened Box: timestep(1) + est_departures(54) + demands(54) + prev_moer(1) + forecasted_moer(36) = 146 boyut |
| Action space (per ajan) | Box(1), [0, 1] araligi (normalize edilmis pilot sinyal) |
| Reward | `(profit - carbon_cost - excess_charge) / num_agents` per agent |
| Reward paylasimi | Global reward / 54 (esit dagitim) |

**Observation Detayi:**
Her ajan, tum global bilgiyi gorur (full observability). Gozlem sunu icerir:
- `timestep`: Gun icindeki pozisyon [0, 1]
- `est_departures`: Tum 54 istasyon icin tahmini kalkis zamanlari
- `demands`: Tum 54 istasyon icin enerji talepleri
- `prev_moer`: Onceki Marginal Operating Emissions Rate
- `forecasted_moer`: 36 adimlik MOER tahmini

`periods_delay > 0` ayarlanirsa, ajanlar diger ajanlarin bilgilerini gecikmeli (delayed) olarak gorur, ancak kendi bilgilerini anlik alir. Projemizde `periods_delay=0` kullaniyoruz (tam gozlemlenebilirlik).

**Action Detayi:**
Her ajan tek bir skaler aksiyon uretir: [0, 1] araliginda normalize edilmis bir pilot sinyal. Bu sinyal, [0, 32] Amper araligina olceklenir. `project_action_in_env=True` ayari, toplam ag kisitlamalarinin (network constraints) ortam icinde projeksiyonla saglanmasini garanti eder.

**Reward Mekanizmasi:**
Tek ajanli ortamdan gelen global reward, ajan sayisina bolunur:
```python
rewards[agent] = reward / self.num_agents  # reward / 54
```

### 4.2 Building (Bina HVAC Kontrolu)

**Kaynak:** EnergyPlus RC termal modeli, ASHRAE 90.1-2019 prototip binalar

| Ozellik | Deger |
|---------|-------|
| Ajan sayisi | ~5 (OfficeSmall, klima bulunan zoneler) |
| Ajan kimlikleri | Tamsayilar (zone indeksleri, `ac_map[i] == 1` olan i degerleri) |
| Episod uzunlugu | 288 timestep (24 saat, 5 dakika cozunurluk) |
| Observation space (per ajan) | Box(n+4): zone_temps(n) + outdoor_temp + ground_temp + ghi + occupower |
| Action space (per ajan) | Box(1), [-1, 1] araligi (negatif=sogutma, pozitif=isitma) |
| Reward | Global reward (normalize edilmis, [-1, 0] araligi) |
| Reward paylasimi | Global reward dogrudan her ajana verilir (bolunmez) |

**Ajan Secimi:**
Ajanlar, `ac_map` dizisindeki `True` degerlerine sahip zonelerdir:
```python
self.possible_agents = np.nonzero(self.single_env.ac_map)[0].tolist()
```
`OfficeSmall` konfigurasyonunda `ac_map=1` (tum zonelerde klima var) kullanilir, dolayisiyla ajan sayisi binadaki zone sayisina esittir (yaklasik 5).

**Observation Detayi:**
Her ajan **ayni global gozlemi** alir. Bu, tum zonelerin sicakliklari ile dis ortam bilgilerini icerir. "Partial observability" uygulanmaz -- her ajan diger zonelerin durumunu da gorur.

**Action Detayi:**
Her ajan, kendi zonesindeki HVAC birimi icin [-1, 1] araliginda bir kontrol sinyali uretir:
- Negatif degerler: sogutma
- Pozitif degerler: isitma
- 0: HVAC kapali

Klimasi olmayan zoneler (`ac_map[i] == 0`) icin aksiyon sabit 0'dir ve bu zoneler ajan olarak tanimlanmaz.

**Reward Mekanizmasi:**
Building ortaminda reward bolunmez. Her ajan ayni global reward'u alir:
```python
rewards[agent] = reward  # Global reward, bolunmeden
```
Bu, EVCharging'den farkli bir yaklasimdir. Building'de ajan sayisi daha az oldugu icin (~5) ve reward zaten normalize edilmis oldugu icin [-1, 0] araliginda, bolunmeye gerek gorulmemistir.

### 4.3 Cogen (Kojenerasyon Santrali)

**Kaynak:** ONNX surrogate modeli (kojenerasyon santrali)

| Ozellik | Deger |
|---------|-------|
| Ajan sayisi | 4 (GT1, GT2, GT3, ST) |
| Ajan kimlikleri | `['GT1', 'GT2', 'GT3', 'ST']` |
| Episod uzunlugu | 96 timestep (1 gun) |
| Observation space (per ajan) | Flattened Box (tum ajanlar ayni) |
| Action space (per ajan) | Dict (ajan bazinda farkli, asagidaki tabloya bakiniz) |
| Reward | Ajan bazinda bireysel maliyet |
| Reward paylasimi | Bireysel (her ajan kendi maliyetinden sorumlu + ortaklasa non-delivery) |

**Ajan Aksiyon Dagilimi:**

| Ajan | Aksiyon Anahtarlari | Aciklama |
|------|---------------------|----------|
| GT1 | `GT1_PWR`, `GT1_PAC_FFU`, `GT1_EVC_FFU`, `HR1_HPIP_M_PROC` | Gaz turbini 1: guc, 2 fan, isitici |
| GT2 | `GT2_PWR`, `GT2_PAC_FFU`, `GT2_EVC_FFU`, `HR2_HPIP_M_PROC` | Gaz turbini 2: guc, 2 fan, isitici |
| GT3 | `GT3_PWR`, `GT3_PAC_FFU`, `GT3_EVC_FFU`, `HR3_HPIP_M_PROC` | Gaz turbini 3: guc, 2 fan, isitici |
| ST | `ST_PWR`, `IPPROC_M`, `CT_NrBays` | Buhar turbini: guc, buhar vanasi, sogutma kulesi bay sayisi |

**Observation Detayi:**
Tum ajanlar ayni flattened observation'i alir. Bu gozlem, orijinal CogenEnv'in Dict observation space'inin duzelenmis (flattened) halidir ve su bilesenlerden olusur:
- `Time`: Zaman bilgisi
- `TAMB`, `PAMB`, `RHAMB`: Cevre kosullari (sicaklik, basinc, nem)
- `Target_Power`, `Target_Steam`: Hedef guc ve buhar talepleri
- `Energy_Price`, `Gas_Price`: Enerji ve dogalgaz fiyatlari
- Her bir bilesen `forecast_horizon + 1` adimlik tahmin icerir

**Reward Mekanizmasi:**
Cogen ortaminda reward, **ajan bazinda bireyseldir** -- diger iki ortamdan temel farki budur:
```python
rewards[agent] = -(
    info['fuel_costs'][agent]          # Ajan'in yakit maliyeti
    + info['ramp_costs'][agent]        # Ajan'in rampa maliyeti
    + info['dyn_cv_costs'][agent]      # Ajan'in kisit ihlali maliyeti
    + info['non_delivery_cost'] / self.num_agents  # Ortaklasa: teslim edilemeyen enerji
)
```

Bu yaklasimda:
- Her ajan kendi yakit, rampa ve kisit ihlali maliyetlerinden birebir sorumludur
- `non_delivery_cost` (talep karsilanamama maliyeti) tum ajanlar arasinda esit bolunur
- Bu tasarim, her ajanin kendi birimini verimli calistirmasi icin dogru tesvikleri olusturur

### 4.4 Ortam Karsilastirma Tablosu

| Ozellik | EVCharging | Building | Cogen |
|---------|:----------:|:--------:|:-----:|
| Ajan sayisi | 54 | ~5 | 4 |
| Episod uzunlugu | 288 | 288 | 96 |
| Obs space tipi | Flattened Box | Box | Flattened Box |
| Action space tipi | Box(1) | Box(1) | Dict (mixed) |
| Homojen ajanlar | Evet | Evet | Hayir |
| Shared policy | Desteklenir | Desteklenir | Desteklenmez |
| Reward paylasimi | Global / N | Global (bolunmez) | Bireysel |
| Desteklenen algoritmalar | PPO, SAC, APPO, IMPALA, TD3 | PPO, SAC, APPO, IMPALA, TD3 | PPO, APPO, IMPALA |

---

## 5. Hyperparametreler

### 5.1 Temel Egitim Parametreleri

| Parametre | EVCharging | Building | Cogen | Aciklama |
|-----------|:----------:|:--------:|:-----:|----------|
| `num_iterations` | 32,000 | 32,000 | 750 | Toplam egitim iterasyonu |
| `num_workers` | 10 | 4 | 4 | Paralel rollout worker sayisi |
| `lr` | 3e-4 | 3e-4 | 3e-4 | Ogrenme hizi (learning rate) |
| `gamma` | 0.99 | 0.99 | 0.99 | Indirgeme faktoru (discount factor) |
| `seed` | 42 | 42 | 42 | Tekrarlanabilirlik icin rastgelelik tohumu |
| `checkpoint_freq` | 10 | 10 | 10 | Checkpoint kaydetme sikligi (iterasyon) |
| `fcnet_hiddens` | [64, 64] | [64, 64] | [64, 64] | Neural network mimarisi |

### 5.2 Parametre Aciklamalari

**`num_iterations` (Iterasyon Sayisi):**
Her bir iterasyonda su islemler gerceklesir:
1. Rollout worker'lar `rollout_fragment_length` adim deneyim toplar
2. Toplanan veriler `train_batch_size` olarak birlestirilir
3. Policy guncellenir (on-policy: `num_sgd_iter` epoch; off-policy: batch sampling)

EVCharging ve Building icin 32,000 iterasyon kullanilir. Bu yuksek sayi, 54 ajanli EVCharging'de yeterli sample sayisina ulasmak ve kararli policy'ler elde etmek icin gereklidir. Cogen icin 750 iterasyon yeterlidir cunku:
- Episod uzunlugu daha kisa (96 vs 288)
- Ajan sayisi az (4)
- ONNX model evaluasyonu pahalidir (computational cost)

**`num_workers` (Rollout Worker Sayisi):**
Rollout worker, ortam kopyasi uzerinde veri toplayan paralel bir surecdir. Her worker bagimsiz bir ortam instance'i calistirir.

- **EVCharging (10 worker):** 54 ajan x 288 timestep = iterasyon basina cok buyuk veri hacmi. 10 worker ile paralel veri toplama, throughput'u 10x arttirir.
- **Building (4 worker):** ~5 ajan x 288 timestep. Veri hacmi daha kucuk, 4 worker yeterlidir.
- **Cogen (4 worker):** 4 ajan x 96 timestep. ONNX model evaluasyonu CPU-yogun oldugu icin fazla worker, CPU contention yaratabilir.

**`train_batch_size` (Egitim Batch Boyutu):**
Bir iterasyonda policy guncellemesi icin kullanilacak toplam timestep sayisi.

| Ortam | Algoritma | `train_batch_size` | Aciklama |
|-------|-----------|:------------------:|----------|
| EVCharging | PPO/APPO | 2,880 | 10 worker x 288 fragment = 2,880 |
| EVCharging | SAC/TD3 | 4,000 | Off-policy, replay buffer'dan sample |
| EVCharging | IMPALA | 2,880 | 10 worker x 288 fragment = 2,880 |
| Building | PPO/APPO/IMPALA | 4,000 | 4 worker'dan biriktirilir |
| Building | SAC/TD3 | 4,000 | Off-policy, replay buffer'dan sample |
| Cogen | PPO/APPO/IMPALA | 4,000 | 4 worker'dan biriktirilir |

EVCharging'deki 2,880 degeri, `10 worker x 288 rollout_fragment = 2,880` hesabiyla dogal olarak ortaya cikar. Building ve Cogen'de 4,000 daha genis bir batch saglar.

**`rollout_fragment_length` (Rollout Fragment Uzunlugu):**
Her worker'in tek seferde toplayacagi timestep sayisi.

| Ortam | Deger | Anlami |
|-------|:-----:|--------|
| EVCharging | 288 | Tam 1 episod (24 saat) |
| Building | 288 | Tam 1 episod (24 saat) |
| Cogen | 200 | ~2 episod (96 timestep/episod) |
| SAC/TD3 | `"auto"` | RLlib otomatik belirler |

288, EVCharging ve Building icin tam bir episod uzunlugudur. Bu, her worker'in her seferinde tam bir episod deneyimi toplamasini garanti eder. Cogen'de 200 kullanilir; bu yaklasik 2 episod (2 x 96 = 192) karsiligi olup biraz fazladan veri icerir.

SAC ve TD3 icin `"auto"` ayari, RLlib'in replay buffer mekanizmasina uygun sekilde fragment uzunlugunu otomatik belirlemesini saglar.

### 5.3 PPO/APPO-Spesifik Parametreler

| Parametre | EVCharging | Building | Cogen | Aciklama |
|-----------|:----------:|:--------:|:-----:|----------|
| `sgd_minibatch_size` | 1,024 (PPO) | 128 | 128 | SGD minibatch boyutu |
| `num_sgd_iter` | 10 | 30 | 30 | Epoch sayisi (veri uzerinden kac kez gecirilecegi) |
| `lambda_` | 0.95 | 0.95 | 0.95 | GAE lambda (Generalized Advantage Estimation) |
| `clip_param` | 0.2 | 0.2 | 0.2 | PPO clipping parametresi |
| `entropy_coeff` | 0.005 | 0.01 | 0.01 | Entropi bonusu katsayisi |

**`sgd_minibatch_size`:**
PPO, toplanan batch'i daha kucuk minibatch'lere bolerek SGD uygular. EVCharging'de 1,024 kullanilir (54 ajan icin daha buyuk batch'ler kararli gradient saglar). Building ve Cogen'de 128 yeterlidir.

**Not:** APPO icin `sgd_minibatch_size` parametresi tanimlanmaz. APPO, veriyi asenkron olarak isler ve kendi batch mekanizmasini kullanir.

**`num_sgd_iter`:**
Her iterasyondaki toplanan veri uzerinden kac epoch gradient descent yapilacagini belirler. EVCharging'de 10 epoch, Building ve Cogen'de 30 epoch kullanilir. Daha kucuk ortamlar (daha az ajan) icin daha fazla epoch, toplanan veriden daha fazla ogrenme saglar.

**`lambda_` (GAE Lambda):**
Generalized Advantage Estimation'da bias-variance tradeoff'unu kontrol eder:
- `lambda_=0`: Yuksek bias, dusuk variance (TD(0))
- `lambda_=1`: Dusuk bias, yuksek variance (Monte Carlo)
- `lambda_=0.95`: Iyi bir denge noktasi

**`clip_param`:**
PPO'nun temel mekanizmasi. Policy oranini `[1 - 0.2, 1 + 0.2]` araliginda tutar. Bu, her guncelleme adiminda policy'nin cok fazla degismesini onler ve egitim kararliligi saglar.

**`entropy_coeff`:**
Entropi bonusu, policy'nin erken yakinsama (premature convergence) yapmasini onler. EVCharging'de 0.005 (54 ajan, zaten yuksek exploration), Building ve Cogen'de 0.01 (daha az ajan, daha fazla exploration ihtiyaci).

### 5.4 SAC-Spesifik Parametreler

| Parametre | EVCharging | Building | Aciklama |
|-----------|:----------:|:--------:|----------|
| `train_batch_size` | 4,000 | 4,000 | Replay buffer'dan sample boyutu |
| `n_step` | 288 | 288 | n-step return hesaplama uzunlugu |
| `lr` | 3e-4 | 3e-4 | Ogrenme hizi |
| `gamma` | 0.99 | 0.99 | Indirgeme faktoru |

**`n_step`:**
SAC'da n-step return, TD target hesaplamasinda n adim ileriye bakarak reward'lari toplar:
```
G_t = r_t + gamma * r_{t+1} + ... + gamma^{n-1} * r_{t+n-1} + gamma^n * V(s_{t+n})
```
`n_step=288` ayari, tam bir episod uzunlugunda return hesaplamasi yapar. Bu, uzun vadeli stratejik kararlarin (ornegin gece sarj planlamasi) etkisinin daha iyi yakalanmasini saglar.

### 5.5 IMPALA-Spesifik Parametreler

| Parametre | EVCharging | Building | Cogen | Aciklama |
|-----------|:----------:|:--------:|:-----:|----------|
| `vtrace` | True | True | True | V-trace importance sampling aktif |
| `vtrace_clip_rho_threshold` | 1.0 | 1.0 | 1.0 | V-trace rho kesme esigi |
| `vtrace_clip_pg_rho_threshold` | 1.0 | 1.0 | 1.0 | Policy gradient rho kesme esigi |
| `entropy_coeff` | 0.005 | 0.01 | 0.01 | Entropi bonusu katsayisi |

**V-trace parametreleri:**
Her iki kesme esigi de 1.0'dir. Bu, oldukca muhafazakar bir ayardir -- asiri buyuk importance weight'lerin gradient guncellemelerini bozmasini engeller. Buyuk deger (ornegin 5.0) daha agresif duzeltme yapar ancak variance'i arttirir.

### 5.6 TD3-Spesifik Parametreler

| Parametre | EVCharging | Building | Aciklama |
|-----------|:----------:|:--------:|----------|
| `train_batch_size` | 4,000 | 4,000 | Replay buffer'dan sample boyutu |
| `twin_q` | True | True | Ikiz Q-network kullanimi |
| `policy_delay` | 2 | 2 | Policy guncelleme gecikmesi |
| `lr` | 3e-4 | 3e-4 | Ogrenme hizi |
| `gamma` | 0.99 | 0.99 | Indirgeme faktoru |

**`twin_q=True`:** Iki ayri Q-network ile overestimation bias'i azaltilir.
**`policy_delay=2`:** Her 2 critic guncellemesinde 1 actor guncellemesi yapilir. Bu, actor'un daha kararli Q-value tahminleri uzerinden guncellenmesini saglar.

### 5.7 Network Mimarisi

Tum ortamlar ve algoritmalar icin ayni network mimarisi kullanilir:

```python
model = {"fcnet_hiddens": [64, 64]}
```

Bu, 2 katmanli, her katmanda 64 noronlu bir fully-connected neural network tanimlar. ReLU aktivasyon fonksiyonu varsayilan olarak kullanilir. Bu nispeten kucuk bir network'tur; MARL'de cok buyuk networkler genellikle overfitting ve yava ogrenme sorunlarina neden olur.

**Not:** SAC ve TD3 icin `model` parametresi acikca ayarlanmaz; RLlib varsayilan mimarisini kullanir (genellikle [256, 256]).

---

## 6. CPU vs GPU

### 6.1 GPU Otomatik Tespiti

Tum MARL scriptleri (hem unified `marl_training.py` hem de ortam bazli `marl_tr_*.py`) ayni GPU tespit mekanizmasini kullanir:

```python
if args.num_gpus is not None:
    num_gpus = args.num_gpus
else:
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except ImportError:
        num_gpus = 0
```

Bu mekanizma su sirada calisir:
1. Kullanici `--num-gpus` belirtmis mi? -> Evet: o degeri kullan
2. PyTorch import edilebiliyor mu? -> Evet: `torch.cuda.device_count()` ile GPU say
3. PyTorch yok veya GPU yok -> `num_gpus = 0` (CPU modu)

### 6.2 Kaynak Dagitimi

```python
.resources(num_gpus=min(1, num_gpus), num_gpus_per_worker=0)
```

Bu konfigürasyon su anlama gelir:
- **Learner (ana egitim sureci):** En fazla 1 GPU kullanir. GPU varsa neural network hesaplamalari (forward/backward pass) GPU'da yapilir.
- **Rollout worker'lar:** GPU kullanmaz (`num_gpus_per_worker=0`). Worker'lar sadece ortam simulasyonu ve inference yapar; bunlar genellikle CPU'da daha verimlidir.

### 6.3 CPU-Only Mod

GPU bulunmadiginda veya `--num-gpus 0` belirtildiginde:
```
[MARL] No GPU detected. Running in CPU-only mode.
```

Bu durumda tum islemler (learner + workers) CPU uzerinde gerceklesir. ARC HPC'de 48 CPU core tahsis edildigi icin, CPU-only mod bile oldukca verimlidir.

### 6.4 ARC HPC'de Tipik Kullanim

SLURM scriptlerinde GPU tahsis edilmez (`--gres=gpu` yok):
- Partition: `normal_q` (CPU-only kuyruk)
- CPU: 48 core
- RAM: 40 GB
- Zaman limiti: 3 gun

MARL egitimi icin tipik bir SLURM scripti:
```bash
#!/bin/bash -l
#SBATCH --partition=normal_q
#SBATCH --constraint=intel&avx512
#SBATCH --cpus-per-task=48
#SBATCH --mem=40G
#SBATCH -t 3-00:00:00

conda activate stdrl_train
python marl_training.py --env evcharging --algo APPO --num-gpus 0
```

**Not:** MARL scriptleri `--num-gpus 0` ile acikca CPU-only moda zorlanabilir. ARC'de A100/H200 GPU'lar mevcut olsa da, MARL egitiminde GPU kullanimi su nedenlerle tercih edilmeyebilir:
- Worker sayisi yuksek (10), bunlar zaten CPU'da calisir
- Network boyutu kucuk ([64, 64]), GPU overhead'i faydasini asabilir
- GPU kuyruk bekleme suresi CPU kuyrugunden fazla

---

## 7. Output Yapisi

### 7.1 Log Dizin Formati

Tum MARL egitimleri su formatta bir dizin olusturur:

```
./logs_marl_train/{env}_{algo}{_SHARED}/{timestamp}_NOISE_0_ACT_0_ENV_0/
```

Ornekler:
```
./logs_marl_train/evcharging_APPO/2026-03-08_14-30-00_NOISE_0_ACT_0_ENV_0/
./logs_marl_train/building_SAC_SHARED/2026-03-08_15-00-00_NOISE_0_ACT_0_ENV_0/
./logs_marl_train/cogen_PPO/2026-03-08_16-00-00_NOISE_0_ACT_0_ENV_0/
```

- `{env}`: Ortam ismi (`evcharging`, `building`, `cogen`)
- `{algo}`: Algoritma ismi (`PPO`, `SAC`, `APPO`, `IMPALA`, `TD3`)
- `_SHARED`: Sadece `--shared-policy` kullanildiginda eklenir
- `NOISE_0_ACT_0_ENV_0`: MARL'de gurultu destegi bulunmadigindan sabit 0 (stdrl ile uyumlu format)

### 7.2 Dizin Icerigi

```
{log_dir}/
  +-- config.json          # Calisma konfigürasyonu
  +-- metrics.csv          # Iterasyon bazinda metrikler
  +-- checkpoints/         # Model checkpoint'leri
      +-- checkpoint_iter_10/
      +-- checkpoint_iter_20/
      +-- ...
```

### 7.3 config.json

Her run'in tam konfigürasyonunu saklar. Ornek:

```json
{
  "env": "evcharging",
  "algo": "APPO",
  "shared_policy": false,
  "policy_mode": "independent",
  "seed": 42,
  "num_iterations": 32000,
  "num_workers": 10,
  "num_gpus": 0,
  "lr": 0.0003,
  "checkpoint_freq": 10,
  "timestamp": "2026-03-08_14-30-00"
}
```

Cogen icin ek olarak `"renewables_magnitude": 300` alani bulunur.

### 7.4 metrics.csv

Her 10 iterasyonda ve egitim sonunda kaydedilir. Sutunlar:

| Sutun | Tip | Aciklama |
|-------|-----|----------|
| `iteration` | int | Iterasyon numarasi (1'den baslar) |
| `mean_reward` | float | Episod basina ortalama toplam reward |
| `mean_length` | float | Episod basina ortalama timestep sayisi |
| `loss` | float | Ilk policy'nin toplam kayip degeri (NaN olabilir) |
| `elapsed_seconds` | float | Egitim baslangicindan itibaren gecen sure (saniye) |

**Not:** `mean_reward` tum ajanlarin toplam reward'ini temsil eder (RLlib raporlamasi). `loss` degeri her zaman mevcut olmayabilir; RLlib'in icinindeki `learner_stats` yapisina baglidir.

Kayit mekanizmasi:
```python
# Her 10 iterasyonda veya egitim sonunda kaydet
if (i + 1) % 10 == 0 or (i + 1) == args.num_iterations:
    pd.DataFrame(metrics_history).to_csv(metrics_path, index=False)
```

Ayrica `try/finally` blogu ile, egitim herhangi bir nedenle kesildiginde bile mevcut metrikler kaydedilir.

### 7.5 Checkpoint'ler

Checkpoint'ler, her `checkpoint_freq` iterasyonda kaydedilir (varsayilan: 10):

```python
if (i + 1) % args.checkpoint_freq == 0:
    trainer.save(checkpoint_dir + f"/checkpoint_iter_{i+1}")
```

Her checkpoint, RLlib'in tam model durumunu icerir:
- Tum policy'lerin agirliklari
- Optimizer durumlari
- Replay buffer (off-policy algoritmalar icin)
- Konfigürasyon

Checkpoint'ler `trainer.restore(path)` ile yuklenebilir.

### 7.6 Konsol Ciktisi

Egitim sirasinda her 10 iterasyonda ve ilk iterasyonda bilgi yazdirilir:

```
[Iter      1/32000] reward=     -5.23  length=   288.0  loss=    0.1234  elapsed=    0.5min
[Iter     10/32000] reward=     -4.87  length=   288.0  loss=    0.0987  elapsed=    5.2min
[Iter     20/32000] reward=     -4.52  length=   288.0  loss=    0.0765  elapsed=   10.1min
```

Egitim tamamlandiginda veya kesildiginde ozet rapor yazdirilir:

```
======================================================================
TRAINING COMPLETE - EVCharging MARL (APPO, independent)
======================================================================
  Iterations completed: 32000/32000
  Training duration:    1440.0 minutes (24.0 hours)
  Final mean reward:    -2.15
  Final mean length:    288.0
  Metrics saved to:     ./logs_marl_train/.../metrics.csv
  Checkpoints in:       ./logs_marl_train/.../checkpoints
======================================================================
```

---

## 8. CLI Kullanimi

### 8.1 Unified Script (marl_training.py)

Tum ortamlari tek bir script uzerinden calistirmak icin:

```bash
# Varsayilanlarla calistirma
python marl_training.py --env evcharging
python marl_training.py --env building
python marl_training.py --env cogen

# Algoritma secimi
python marl_training.py --env evcharging --algo PPO
python marl_training.py --env evcharging --algo SAC
python marl_training.py --env evcharging --algo APPO
python marl_training.py --env evcharging --algo IMPALA
python marl_training.py --env evcharging --algo TD3

# Shared policy (MAPPO/MASAC)
python marl_training.py --env evcharging --algo PPO --shared-policy
python marl_training.py --env building --algo SAC --shared-policy

# Cogen ile ozel parametreler
python marl_training.py --env cogen --algo PPO --rm 300

# Kaynak kontrolu
python marl_training.py --env evcharging --algo APPO --num-gpus 0 --num-workers 10

# Ogrenme hizi ayarlama
python marl_training.py --env building --algo SAC --lr 1e-4
```

### 8.2 Ortam-Bazli Scriptler

Her ortam icin ayri script de mevcuttur:

```bash
# EVCharging
python marl_tr_ev.py --algo APPO --seed 42 --num-iterations 32000
python marl_tr_ev.py --algo PPO --shared-policy
python marl_tr_ev.py --algo TD3 --num-workers 10

# Building
python marl_tr_bu.py --algo SAC --seed 42 --num-iterations 32000
python marl_tr_bu.py --algo PPO --shared-policy
python marl_tr_bu.py --algo IMPALA --num-workers 4

# Cogen
python marl_tr_co.py --algo PPO --seed 42 --num-iterations 750
python marl_tr_co.py --algo APPO --rm 300
python marl_tr_co.py --algo IMPALA --num-workers 4
```

### 8.3 Tam Arguman Listesi

| Arguman | Tip | Varsayilan | Ortam | Aciklama |
|---------|-----|:----------:|:-----:|----------|
| `--env` | str | _(zorunlu)_ | Hepsi | Ortam secimi: `evcharging`, `building`, `cogen` |
| `--algo` | str | Ortam-bazli* | Hepsi | Algoritma secimi |
| `--seed` | int | 42 | Hepsi | Rastgelelik tohumu |
| `--num-iterations` | int | Ortam-bazli** | Hepsi | Toplam egitim iterasyonu |
| `--num-workers` | int | Ortam-bazli*** | Hepsi | Rollout worker sayisi |
| `--checkpoint-freq` | int | 10 | Hepsi | Checkpoint kayit sikligi |
| `--num-gpus` | int | auto-detect | Hepsi | GPU sayisi (0 = CPU-only) |
| `--lr` | float | 3e-4 | Hepsi | Ogrenme hizi |
| `--rm` | int | 300 | Cogen | Yenilenebilir enerji buyuklugu |
| `--shared-policy` | flag | False | EV, Building | Tek paylasimli policy kullan |

*Varsayilan algoritmalar: EVCharging=APPO, Building=SAC, Cogen=PPO
**Varsayilan iterasyonlar: EVCharging=32,000, Building=32,000, Cogen=750
***Varsayilan worker'lar: EVCharging=10, Building=4, Cogen=4

### 8.4 Ortam Bazinda Desteklenen Algoritmalar

| Ortam | Desteklenen Algoritmalar |
|-------|:------------------------:|
| EVCharging | PPO, SAC, APPO, IMPALA, TD3 |
| Building | PPO, SAC, APPO, IMPALA, TD3 |
| Cogen | PPO, APPO, IMPALA |

Desteklenmeyen bir kombinasyon seciliginde (ornegin `--env cogen --algo SAC`):
```
error: Algorithm 'SAC' not supported for cogen. Choose from: ['PPO', 'APPO', 'IMPALA']
```

---

## 9. Deneysel Plan

### 9.1 Toplam Run Sayisi

Paper icin planlanan MARL deneyleri su 23 run'dan olusmaktadir:

**EVCharging (13 run):**

| # | Algoritma | Policy Modu | Iterasyon |
|:-:|-----------|:-----------:|:---------:|
| 1 | PPO | Independent | 32,000 |
| 2 | PPO | Shared (MAPPO) | 32,000 |
| 3 | SAC | Independent | 32,000 |
| 4 | SAC | Shared (MASAC) | 32,000 |
| 5 | APPO | Independent | 32,000 |
| 6 | APPO | Shared | 32,000 |
| 7 | IMPALA | Independent | 32,000 |
| 8 | IMPALA | Shared | 32,000 |
| 9 | TD3 | Independent | 32,000 |
| 10 | TD3 | Shared | 32,000 |

**Building (10 run):**

| # | Algoritma | Policy Modu | Iterasyon |
|:-:|-----------|:-----------:|:---------:|
| 11 | PPO | Independent | 32,000 |
| 12 | PPO | Shared (MAPPO) | 32,000 |
| 13 | SAC | Independent | 32,000 |
| 14 | SAC | Shared (MASAC) | 32,000 |
| 15 | APPO | Independent | 32,000 |
| 16 | APPO | Shared | 32,000 |
| 17 | IMPALA | Independent | 32,000 |
| 18 | IMPALA | Shared | 32,000 |
| 19 | TD3 | Independent | 32,000 |
| 20 | TD3 | Shared | 32,000 |

**Cogen (3 run):**

| # | Algoritma | Policy Modu | Iterasyon |
|:-:|-----------|:-----------:|:---------:|
| 21 | PPO | Independent | 100 |
| 22 | APPO | Independent | 100 |
| 23 | IMPALA | Independent | 100 |

**Toplam:** 10 (EV) + 10 (Building) + 3 (Cogen) = **23 run**

### 9.2 Planlanan Karsilastirmalar

**Karsilastirma 1: Algoritma Performansi (ayni ortam, ayni policy modu)**
- EVCharging Independent: PPO vs SAC vs APPO vs IMPALA vs TD3
- Building Independent: PPO vs SAC vs APPO vs IMPALA vs TD3
- Cogen Independent: PPO vs APPO vs IMPALA

Bu karsilastirma, her ortam icin hangi algoritmanin en iyi performansi gosterdigini ortaya koyar. Beklenti: off-policy algoritmalar (SAC, TD3) sample efficiency'de on-policy'lerden (PPO, APPO) daha iyi olabilir, ancak APPO'nun throughput avantaji uzun egitimde bunu kapatabilir.

**Karsilastirma 2: Independent vs Shared Policy (ayni algoritma, ayni ortam)**
- EVCharging: PPO_Indep vs MAPPO, SAC_Indep vs MASAC, APPO_Indep vs APPO_Shared, ...
- Building: Ayni karsilastirmalar

Bu karsilastirma, parameter sharing'in etkisini olcer. Beklenti: homojen ajanlar icin (EVCharging, Building) shared policy daha iyi sample efficiency gosterir. Ancak cok ajanli (54) ortamlarda independent policy'ler bireysel uzmanlasmaya izin verdigi icin nihai performansta yarisabilir.

**Karsilastirma 3: Single-Agent vs Multi-Agent (ayni ortam, ayni algoritma)**
- EVCharging: SB3-PPO (single) vs RLlib-PPO (MARL Independent) vs RLlib-PPO (MAPPO)
- Building: SB3-SAC (single) vs RLlib-SAC (MARL Independent) vs RLlib-SAC (MASAC)
- Cogen: SB3-PPO (single) vs RLlib-PPO (MARL Independent)

Bu, tezin en temel karsilastirmasidir. Ayni algoritmayy kullanan single-agent ve multi-agent yaklasimlarinin reward, ogrenme hizi ve kararlilik acisindan karsilastirmasi.

**Karsilastirma 4: Olceklenebilirlik Analizi**
- EVCharging (54 ajan) vs Building (~5 ajan) vs Cogen (4 ajan)
- Ajan sayisi arttikcca MARL performansi nasil degisiyor?
- Independent policy ile 4 vs 5 vs 54 ajani yonetmenin zorlugu

### 9.3 Beklenen Sonuclar ve Analizler

**Hipotez 1:** Homojen ortamlar (EVCharging, Building) icin shared policy, independent policy'den daha hizli yakinsar. Cunku 54 ajanin deneyimi tek bir network'te birlesir, sample efficiency cok yuksektir.

**Hipotez 2:** Cogen'de MARL, single-agent'a kiyasla ajan bazinda daha iyi uzmanlasmis stratejiler olusturur. Her ajan (GT1, GT2, GT3, ST) kendi maliyet fonksiyonundan sorumlu oldugu icin, MARL bireysel optimizasyona olanak tanir.

**Hipotez 3:** 54 ajanli EVCharging ortaminda independent policy scalability problemi gosterir: cok fazla parametre, yavas ogrenme, yuksek variance. Shared policy bu problemi cozer.

**Hipotez 4:** Off-policy algoritmalar (SAC, TD3), 32,000 iterasyonluk uzun egitimlerde on-policy algoritmalardan (PPO, APPO, IMPALA) daha yuksek nihai performansa ulasir. Ancak APPO, asenkron veri toplama sayesinde wall-clock suresi acisindan daha verimli olabilir.

### 9.4 Analiz Metrikleri

Her run icin su metrikler degerlendirilecektir:

| Metrik | Aciklama | Kaynak |
|--------|----------|--------|
| Final Mean Reward | Son 100 iterasyonun ortalama reward'u | `metrics.csv` |
| Learning Speed | Belirli bir reward esigine ulasma suresi | `metrics.csv` (iteration + elapsed_seconds) |
| Training Stability | Reward egrisinin varyansi (smoothed) | `metrics.csv` (mean_reward std) |
| Wall-Clock Time | Toplam egitim suresi | `metrics.csv` (elapsed_seconds) |
| Sample Efficiency | Belirli bir reward icin gereken toplam timestep | Iterasyon x train_batch_size |

### 9.5 Grafik Uretimi

Metrikler, `ttp/marl_plot.py` scriptiyle gorsellestirilir:

```bash
# EVCharging tum algoritmalar
python ttp/marl_plot.py --env evcharging --t_steps 32000 --w_size 500

# Building otomatik Y ekseni
python ttp/marl_plot.py --env building --w_size 400 --auto_ylim --pdf

# Cogen kisa egitim
python ttp/marl_plot.py --env cogen --t_steps 750 --w_size 20
```

Grafikler `./graphs/C_MARL/` dizinine kaydedilir ve su formatta isimlendirilir:
```
{timestamp}_{env}_MARL_{algo1}_{algo2}_...{algoN}.png
```

EMA (Exponential Moving Average) smoothing ile gurultulu ogrenme egrileri yumusatilir. `w_size` parametresi smoothing penceresini kontrol eder. Ayrica her egri etrafinda `mean +/- 0.5 * std` bandini gosteren dolgu alani cizdirilerek varyans gorsellestirilir.

---

## Ekler

### A. Script Dosya Haritalari

| Dosya | Tam Yol | Islem |
|-------|---------|-------|
| Unified MARL | `marl_training.py` | Tum ortamlar, tum algoritmalar |
| EVCharging MARL | `marl_tr_ev.py` | Sadece EVCharging |
| Building MARL | `marl_tr_bu.py` | Sadece Building |
| Cogen MARL | `marl_tr_co.py` | Sadece Cogen |
| EV Multi-Agent Env | `envs/evcharging/multiagent_env.py` | PettingZoo ParallelEnv |
| Building Multi-Agent Env | `envs/building/multiagent_env.py` | PettingZoo ParallelEnv |
| Cogen Multi-Agent Env | `envs/cogen/multiagent_env.py` | PettingZoo ParallelEnv |
| MARL Plot | `ttp/marl_plot.py` | Ogrenme egrisi cizdirme |

### B. Bagimliliklar

```
ray[rllib]          # Dagitik RL egitim framework'u
pettingzoo          # Multi-agent ortam API'si
torch               # Neural network backend
gymnasium           # Ortam API standardi
numpy, pandas       # Veri isleme
```

### C. Bilinen Sinirlamalar ve Gelecek Calisma

1. **Noise destegi yok:** MARL scriptleri su anda observation/action/environment noise desteklememektedir. Dizin formati `NOISE_0_ACT_0_ENV_0` olarak sabittir. Gelecekte noise-robust MARL deneyleri icin bu destek eklenebilir.

2. **Evaluation callback yok:** stdrl scriptlerinden farkli olarak, MARL egitiminde ayri bir evaluation ortami kullanilmamaktadir. `mean_reward` metrigi, egitim ortamindaki episodlardan gelmektedir.

3. **VecNormalize destegi yok:** RLlib kendi observation normalizasyonunu yapmaktadir; SB3'un VecNormalize wrapper'ina esdeger bir mekanizma kullanilmamaktadir.

4. **Cogen icin off-policy algoritmalar:** SAC ve TD3, Dict action space'ler icin uyumsuz oldugu icin kullanilamamaktadir. Gelecekte, Cogen multiagent ortaminin Box action space'e duzelenmis (flattened) bir versiyonu ile off-policy MARL destegi eklenebilir (mevcut `envs/cogen/env_marl.py` dosyasinda bu yonde bir calişma mevcuttur).

5. **Communication mekanizmasi yok:** Ajanlar arasi acik iletisim (message passing) desteklenmemektedir. Ajanlar yalnizca paylasilam reward ve ortak observation araciligiyla dolayli etkilesim kurmaktadir.
