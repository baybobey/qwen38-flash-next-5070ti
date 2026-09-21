# qwen38-flash-next-5070ti

**Türkçe** | [English](README.md)

**Qwen3.8-Flash-Next EXL3 4.05bpw** modelini **1 veya 2 adet 16 GB GPU** üzerinde
çalıştırmak için kendi kendine yeten (standalone) kurulum betikleri ve ayarlanmış
sunum yapılandırmaları — 2x RTX 5070 Ti üzerinde ayarlandı.
[turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) ve
[theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI) kullanır.

İhtiyaç duyduğu her şeyi (motor, sunucu, model) indirir ve 262144 token bağlamlı
(context) çalışan bir yapılandırma yazar. Bu depoda hiçbir gizli bilgi yoktur:
API anahtarı kurulum sırasında sizin makinenizde üretilir.

```
./install-2gpu.sh          # veya ./install-1gpu.sh
./serve-2gpu.sh            # veya ./serve-1gpu.sh
./verify.sh                # ikinci bir terminalde
```

## Donanım gereksinimleri

| | |
|---|---|
| GPU | 1 veya 2 adet NVIDIA, 16 GB sınıfı (Blackwell sm_120 üzerinde ayarlandı; Ampere ve sonrası kutudan çıktığı gibi çalışır — aşağıya bakın; en az 14 GB, VRAM arttıkça daha iyi) |
| Sürücü | NVIDIA >= 570 (CUDA 12.8 çalışma zamanı; kurulum betiği kontrol eder) |
| Sistem RAM'i | 128 GB önerilir. Ağırlıklar ~100 GiB tutar ve bellekte kalır (uzmanlar CPU'da + 36 GiB n-gram tablosu RAM'de). Kurulum betiği 90 GiB altında devam etmez |
| Disk | ~120 GB boş alan (model 100.1 GiB + venv ~8 GB) |
| İşletim sistemi / Python | Linux x86_64, Python 3.10–3.13 (hazır derlenmiş motor wheel'i; 3.14 JIT ile çalışır, o durumda CUDA toolkit kurulu olmalı) |
| PCIe | Geniş bant daha hızlı: decode/prefill CPU uzmanlarını PCIe üzerinden aktarır, x8/x16 fark eder |
| CPU | 16+ çekirdek önerilir (CPU uzmanları ana makinede çalışır; yapılandırmalar ~16 çekirdek varsayar) |

**Hangi GPU'lar çalışır:** hazır derlenmiş motor wheel'i **sm_80, sm_86, sm_89, sm_90, sm_100
ve sm_120** için derlenmiş çekirdekler içerir — Ampere, Ada, Hopper ve Blackwell (30/40/50
serisi GeForce kartlar, A100/A6000, H100, B200). Turing (sm_75) ve Volta (sm_70) bu wheel'den
çekirdek alamaz; kurulum betiği bunu daha hiçbir şey indirmeden algılar, JIT derlemesine geçer
(ilk içe aktarmada derlenir, yani CUDA toolkit gerekir) ve size söyler. Yalnızca NVIDIA —
AMD/Intel/CPU yolu yok. Karışık nesil ikililer (örn. bir 4090 ile bir 5070 Ti) her iki kart da
sm_80 veya üstü olduğu sürece sorunsuz çalışır. `./install-2gpu.sh --dry-run` hiçbir şeyi
değiştirmeden planı — hangi motor wheel'ini seçtiği dahil — yazdırır.

## Kurulum betiği ne yapar

Bu dizine (ve varsayılan olarak `$HOME/models/Qwen3.8-Flash-Next` altına model):

- `.venv/` — torch `2.9.0+cu128`, exllamav3 `1.5.1` (Python sürümünüze uyan hazır
  derlenmiş wheel; uygun wheel yoksa JIT wheel + CUDA toolkit), TabbyAPI bağımlılıkları
  ve `hf` CLI
- `tabbyAPI/` — `7208273` commit'ine sabitlenmiş klon
- `tabbyAPI/config.yml` — `configs/config-<N>gpu.yml` dosyasından yazılır; `models/<ad>`
  sembolik bağlantısı ve zorunlu (force) sampler preset'i kurulur
- `tabbyAPI/api_tokens.yml` — rastgele üretilmiş `api_key` ve `admin_key`, `chmod 600`, hiçbir
  zaman ekrana basılmaz ve git'e girmez
- `tabbyAPI/templates/froggeric-qwen38-v225.jinja` — froggeric'in düzeltilmiş Qwen sohbet
  şablonu (v22.5; dosya adında nokta yok, çünkü tabbyAPI adları `Path.with_suffix` ile
  çözüyor); yazarın kendi deposundan sabitlenmiş bir revizyondan indirilir ve sha256 ile
  doğrulanır, yapılandırmadaki `prompt_template:` onu seçer (aşağıdaki “Sohbet şablonu”)
- model — `turboderp/Qwen3.8-Flash-Next-exl3`, `4.05bpw_h6_ng6` dalı, `55a732e0`
  commit'ine sabitli (100.1 GiB, kaldığı yerden devam eder; indirme sonrası HF
  manifestosuyla doğrulanır)

Seçenekler: `--skip-model` (ağırlıklar zaten varsa), `--model-dir D`, `--venv D`,
`--start` (kurulum bitince sunucuyu başlat), `--dry-run` (planı yazdır, hiçbir şeyi
değiştirme), `--latest-template` (sabitlenmiş revizyon yerine upstream'in o an sunduğu
şablon dosyasını indir), `--official-template` (Qwen'in kendi sohbet şablonunu kullan), `-y`.
Tam liste: `./install.sh --help`.

Kurulum betiğini tekrar çalıştırmak güvenlidir (idempotent): venv, klon ve yapılandırma
yenilenir; model ve API anahtarı korunur.

## Sohbet şablonu (chat template)

Yapılandırmalar, modelin içindeki şablon yerine **froggeric'in düzeltilmiş Qwen sohbet
şablonunu** kullanır. `install.sh` onu yazarın kendi deposundan sabitlenmiş bir revizyondan
indirir ve sha256'sını doğrular — bu depoyla **paketlenmez**, böylece hem emeğin sahibi hem de
güncel kopya yazarın kendisinde kalır:

- Kaynak: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) (Apache-2.0)
- `tabbyAPI/templates/froggeric-qwen38-v225.jinja` olarak kurulur ve `config.yml` içindeki
  `prompt_template: froggeric-qwen38-v225` ile seçilir; açılış günlüğü
  `Using template "froggeric-qwen38-v225" for chat completions.` satırını yazar

Qwen'in kendi şablonunu mu tercih ediyorsunuz? `--official-template` ile kurun,
`tabbyAPI/config.yml` içindeki `prompt_template:` değerini boşaltın ya da çalışma anında
değiştirin: `POST /v1/template/switch {"prompt_template_name": "…"}`.

Şablon revizyon + sha256 ile sabitlenmiştir, böylece bu deponun her kurulumu tekrarlanabilir
olur. Upstream daha yeni bir revizyon yayınlarsa kurulum betiği bunu (ve hangi sabitlemeyi
kullandığını) söyler; `--latest-template` ise upstream'in o an sunduğu dosyayı indirip aldığı
sürüm damgasını ve sha256'yı yazdırır — elinizde tutacaksanız bunu kendiniz sabitleyebilirsiniz.

## Çalıştırma

```
./serve-2gpu.sh                    # 2 GPU: autosplit + MTP draft
GPU_ORDER=1,0 ./serve-2gpu.sh      # kart sırasını seç (yazarın makinesi 1,0 kullandı)
GPU_ID=1 ./serve-1gpu.sh           # tek GPU'yu belirli bir kartta çalıştır
MODEL_DIR=/data/models/qwen ./serve-2gpu.sh
```

Durdurmak için Ctrl-C (`pkill -f "main.py"` de çalışır; düzgün boşaltma ~20 sn sürer).
Sunucu `0.0.0.0:8001` adresini dinler ve `tabbyAPI/api_tokens.yml` içindeki anahtarı
ister:

```
grep api_key tabbyAPI/api_tokens.yml
```

OpenAI uyumlu her istemci: `base_url = http://127.0.0.1:8001/v1`,
`api_key = <bu değer>`, `model = Qwen3.8-Flash-Next-EXL3-405`.
`./verify.sh` sağlık durumunu, `n_ctx` değerini kontrol eder ve bir deneme üretimi yapar.

Aynı dosyada bir de `admin_key` bulunur (32 hex karakter; sohbet için gerekmez). Yönetim uç
noktalarını korur — `/v1/model/load`, `/v1/model/unload`, `/v1/download`, LoRA ve embedding
yükleme/boşaltma çiftleri ile `/v1/template/*` ve `/v1/sampling/override/*`; okumak için
`grep admin_key tabbyAPI/api_tokens.yml`.

## Ölçülen performans

Yazarın makinesi: 2x RTX 5070 Ti (her biri 16 GB, x8 + x4 PCIe), Ryzen 9 9950X3D2,
128 GiB RAM. Sunucu tarafında zorlanan sampler: **temp 1.0 / top-k 20 / top-p 0.95 /
repetition 1.0**, xhigh reasoning, 262144 bağlam. Aşağıdaki sayılar o makineden —
sizinkiler PCIe genişliği, RAM hızı ve CPU çekirdek sayısına göre farklı çıkabilir.

| Düzen | Yapılandırma | Prefill (33K soğuk) | Decode |
|---|---|---|---|
| 2 GPU | `config-2gpu.yml` — CPU uzmanları 416, MTP dinamik draft (8 token üst sınırı), chunk 2048 | ~825 t/s | kararlılık turları 32.7–34.3 t/s, sıcak tekrar 42.9 t/s |
| 1 GPU | `config-1gpu.yml` — CPU uzmanları 496, draft kapalı, chunk 3072 | ~943 t/s | 30.5–31.3 t/s |

2 GPU düzeni prefill'de kazanır ve uzun promptlarda daha iyi hizmet verir; 1 GPU düzeni
kısa bağlamda decode'da daha hızlıdır ve ikinci kartı boş bırakır. Decode, CPU
uzmanlarına bağlıdır (CPU-expert bound): `cpu_moe_threads` ≈ fiziksel çekirdek / 2
kullanın ve yanında ikinci bir model çalıştırmayın.

## Ayar notları (gerçekten fark yaratan düğmeler)

- `cpu_moe_split_experts` — her katmanda sistem RAM'inde kalan uzman sayısı (2 GPU'da
  512'nin 416'sı, 1 GPU'da 496'sı). Düşürmek = daha fazla VRAM, genelde daha hızlı;
  OOM alırsanız yükseltin.
- `cpu_moe_threads` — 16 çekirdekli makinede 16 (2 GPU) / 14 (1 GPU). Fazla iş parçacığı
  zarar verir.
- `chunk_size` — burada 2048 (2 GPU) / 3072 (1 GPU); büyütmek prefill'i hızlandırır,
  ta ki PLE workspace ayırması prefill ortasında OOM alana kadar.
- `autosplit_reserve` — autosplit'in boş bıraktığı pay (kart başına MB). 2 GPU'da kart
  başına 1024 MB, uzun prefill'lerde 40 MiB'lik PLE workspace'ini güvende tutar. 1 GPU'da
  küçük tutun (96 MB): sığdırma zaten sıkı ve ölçülen tek-GPU sayıları bu varsayılanla
  alındı.
- `draft_mode: mtp` (yalnızca 2 GPU) + `dynamic_draft: true` ve 8 token üst sınırı —
  modelin kendi MTP başlığı; 2 GPU'da gerçek kazanç, 1 GPU'da ölçülebilir bir şey yok.
- `cache_mode: 6,6` — 262144 bağlamda 6-bit K/V. Bağlamı düşürmeyin.
- **Zorunlu sampler preset'i** (`sampler_overrides/qwen38_flash_next_thinking.yml`)
  isteğe bağlı değildir: onsuz, sampler parametresi göndermeyen istemciler
  `top_k 0 / top_p 1.0` ile çalışır (kırpılmamış = uzun bağlamda "token çorbası").
  Açılış günlüğünde "No sampler override preset is configured" uyarısı OLMAMALIDIR.
- `tool_format: qwen3_coder` + `reasoning: true` — modelin araç çağırma biçimi ve
  `reasoning_content` değerini `content`'ten ayrı döndüren reasoning ayrıştırıcısı.

## Sorun giderme

- **Yüklemede "Insufficient VRAM"** — `cpu_moe_split_experts` değerini 8–16 düşürün ya da
  `cache_size` değerini azaltın. Bu yığında bağlamı 262144'ün altına indirmeyin.
- **Prefill ortasında küçük bir ayırma hatasıyla ölüyor** — bu PLE workspace'idir;
  `autosplit_reserve` değerini yükseltin (örn. 512) ya da daha küçük bir `chunk_size`
  kabul edin.
- **İlk açılış dakikalar sürüyor / çekirdekler yeniden derleniyor** — exllamav3 ilk içe
  aktarmada CUDA çekirdeklerini JIT ile derler (kurulum betiği bunu önceden ısıtır).
  Derleme yarıda kalırsa `~/.cache/torch_extensions` dizinini silip yeniden başlatın.
- **Saçmalama veya dil kayması** — açılış günlüğünde yukarıdaki sampler uyarısını ve
  istemcinizin `api_tokens.yml` kimlik doğrulamasını gerçekten gönderdiğini kontrol edin.
- **İndirme yarıda kaldı** — kurulum betiğini yeniden çalıştırın; `hf download` kaldığı
  yerden devam eder.

## Emeği geçenler ve lisanslar

- [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) — MIT
- [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI) — **AGPL-3.0**. Bu depo
  onu kurulum sırasında klonlar, içine gömmez (vendor etmez); onu değiştirip ağ üzerinden
  hizmet olarak sunarsanız AGPL'in kaynak kodu paylaşma yükümlülükleri size aittir.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — Apache-2.0. Düzeltilmiş sohbet şablonu **kurulum sırasında indirilir, burada yeniden
  dağıtılmaz** (sabitlenmiş revizyon `855bffc4…`, sha256 `e57684ba…`). Başkasına verirseniz
  froggeric'i anın ve yeni sürümler için yukarı kaynağı kontrol edin.
- Model ağırlıkları burada **indirilir, yeniden dağıtılmaz**:
  [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3),
  Qwen'in sürümünden nicemlenmiş (quantized) ve Qwen topluluk lisansı altında — ticari
  kullanımdan önce lisans koşullarını kontrol edin.

Garanti verilmez; yapılandırmalar yukarıdaki donanım için ayarlandı, başka bir donanımda
değiştirmeniz gerekir.
