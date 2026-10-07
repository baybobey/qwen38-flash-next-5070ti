# qwen38-flash-next-5070ti — Strata sürümü

[English](README.md) | **Türkçe**

**Qwen3.8-Flash-Next GSQ-RCO IQ3_S** modelini **1 veya 2 adet 16 GB GPU** üzerinde
çalıştırmak için kendi kendine yeten kurulum betikleri ve ayarlanmış sunum
yapılandırmaları — 2x RTX 5070 Ti üzerinde ayarlandı. Motor olarak
[Niko1221/Strata](https://github.com/Niko1221/Strata) kullanır.

İhtiyaç duyduğu her şeyi (motor kaynağı, model, MTP taslak katmanı, sohbet şablonu)
indirir, CUDA motorunu yerelde derler ve bu kutuda ölçülen ayarlarla çalışan 262.144
token bağlamlı (context) bir yapılandırma yazar. Bu depoda hiçbir gizli bilgi yoktur:
API anahtarı kurulum sırasında sizin makinenizde üretilir.

**exllamav3 + TabbyAPI yığını** (EXL3 4.05; daha yavaş ama daha yüksek hassasiyetli
yol) [`exllamav3`](https://github.com/baybobey/qwen38-flash-next-5070ti/tree/exllamav3)
ve [`latest`](https://github.com/baybobey/qwen38-flash-next-5070ti/tree/latest)
dallarındadır.

```
./install-strata-2gpu.sh         # veya ./install-strata-1gpu.sh
./serve-2gpu-strata.sh           # veya ./serve-1gpu-strata.sh
./verify-strata.sh                # ikinci bir terminalde
```

## Donanım gereksinimleri

| | |
|---|---|
| GPU | 1 veya 2 adet NVIDIA, 16 GB sınıfı (Blackwell sm_120 üzerinde ayarlandı; motor sizin mimariniz için derlenir) |
| Sürücü | RTX 50 / sm_120 derlemesi için NVIDIA >= 580 ve CUDA 13 (sm_86+ için CUDA >= 12.4 yeterlidir) |
| Derleme araçları | `cmake`, `ninja`, `nvcc` — Strata'nın hazır Linux motoru yoktur; kurulum betiği derler (~16 çekirdekte ~2 dk) |
| Sistem RAM'i | 128 GB önerilir. GGUF ~47 GiB uzman (expert) + 26.9 GiB n-gram (PLE) tablosunu bellekte tutar; kurulum betiği ~62 GiB altında devam etmez |
| Disk | ~120 GB boş alan (model 83.6 GB + hazırlanmış paket ~1.5 GB + MTP taslağı ~5 GB + motor derlemesi) |
| OS / Python | Linux x86_64, Python 3.10+ (venv'i Strata'nın kendi kurulumu oluşturur) |
| PCIe | Geniş bant daha hızlı: decode önbelleğe alınmamış uzmanları PCIe üzerinden akıtır, x8/x16 fark eder |
| CPU | 16+ çekirdek önerilir (CPU uzman havuzları + prompt aşamalandırıcı) |

## Kurulum betiği ne yapar

Bu dizine:

- `Strata/` — `e8ca9af` commit'ine (v0.1.40.2) sabitlenmiş Strata klonu; ardından
  kendi `./setup.sh --yes --family qwen --model IQ3_S --context 262144 --kv int8
  --vision no --no-start --build` akışı: venv, CUDA motoru (sizin GPU'nuz için
  derlenir), model indirme, hazırlanmış paket (`Strata-data/packs/iq3_s`) ve MTP
  taslak katmanı (`Strata-data/mtp`; Strata'nın kendi `tools/mtp_fetch.py` aracıyla
  Qwen'in BF16 ağırlık dosyasından ~5 GB'lık aralık okumalarıyla getirilir — 360 GB'lık
  checkpoint asla indirilmez)
- `Strata/api_key.txt` — rastgele üretilmiş anahtar, `chmod 600`, hiçbir zaman ekrana
  basılmaz ve git'e girmez
- `Strata/strata-iq3_s.json` — çalıştırma yapılandırması; `configs/config-<N>gpu.json`
  dosyasından, bu makinenin yolları çözülmüş ve anahtar yerleştirilmiş olarak yazılır
  (git-dışı)
- sohbet şablonu — froggeric'in düzeltilmiş Qwen şablonu (`froggeric-v22.5`), yazarın
  deposundan sabitlenmiş revizyondan indirilir ve sha256 ile doğrulanır; tek yerel
  değişiklik uygulanır: `_default_reasoning_effort` `medium` → `xhigh` olarak
  sabitlenir. Paketin tokenizer dizinine kurulur (Strata, gömülü renderer yerine
  `<tokenizer>/chat_template.jinja` dosyasını tercih eder); orijinal şablon yanına
  `chat_template.orig.jinja.bak` olarak saklanır
- `Strata/strata-iq3_s.shared-settings.json` — kendisi göndermeyen istemciler için
  `reasoning_effort: high` (git-dışı)

Seçenekler: `--skip-model` (ağırlıklar zaten indirildi), `--model-dir D`, `--data-dir D`,
`--calibrate` (~10 dk; `--pcie-frac` / `--spec-min-p` değerlerini kendi PC'niz için
ölçer — yazarın kutusu dışındaki her makinede önerilir), `--context N`
(262144 | 524288 | 1048576), `--port N`, `--start`, `--dry-run`, `-y`. Tam liste:
`./install-strata.sh --help`.

Kurulumu tekrar çalıştırmak güvenlidir ve idempotenttir: klon, motor, venv, paket,
yapılandırmalar ve şablon yenilenir; model ve API anahtarı korunur.

## Çalıştırma

```
./serve-2gpu-strata.sh                     # 2 GPU, layer split auto, 262k (ayarlanmış varsayılan)
./serve-1gpu-strata.sh                   # 1 GPU
CTX=500k ./serve-2gpu-strata.sh        # 524.288 pencere (deneysel yarn ölçekleme)
CTX=1m   ./serve-2gpu-strata.sh            # 1.048.576 pencere (deneysel; ~95 GiB RAM)
GPU_IDS=1,0 ./serve-strata.sh --gpus 2     # kart sırası (nvidia-smi numaralandırması)
MODEL_DIR=/data/models ./serve-2gpu-strata.sh   # varsayılan dışı GGUF konumu
```

Her `serve-*` betiği önce kanonik yapılandırmayı (yollar + anahtar) yeniden uygular,
sonra sunucuyu ön planda çalıştırır; durdurmak için Ctrl-C ya da başka bir terminalden
`./stop-strata.sh`. Sunucu `0.0.0.0:8001` adresini dinler (anahtar korur — ağınız
için önemliyse güvenlik duvarı arkasında tutun) ve `Strata/api_key.txt` içindeki
anahtarı ister:

```
cat Strata/api_key.txt
```

OpenAI uyumlu her istemci: `base_url = http://127.0.0.1:8001/v1`,
`api_key = <bu değer>`, `model = qwen3.8-flash-next-iq3_s`. Ayrıca Anthropic uyumlu
`/v1/messages` uç noktası ve `http://127.0.0.1:8001/` adresinde yerleşik bir web
sohbet arayüzü vardır. `./verify-strata.sh`; sağlık durumunu, bağlam uzunluğunu kontrol
eder ve 4'lü işlevsel testi (`scripts/smoke-strata.py`: sağlık, OpenAI tool call, çok
turlu sohbet, Anthropic) çalıştırır.

## Ölçülen performans

Yazarın makinesi: 2x RTX 5070 Ti (her biri 16 GB, x8 + x4 PCIe), Ryzen 9 9950X3D2,
123 GiB RAM, sm_120 için yerelde derlenmiş motor 0.1.40.2. Sunucu tarafında ayarlanmış
örnekleici (sampler): **temp 1.0 / top-k 20 / top-p 0.95 / repetition 1.0**, xhigh
akıl yürütme. Aşağıdaki sayılar — aksi belirtilmedikçe — 58.7k tokenlık prompt
ile 2–4 çalıştırmanın medyanlarıdır; kendi kutunuzda PCIe genişliği, RAM hızı ve çekirdek
sayısına göre farklılık bekleyin (motorun kendi düğmelerini `--calibrate` yeniden
ölçer).

| Profil | Prefill (58.7k soğuk) | Decode 2k çıkış | Decode @58.7k derinlik |
|---|---|---|---|
| 2 GPU, `--prefill 12288 --pipeline-windows 2` | ~4.290 t/s | ~111–134 t/s | ~106–110 t/s |
| 1 GPU, chunk 12288 (ödünç alma) | ~3.370 t/s | ~88–90 t/s | ~75 t/s |
| 258k tam pencere promptu (262k'nın %98.4'ü) | 4.843 t/s motor (53 s) | — | ~130 t/s örnek |

Bağlam varyantları (deneysel; `scripts/check-ctx.py` uçtan uca doğrular — %90
derinlikte iğne (needle) bulunur, ardından derinlikte devam ölçülür):

| Pencere | Okuma | Derinlikte decode | Notlar |
|---|---|---|---|
| 524.288 (yarn 2) | 4.187 t/s @ 509.861 tok | ~99–109 t/s | `--kv-resident 98304`; park etme 12 GiB |
| 1.048.576 (yarn 4) | 3.462–3.499 t/s @ 1.016.182 tok | ~71–84 t/s | `--kv-resident 65536` bırakın (ödünç tavanına bakın) |

Karşılaştırma: aynı kutudaki exllamav3 yığını 825–943 t/s prefill, 30–34 t/s decode
yapar. IQ3_S ~3.5-bit'lik bir kuantlamadır, EXL3 4.05 bpw daha yüksek; Strata hızlı
şerittir, daha yüksek hassasiyetli yol exllamav3 dalında durur.

## Ayar notları (ibreyi gerçekten hareket ettiren düğmeler)

- `--prefill 12288` + prompt-önbellek ödünç alma — 0.1.39+'ta ölçülen dizgin
  (4096→2.2k, 6144→3.0k, 8192→3.4k, 12288→4.3k t/s; 16384, kart önbelleğinin %92'sini
  geçici ödünç alıp yalnızca +%1 getirir). Split üzerinde `--no-prefill-borrow`
  **eklemeyin**: kendi prompt tamponları uzman önbelleğini küçültür (kart başına 3.880
  vs 5.222 slot) ve derin decode'u ~%10 düşürür.
- `--pipeline-windows 2` (yalnızca 2 GPU split'i): decode +%15–20; maliyeti kart başına
  272 MiB.
- `--kv int8 --kv-resident 65536`: akışkan KV halkası. Tam-VRAM (`--kv-resident 0`)
  burada decode'u %18–25 düşürür — bu motorda uzman önbelleği KV kalıcılığından daha
  değerli.
- `--ple-io ram`: 26.9 GiB'lık n-gram tablosu bellekte (`ulimit` izin vermezse `mlock`
  yerine sayfa ön-dokunuşuna düşer; bu sorun değildir).
- Konuşma park etme (`--conversation-cache-mib 8192 --conversation-cache-slots 4`):
  ayrık bir istek (başka bir sohbet, istemcinin yardımcı çağrıları) canlı konuşmayı
  checkpoint zincirini silmek yerine RAM'e park eder; dönmek ~30 sn'lik tam re-prefill
  (~48k token) yerine ~0.4 sn sürer. 0.1.39'dan beri `--layer-split` ile de çalışır.
- Yapılandırmadaki `"sampling"` bloğu (temp 1.0 / top-k 20 / top-p 0.95): Strata bunları
  **varsayılan** olarak uygular — kendi değerini gönderen istemci yine kazanır. Bunlar
  olmadan, sampling göndermeyen istemciler kırpılmamış çalışır (uzun bağlamda "token
  çorbası").
- `env.STRATA_IQ_MT_MIN=1`: IQ paketlerinde tekrarlanabilir greedy çıktı (upstream #152)
  — decode'ta ~%1–3 maliyet; geri almak için satırı silin.
- GPU yerleşimine göre kalibrasyon: yazarın kutusunda `--pcie-frac 0.20 /
  --spec-min-p 0.70` (2-GPU), `0.35 / 0.70` (1-GPU) — kendi değerinizi `--calibrate` ya
  da `cd Strata && ./setup.sh --calibrate --no-start` ile yeniden ölçün.

## Bağlam varyantları ve ödünç tavanı

Eğitilmiş pencere 262.144'tür. 500k/1m profilleri onu `--rope-scaling yarn
--rope-scale F` (F = pencere / 262144) ile genişletir — **deneyseldir**: rope
değişikliği her konuma uygulanır, yani ölçekli bir çalıştırma 262k içinde bile hafifçe
farklı bir modeldir; her pencereyi kendi yapılandırmasında tutun ve turları pencereler
arasında karıştırmayın.

Bilinmesi gereken tuzak: bir prompt chunk, yapılandırıldığı boyutta ancak kart başına
ödünç aldığı (~0.27 slot/token: 12288 → ~3.310) en küçük uzman önbelleğinden 128 slot
düşüldüğünde sığarsa çalışır. Aksi halde motor `prompt chunk X -> Y` diye loglar,
sessizce küçültür ve okumalar %20–30 düşer. `--kv-resident`, `--pipeline-windows` ve
pencere boyutunun hepsi önbellekleri hareket ettirir — her değişiklikten sonra motor
logunda (`Strata/strata-iq3_s.log`) `prompt chunk` satırını arayın.

## Motoru güncelleme

`cd Strata && git fetch --tags && git checkout <yeni tag> && ./update.sh` — ardından
`./setup.sh --calibrate --no-start` yeniden çalıştırın (ölçülen optimumlar motor
kernel'leriyle birlikte kayar) ve kendi bağlamınızda `prompt chunk` satırını tekrar
kontrol edin. Buradaki yapılandırmalar yazarın kutusunda en son doğrulananı izler; şu
anki sabitleme motor v0.1.40.2 (`e8ca9af`).

## Sorun giderme

- **Tablodan çok daha yavaş okuma** — `Strata/strata-iq3_s.log` içinde `prompt chunk X
  -> Y` küçültmesi var mı bakın (ödünç tavanı) ve `--calibrate`ı kendi yerleşiminizle
  yeniden çalıştırın.
- **Başlangıçta "VRAM free … LOW" uyarısı** — `--vram-reserve-mib` yükseltin (bu
  ayarlama 1400 (2-GPU) / 1000 (1-GPU) kullanır).
- **Motor öldü / 503 "the engine is starting"** — sunucu onu yeniden başlatır;
  kutudaki bir kernel/sürücü reset'i masaüstü oturumunu düşürürken motor sunmaya devam
  edebilir (yazarın AMD iGPU'lu kutusunda olur; ağır işleri TTY/SSH'ten başlatın).
- **Uzun bir sohbette ilk cevap yavaş, sonrakiler hızlı** — tasarlandığı gibi: motor tüm
  promptu bir kez okur, her 16k tokende checkpoint alır ve tur başında yeniden
  kullanır. Ayrık bir istek artık silmek yerine *park eder* (park etmeye bakın).
- **İndirme yarıda kesildi** — kurulumu tekrar çalıştırın; HF indirmesi ve MTP
  getirmesi kaldığı yerden devam eder.
- **Tekrarlanan greedy çıktı farklı token'lar üretiyor** — `STRATA_IQ_MT_MIN=1`
  ayarlayın (yapılandırmalarda mevcut; motorlar env'i spawn'da okur, yapılandırma
  değişikliğinden sonra yeniden başlatın).

## Emeği geçenler ve lisanslar

- [Niko1221/Strata](https://github.com/Niko1221/Strata) — **MIT**. Kurulum sırasında
  klonlanır, depoya gömülmez; motor sizin makinenizde bu kaynaktan derlenir.
- Model ağırlıkları burada **indirilir, yeniden dağıtılmaz**:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  (revizyon `ed59f920…`, her iki shard Strata'nın kurulumunca sha ile doğrulanır);
  Qwen'in sürümünden Qwen topluluk lisansı altında kuantlanmıştır — ticari kullanımdan
  önce lisans koşullarını inceleyin. MTP taslak katmanı, Strata'nın araçlarıyla Qwen'in
  [BF16 ağırlık dosyasından](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) aralık
  okumasıyla getirilir.
- [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
  — Apache-2.0. **Kurulum sırasında indirilir, burada yeniden dağıtılmaz** (sabitlenmiş
  revizyon `855bffc4…`, sha256 `e57684ba…`); başkasına geçirirken froggeric'i anın.

Garanti verilmez; yapılandırmalar yukarıdaki donanım için ayarlanmıştır ve başka bir
donanımda değiştirmeniz gerekir.
