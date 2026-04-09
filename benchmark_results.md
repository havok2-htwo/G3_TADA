# TADA True Batch Inference Benchmark Report
**Hardware:** NVIDIA RTX 5090
**Inference Engine:** Flow-Vocoder (20 Steps) mit dynamischem Causal LM Attention Batching

In den folgenden Tests wurde analysiert, wie stark die TADA-Modellfamilie (1B und 3B) von großen Batch-Sizes profitiert. Angegeben ist jeweils auch der Speedup (Echtzeitfaktor - "Real-Time Factor"), der angibt, wie viel *schneller als in Echtzeit* die Audio-Dateien berechnet werden (z.B. "`10x`" bedeutet "die GPU erzeugt in 1 Sekunde Rechenzeit fette 10 Sekunden Sprachaudio").

Bei True Batching skalieren die Berechnungen nahezu perfekt auf der GPU: Die Rechenzeit steigt kaum, aber der Yield an Audio schießt in den Himmel.

---

## 1. Modell: `HumeAI/tada-3b-ml` (German Benchmark)
Das große, mehrsprachige Modell liefert extrem hohe Sprachnuancen und Metadaten-Treue, fordert die 5090 aber entsprechend deutlicher.
**Audio-Ausgabe pro Item:** ~3.48 Sekunden (24 kHz)

| Batch Size | Gesamtzeit | Generiertes Audio (Gesamt) | Real-Time Factor (RTF) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | **2.92 s** | 3.48 s | **~1.19x** | 8.6 GB |
| Batch 2 | 2.41 s | 6.96 s | ~2.9x | 8.6 GB |
| Batch 4 | 2.21 s | 13.92 s | ~6.3x | 8.7 GB |
| Batch 8 | 2.36 s | 27.84 s | ~11.8x | 8.7 GB |
| Batch 16 | 2.69 s | 55.68 s | ~20.7x | 8.9 GB |
| Batch 32 | 3.02 s | 111.36 s | ~36.9x | 9.3 GB |
| Batch 64 | 3.96 s | 222.72 s | ~56.2x | 10.1 GB |
| **Batch 128** | **7.28 s** | **445.44 s** | **~61.2x** | **11.8 GB** |

**Fazit 3B:** Das große Modell schießt bei maximaler GPU-Auslastung (`B=128`) auf einen fantastischen Faktor von **~61x Echtzeit**. Das entspricht siebeneinhalb Minuten purem Multi-Voice-Audio in etwas über 7 Sekunden Wartezeit! Der VRAM steigt elegant von 8.6 GB auf machbare 11.8 GB – und bleibt damit weit unter dem 24 GB Limit!

---

## 2. Modell: `HumeAI/tada-1b` (English Benchmark)
Das kleine, pfeilschnelle 1B-Englisch-Modell benötigt deutlich weniger VRAM-Bandbreitendurchsatz und erlaubt damit selbst auf Consumer-GPUs aberwitzige Geschwindigkeiten.
**Audio-Ausgabe pro Item:** ~3.04 Sekunden (24 kHz)

| Batch Size | Gesamtzeit | Generiertes Audio (Gesamt) | Real-Time Factor (RTF) | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **Batch 1** | **2.28 s*** | 3.04 s | **~1.33x** | 3.8 GB |
| Batch 2 | 0.96 s | 6.08 s | ~6.3x | 4.0 GB |
| Batch 4 | 1.23 s | 12.16 s | ~9.8x | 3.9 GB |
| Batch 8 | 1.27 s | 24.32 s | ~19.1x | 3.9 GB |
| Batch 16 | 1.51 s | 48.64 s | ~32.2x | 3.9 GB |
| Batch 32 | 1.77 s | 97.28 s | ~54.9x | 4.1 GB |
| Batch 64 | 2.33 s | 194.56 s | ~83.5x | 4.5 GB |
| **Batch 128** | **3.56 s** | **389.12 s** | **~109.3x** | **5.2 GB** |

*\*Batch 1 beinhaltet den Initial-Warmup des Pytorch-Cuda-Allocators – der reguläre Throughput beginnt bei Batch 2.*

**Fazit 1B:** Unglaubliche **~109x Echtzeit**! Die GPU berechnet in nur knapp über dreieinhalb (!) Sekunden satte 6½ Minuten Sprachmaterial der höchsten Qualitätsstufe. Der VRAM-Verbrauch ist mit lachhaften **5.2 GB peak** in Batch 128 kaum existent. In einem Live-Einsatz könnte dieses Modell problemlos Hunderte simulierte KI-Agenten in Echtzeit vertonen, ohne jemals in einen CPU/GPU-Bottleneck zu laufen. Das Limit setzt hier rein der Hardware-Speicher der 5090 (24 GB VRAM).
