# Feature 4 — Metadati delle immagini come elemento di embedding (valutazione di fattibilità)

> Stato: **analisi**, nessuna implementazione. Documento di valutazione richiesto per
> decidere se e come integrare i metadati incorporati nelle immagini (EXIF/IPTC/XMP)
> nella pipeline di embedding dei modelli multimodali.

## 1. Obiettivo

Oggi la proiezione e la ricerca usano solo i **pixel** dell'immagine. Molti archivi (specie
artistici/fotografici) contengono però metadati testuali ricchi — didascalie, titoli, parole
chiave, autore — che descrivono il contenuto in modo complementare al segnale visivo.
L'idea è usare questo testo come **secondo segnale** da fondere con l'embedding immagine,
per i soli modelli che sanno encodare anche il testo.

## 2. Quali metadati e come estrarli

| Standard | Campi utili | Libreria |
|---|---|---|
| EXIF | `ImageDescription`, `XPKeywords`, `XPTitle`, `Artist` | Pillow `Image.getexif()` (già dipendenza) |
| IPTC (IIM) | `caption/abstract`, `keywords`, `headline`, `by-line` | `iptcinfo3`, oppure Pillow `IptcImagePlugin` |
| XMP | `dc:description`, `dc:subject`, `dc:title`, `photoshop:Headline` | `python-xmp-toolkit` (richiede Exempi), oppure parsing del blob XMP via regex/`defusedxml` |

Raccomandazione pratica: partire da **EXIF (Pillow, già presente)** + parsing leggero del blob
**XMP** (stringa XML dentro il file, estraibile senza dipendenze native). IPTC come estensione
opzionale. Evitare `pyexiv2`/`python-xmp-toolkit` in prima battuta per non introdurre
dipendenze binarie native poco portabili su Windows/macOS dei launcher one-click.

Normalizzazione consigliata: concatenare i campi presenti in un'unica stringa
(`"{title}. {description}. {keywords}"`), ripulita e troncata alla lunghezza del tokenizer.

## 3. Quali modelli possono sfruttarli

Il discriminante è `supports_text_embedding` nel registry ([backend/encoders/registry.py](../backend/encoders/registry.py)):

- **Idonei** (CLIP-like, spazio immagine-testo condiviso): OpenCLIP, MetaCLIP, MobileCLIP,
  CLIP OpenAI (`transformers_clip`), SigLIP/SigLIP2 (`transformers_image_features`).
  Per questi esiste già `encode_texts(...)` ([backend/encoders/base.py](../backend/encoders/base.py)).
- **Non idonei**: DINOv2 (`transformers_vision_pool`), Nomic Vision, ImageBind, e il nuovo
  **Visual Features** (solo formale): nessuno spazio testo allineato → i metadati non si
  fonderebbero in modo significativo. Per questi la funzione va semplicemente disattivata.

Poiché immagine e testo CLIP vivono nello **stesso spazio** ed entrambi gli embedding sono
L2-normalizzati, la fusione è matematicamente diretta.

## 4. Strategie di fusione

1. **Late fusion pesata (consigliata)**
   `v = normalize(α · v_image + (1−α) · v_text)`, con `α` configurabile (es. `metadata_weight`).
   - Pro: mantiene la stessa dimensionalità → compatibile con proiezione **e** ricerca testuale
     esistenti senza modifiche a valle; degrada con grazia (se manca il testo, `α=1`).
   - Contro: richiede che testo e immagine siano nello stesso spazio (vero per CLIP-like).
2. **Concatenazione** `v = [v_image ; β·v_text]` + ri-normalizzazione.
   - Pro: non assume allineamento tra i due spazi.
   - Contro: **raddoppia** la dimensione → rompe la ricerca testuale (la query testo non vive
     nello spazio concatenato) e cambia il significato delle distanze. Sconsigliata qui.

→ Per coerenza con la ricerca semantica già presente, la **late fusion pesata** è l'unica
strada che non comporta riscritture della pipeline di ricerca.

## 5. Punto di innesto nel codice (se si procederà)

- Estrazione: nuovo modulo `backend/metadata.py` con `read_image_text(path) -> str`.
- Encoder: in `encode_images` per i provider CLIP-like, dopo l'embedding immagine, se
  `metadata_weight < 1` e c'è testo, calcolare `encode_texts([meta], spec)` e fondere.
  In alternativa, fonderli in `jobs._run_clip_projection` per tenere gli encoder puri.
- Richiesta: aggiungere `metadata_weight: float = 1.0` (1.0 = solo immagine, come oggi) a
  `ClipProjectionRequest` ([backend/schemas.py](../backend/schemas.py)) e un controllo nel
  pannello sinistro, mostrato solo per i modelli con `supports_text_embedding`.
- Cache: includere `metadata_weight` (e un hash dei metadati) nel `dataset_hash` così le
  combinazioni immagine-solo e fuse non si sovrascrivono a vicenda.

## 6. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Metadati assenti o vuoti su molti file | Fallback automatico a `α=1` per quell'immagine; nessun errore. |
| Testo rumoroso (nomi file, software, watermark) | Whitelist dei campi (description/title/keywords), pulizia, lunghezza massima. |
| Lingua diversa dal modello | SigLIP2/alcuni CLIP sono multilingue; per gli altri il segnale è comunque additivo e pesato. |
| Aspettativa che "migliori sempre" | Esporre `metadata_weight` come cursore e documentare che è sperimentale; default 1.0 (comportamento attuale invariato). |
| Privacy (EXIF GPS/seriali) | Usare solo i campi testuali descrittivi, mai GPS/identificativi. |

## 7. Raccomandazione

**Fattibile e a basso rischio architetturale** per i modelli CLIP-like, tramite **late fusion
pesata** e un parametro `metadata_weight` con default 1.0 (nessuna regressione). Proposta di
fasaggio:

1. **Fase 1** — `backend/metadata.py` con estrazione EXIF (Pillow) + XMP leggero; copertura su
   un campione dell'archivio per misurare quanti file hanno testo utile.
2. **Fase 2** — fusione pesata dietro `metadata_weight`, solo provider con
   `supports_text_embedding`; validazione su una collezione reale (es. confronto qualitativo
   dei cluster con/senza metadati).
3. **Fase 3** — controllo UI nel pannello sinistro + nota nel manifest/JSON della proiezione
   per tracciare il peso usato.

Non bloccante per le altre funzionalità di questa release; può essere sviluppata in modo
indipendente quando si deciderà di procedere.
