# Resize Images In-Place

Script Python per ridimensionare e ricomprimere immagini in modo ricorsivo all'interno di una cartella, mantenendo il formato originale.

---

## Funzionalità

- Ridimensiona le immagini in base alla dimensione massima specificata (in pixel).
- Supporta i formati: **JPEG, PNG, WEBP, TIFF**.
- Ricomprime le immagini per ridurre lo spazio occupato su disco.
- Gestisce la trasparenza per i formati che la supportano (es. PNG → JPEG con sfondo bianco).
- Registra un log dettagliato delle operazioni eseguite.

---

## Requisiti

- Python 3.7+
- Librerie richieste:
  - `Pillow` (per la gestione delle immagini)
  - `tqdm` (per la barra di progresso)

Installazione delle dipendenze:

```bash
pip install Pillow tqdm
```

---

## Utilizzo

Esegui lo script da linea di comando con i seguenti argomenti:

```bash
python resize_images_inplace.py --root <PERCORSO_CARTELLA> [OPZIONI]
```

### Argomenti principali


| Argomento               | Descrizione                                                          | Valore predefinito  |
| ----------------------- | -------------------------------------------------------------------- | ------------------- |
| `--root`                | Percorso della cartella contenente le immagini.                      | Obbligatorio        |
| `--max-side`            | Dimensione massima (in pixel) del lato più lungo dell'immagine.      | 1024                |
| `--quality`             | Qualità di compressione (1-95, solo per JPEG/WEBP).                  | 85                  |
| `--allow-upscale`       | Consente l'ingrandimento delle immagini più piccole di `--max-side`. | Disattivato         |
| `--allow-larger-output` | Consente il salvataggio anche se il file risultante è più grande.    | Disattivato         |
| `--dry-run`             | Simula l'operazione senza modificare i file.                         | Disattivato         |
| `--log`                 | Percorso del file di log.                                            | `resize_images.log` |


---

## Esempi

1. Ridimensiona tutte le immagini in una cartella a 800px (lato massimo):
  ```bash
   python resize_images_inplace.py --root ./immagini --max-side 800
  ```
2. Ricomprimi le immagini con qualità 90 e registra il log in un file personalizzato:
  ```bash
   python resize_images_inplace.py --root ./immagini --quality 90 --log ./mio_log.log
  ```
3. Simula l'operazione senza modificare i file:
  ```bash
   python resize_images_inplace.py --root ./immagini --dry-run
  ```

---

## Note

- Le immagini vengono sovrascritte **in-place**. Assicurati di avere un backup se necessario.
- Lo script ignora i file non supportati e registra gli errori nel file di log.