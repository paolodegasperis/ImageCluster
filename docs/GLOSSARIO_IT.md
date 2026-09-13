# Regole per la traduzione italiana

Documento di lavoro, non destinato agli utenti. Fissa le scelte che devono
restare costanti nelle circa 4.000 parole di documentazione italiana, perché
tradotte in momenti diversi divergerebbero e il risultato sembrerebbe scritto da
tre persone.

## A chi si rivolge il testo

A **studiosi italiani**, non a chi sviluppa. Sanno di cosa parlano nel loro
ambito e non sanno — né devono sapere — cosa sia un vettore normalizzato.
L'obiettivo dichiarato è farli arrivare a due cose: *che cosa fa il software* e
**quali informazioni è lecito dedurne**.

Ne discendono tre conseguenze concrete:

1. **I limiti non si attenuano mai.** Dove l'originale inglese dice che una
   distanza non è attendibile, o che il modello è cieco a una distinzione, la
   traduzione lo dice con la stessa nettezza. Per chi deve trarne conclusioni,
   sapere cosa il dato *non* autorizza vale quanto sapere cosa mostra.
2. **Niente entusiasmo aggiunto.** L'italiano tecnico divulgativo tende
   all'enfasi; qui non serve. Il testo inglese è asciutto di proposito.
3. **Si spiega il perché, non solo il come.** Un lettore che deve citare o
   contestare un risultato ha bisogno di sapere su cosa poggia.

## Registro

**Impersonale**, non «tu» né «lei»: *si apre il pannello*, *conviene partire da*,
*il risultato va letto come*. È la forma consueta della prosa tecnica italiana e
non presume confidenza con un lettore accademico.

Frasi brevi. L'inglese di partenza usa spesso incisi lunghi fra lineette: in
italiano si spezzano, perché la subordinazione lunga stanca prima.

## Ciò che NON si traduce

### I nomi di ciò che appare sullo schermo

Si scrive: *«si apre il pannello **Structure** e si sceglie una community»*.
Non: *«si apre il pannello Struttura»*.

Chi legge deve poter ritrovare l'etichetta nell'interfaccia, che resta inglese.
Una guida che nomina pulsanti inesistenti è **peggio** di una guida in inglese.

Restano quindi invariati:

- le viste: `Dashboard`, `Embedding Projection`, `Neighbor graph`, `Lenses`,
  `Guide`, `Info and settings`
- i pannelli e le schede: `Structure`, `Selected`, `Cluster gallery`,
  `Search results`, `Similar images`, `Analysis`
- i controlli: `Reducer`, `Layout`, `Build graph`, `Color nodes by`,
  `Neighbours per image (k)`, `Minimum similarity`, `Only reciprocal edges`,
  `Recompute`, `Generate projection`, `Compare`, `Preview`
- i nomi dei modelli: `OpenCLIP ViT-B/32`, `MetaCLIP 2`, `DINOv2`,
  `Visual Features`, `SigLIP`, `ImageBind`

In corpo di testo vanno in **grassetto**, così si distinguono dalla prosa e si
leggono come riferimenti all'interfaccia.

### I termini tecnici consolidati

Tradurli inventerebbe un vocabolario che non esiste in letteratura e che
renderebbe irreperibile qualsiasi approfondimento.

`CLIP` · `UMAP` · `t-SNE` · `PCA` · `k-NN` · `k-means` · `Louvain` · `DINOv2`
· `SigLIP` · `embedding` · `cluster` · `clustering` · `token` · `GPU` · `CPU`
· `min_dist` · `perplexity` · `seed` · `silhouette` · `modularity`

Sono maschili invariabili: *l'embedding*, *gli embedding*; *il cluster*, *i
cluster*. Mai «embeddings» o «clusters» con la esse del plurale inglese.

`community` resta invariato, femminile: *la community*, *le community*. Tradurlo
con «comunità» lo confonderebbe con il senso sociale e lo staccherebbe
dall'etichetta `Communities` che compare nel pannello.

## Rese fisse

| Inglese | Italiano | Nota |
|---|---|---|
| projection | proiezione | |
| to project | proiettare | |
| reducer | *resta inglese* | è un'etichetta dell'interfaccia |
| mosaic / grid mosaic | mosaico | |
| thumbnail | miniatura | |
| outlier | anomalia | «outlier» al primo uso fra parentesi |
| neighbour / neighbor | vicino | *i vicini più prossimi* |
| nearest neighbours | vicini più prossimi | |
| edge (del grafo) | collegamento | «arco» solo se il contesto è il grafo in senso stretto |
| node | nodo | |
| bridge | ponte | |
| articulation point / load-bearing | immagine portante | più chiaro di «punto di articolazione» |
| centroid | centroide | |
| cosine similarity | similarità coseno | |
| similarity score | punteggio di similarità | |
| query image | immagine di partenza | |
| duplicate / near-duplicate | duplicato / quasi-duplicato | |
| chained (gruppo) | concatenato | |
| collection | collezione | |
| dataset | collezione | «dataset» solo se si parla dell'addestramento |
| contact sheet | foglio di contatto | termine fotografico, corretto per il pubblico |
| caption | didascalia | ma «caption» quando si parla dei dati d'addestramento |
| scatter plot | grafico a dispersione | |
| plot | grafico | |
| axis / axes | asse / assi | |
| named axes | assi nominati | l'etichetta `Named axes` resta inglese |
| modality gap | scarto fra modalità | con l'inglese fra parentesi al primo uso |
| overlap | sovrapposizione | |
| tie / tied | pari merito | |
| threshold | soglia | |
| ranking | classifica | |
| rank | posizione | |

## Trappole ricorrenti

**`similar` non è sempre «simile».** In *«images the model finds similar»* il
soggetto è il modello: *«immagini che il modello giudica simili»*. La sfumatura
conta, perché tutto il testo insiste sul fatto che la somiglianza è un'opinione
del modello, non una proprietà delle immagini.

**`tells you` non è «ti dice».** Impersonale: *«che cosa rivela»*, *«che cosa si
ricava»*.

**`you can trust` → «su cosa si può fare affidamento»**, non «di cui ti puoi
fidare».

**`meaningless` non è «senza senso»** ma *«non significa nulla»*: le due cose in
italiano non coincidono, e la seconda è quella giusta quando si dice che gli
assi di UMAP non sono grandezze.

**`It is not a verdict`** e formule simili vanno rese letterali. Sono le frasi
che impediscono al lettore di sovrainterpretare, e sono la ragione per cui
questa documentazione esiste.

## Inglese come sorgente e ripiego

I file inglesi restano la versione di riferimento. Se una traduzione manca,
compare l'inglese: mai una pagina rotta, mai un buco. Una modifica al testo
inglese va riportata in italiano, ma il contrario non vale — l'italiano non
diverge per conto proprio.
