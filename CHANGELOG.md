# Changelog

## v6.6.0 - the documentation in Italian

The explanations are now available in Italian, for a readership of scholars
rather than developers. **Only the documentation** is translated: the interface
stays in English, and the Italian text says so in its opening lines rather than
letting anyone discover it by clicking a button.

| Surface | How |
|---|---|
| The guide | a parallel page at `/guide/it`, linked both ways |
| The graph explanation | both languages in the page, one hidden |
| The lenses explanation | likewise |
| The glossary and the notes on `/models` | likewise |

One shared choice, remembered, English by default: picking Italian in the guide
and then opening a help panel gives Italian, without choosing twice.

Parallel content rather than a branch or a string catalogue. A branch would give
two builds and no switch at all, and every future fix would have to be applied
twice, with the Italian copy quietly falling behind the first time a merge was
forgotten. This is prose with no logic and no interpolation, so a second copy
costs nothing and cannot drift from any code. English remains the source and the
fallback.

### The rule that matters most

Inside the Italian text, **the names of things on screen stay in English**: one
reads *«si apre il pannello Structure»*, never *«il pannello Struttura»*. A guide
naming buttons that do not exist would be worse than a guide in English.
`docs/GLOSSARIO_IT.md` records this and the fixed rendering of every recurring
term, so that four thousand words translated at different moments do not drift
into three different vocabularies.

The glossary on `/models` keeps its terms in English and translates only the
definitions, for the same reason.

### Named axes, findable

The feature existed and worked, but nobody could find it. Measured: the
projection-method control sat at y=887 in a 720px window, inside a rail that
scrolls 1775px of content through a 372px opening, and the **Preview** button -
the one action that works on the projection already on screen, in about a
second - landed at y=1547.

It was also filed inconsistently against its own sibling. Two functions from
v6.4.0, both optional, both driven by typed phrases, both operating on a loaded
projection: *Color by a concept* went into the Explore panel, *Named axes* stayed
at the bottom of the generation rail.

The phrases now live in Explore, next to the concept colouring they belong with,
and are always available rather than appearing only when the reducer is set to
semantic - Preview reads whatever projection is on screen, whichever method
produced it. The reducer option stays in step 3, where generating a projection
whose coordinates *are* the named axes belongs, and points at the panel. One set
of fields either way: everything reads them by id, so their location does not
matter.

Preview moved from y=1547 in a 372px scroll area to y=774 in a 672px one, with
the heading, the presets and most of the fields in view. The presets now come
before the fields, which is the order they are used in: one click fills all four.

### Alignment, and the rule behind it

Labels were being cut off and controls squeezed, in two places that turned out
to have one cause.

In the plot toolbar the buttons were fixed 32px squares while their labels are
words of different lengths, so **reset** (42px), **graph** (35px) and **grid**
(34px) were clipped - three of the ten. The column now sizes itself to its
widest label and flex stretch keeps every button identical.

The dock tabs looked like a tab problem and were not. **Expand**, a real
`<button>`, was 543px wide in a 912px bar because of the global
`select, input, button { width: 100% }` rule; that squeezed the tabs from their
natural 117px down to 78px, which wrapped their two-word labels onto a second
line, which overflowed the 32px bar. One cause, three visible symptoms.

That rule has now produced six separate defects: the dock controls, the export
buttons on community rows, the glossary toggle, the language toggle on
`/models`, the Expand button, and "How to read this" occupying 947px of a 1497px
topbar on two pages. Anchors styled as buttons are unaffected, which is why it
only ever appears once a real `<button>` joins a row - and why it kept being
rediscovered rather than recognised. The four scattered one-off fixes are now a
single rule that names the trap and lists the rows it applies to.

Narrowing the global rule itself would be the real fix, but it governs every
full-width button in the rails and deserves its own pass rather than being
changed in passing.

### Also

`#hfTokenMeta` was briefly made bilingual and then deliberately un-made: the
script writes live status and error messages into that element, so hiding it in
Italian would have replaced a real message with a fixed sentence. It is a status
line, not documentation.

A fourth instance of the global `select, input, button { width: 100% }` rule
claiming a whole flex row, this time under the new toggle on `/models`.

## v6.5.0 - results that can leave the screen

Every view in this app is a working instrument, and until now a result could
only be looked at. A community, a neighbourhood, a similarity ranking: each is
an answer worth taking away, annotating and returning to. Three surfaces now
export what they are showing.

### Contact sheets

| Where | What it exports |
|---|---|
| Neighbor graph, Structure, a community | every image in that community |
| Neighbor graph, Selected, an image | that image and its nearest neighbours, in order |
| Embedding projection, Similar images | the query and its matches, ranked |

One layout serves all three, deliberately: they are different questions, but a
folder of these documents should be readable without relearning the format each
time. Each sheet carries a heading saying what it contains and where it came
from, two images per row on A4, and under each the filename plus what that view
knows - similarity, rank, cluster, number of links, whether the image is
load-bearing in the graph.

**No image is embedded smaller than 500px on its long side**, which is the whole
point: a sheet you cannot look closely at answers nothing. Images whose original
is already smaller are included at their own size and **marked** rather than
enlarged - blowing them up would invent detail and hide that the source is
small. Measured on a real 43-image community: long side median 1000px, 11 pages.

Generated server-side, and not for convenience: the browser can only reach
images through the thumbnail endpoint, which clamps at 512px - no headroom at
all under that 500px floor. On the server the originals are on disk.

### CSV

The same results as numbers, for whatever comes after looking at them. Columns
differ between surfaces because the questions do, but the shape is constant: an
ordering column, then filename and path, then what the view knows, then the plot
coordinates. Two exports from different panels line up in one spreadsheet.

In both neighbour exports the image itself is **row 0 with no similarity**. A
ranking without the thing it was ranked against cannot be read back in six
months.

Files are written with a semicolon separator and a UTF-8 BOM. Excel follows the
system list separator, so a comma-separated file opens entirely in column A on
an Italian machine, and without the BOM every accent is mangled. The escaping is
covered by `node --test tests/js/*.test.js`, using Node's built-in runner - no
dependency, no package.json.

### A resizable results panel

340px was often not enough to read a cluster gallery in. Drag the panel's top
edge, double-click to reset, or use the arrows once the grip has focus; the
height is remembered. The plot keeps at least a third of the space, because a
panel that covers the view it belongs to has destroyed it.

The plot survives this without special handling - a ResizeObserver already
watched the canvas container, so the backing store, the click-to-coordinate
cache and the drawing all follow. Two things needed suppressing: the eased
flex-basis, which fought the pointer, and the mosaic reload, which asks the
server for a new layout and now fires once on release rather than per move.

### fpdf2, optional

Declared in `requirements-core.txt` and registered in `dependency_check.py`. It
is pure Python, so it installs from a wheel everywhere without a compiler.
Absent it, the export route answers 503 with install instructions and
**everything else is unchanged** - the same treatment networkx already gets.

---

## Fixes

A full audit before release found twelve defects. The ones that mattered:

**One corrupt embedding silently poisoned the semantic answers.** `semantic.py`
never sanitised its vectors, unlike `analysis.py`, `neighbors.py` and
`disagreement.py`. The subtle part: every guard in that module compares a
magnitude against a threshold, and **any comparison against NaN is false**, so
the zero-length check did not fire and `distinguish` went on to rank the
vocabulary along an all-NaN direction - returning a confidently ordered list of
words with null scores. Concept arithmetic returned zero results ("nothing
matched") when the real cause was a broken vector. One row was enough.

**The Structure panel blamed networkx for being absent when it was installed.**
The unavailable flag covered two unrelated causes - missing package, or no edges
- and the UI always reported the first, sending the reader to install something
they already had. The causes are now distinct, and a threshold that removed
every edge says so and names the three settings that would bring them back.

**A plain click on the new resize grip destroyed the remembered height.**
Pressing without dragging measured the collapsed 44px, clamped it up to the
140px floor, expanded the panel and wrote 140 over whatever the user had chosen.
Grabbing a grip is a natural gesture; it must not be destructive.

**Duplicate detection could exhaust memory.** Group statistics built a full
square matrix plus a boolean mask of the same shape. A scan series or a set of
video frames merges into one group approaching the 20 000-image ceiling, where
that is gigabytes and raises MemoryError - a type no route catches. Now
accumulated in row blocks: 225MB down to 32MB at 5 000 images, results identical.

**Five routes answered 500 where a 4xx belonged**, all reachable in ordinary use:
a log already pruned by housekeeping, a result deleted from `output/`, a
projection read on a directory, an unknown model key. The four job and
projection readers disagreed with each other - one checked the file still
existed, three did not.

**PDF generation had two paths to a 500.** PIL decodes lazily, so a truncated
file small enough to skip the resize failed at save time, which sat outside the
per-item guard: identical corruption produced a clean "1 missing" note when the
image was large and a 500 when it was small. And DecompressionBombError descends
from Exception, not OSError, so one very large panorama failed the whole sheet.

**The exact mosaic had no size ceiling.** The 2000-image limit gated the
automatic choice only; asking for it explicitly walked past it into a
multi-gigabyte allocation.

Also: the similar-images panel survived a projection reset and its export
buttons then posted a null path; FastAPI validation errors reached the user as
`[object Object]`; the three CSV exports had no error handling at all, so a
failure left a dead button; the dock body did not account for the grip's 3px and
the pane spilled past its border.

### Three smaller things

**A favicon.** The project shipped icons but no page declared one, so every page
view logged a 404 and the tab stayed blank. `/favicon.ico` now serves
`icone/ImageClusetIcon.ico` - the icon matching the current name, not the legacy
one carrying the old name - and every page declares it.

**Ties in the model-disagreement ranking are no longer disguised as an order.**
Overlap is a count out of k, so it takes k+1 distinct values and ties are the
rule: on two genuinely different models, dozens of images sit at zero together.
Numbering them 1, 2, 3 presented the tie-break - index order, therefore
alphabetical - as a measurement, and made the first few look singled out. Equal
scores now share a rank, shown as "=1 of 25 tied", and the panel says the order
within a tie is arbitrary.

**Obsolete machinery removed from `semantic.py`.** `_text_centroid`,
`_TEXT_CENTROIDS` and `DEFAULT_ARITHMETIC_STRENGTH` were all uncalled, and all
described the *first* implementation of concept arithmetic - the one that added
a phrase vector to an image vector and was swamped by the modality gap. That was
replaced in v6.4 by a combination in score space, where the gap stops mattering.
The strength constant carried measurements taken against the old behaviour:
re-measured on the current implementation, strength 1 moves 64% of results and
reaches 1.50 of the 2.02 sigma a term can achieve alone - not the 24% and 0.5
sigma the comment claimed. The default stays 1.0, now with the real figure
recorded where it is declared; at 3.0 the starting image has stopped mattering,
which is not what "images like this one, but more X" means.

### Verification

349 Python tests, 16 JavaScript tests, ruff clean, every `app/js` file passes
`node --check`. Exercised against real projections from four different models
rather than synthetic data alone.

## v6.4.1 - two things that did not work, and a page explaining the rest

### Concept arithmetic returned the same images whatever you typed

The feature was built on the obvious operation: add the phrase's vector to the
image's vector and rank the collection against the result. That does not work,
and the reason is a property of CLIP itself.

CLIP does not put images and words in the same region of its space. It puts them
in two separate cones. Measured on this collection, an image sits between
**0.13 and 0.23** from *every* phrase - "a landscape", "night", "a sculpture",
"snow" - regardless of what either depicts, while two images reach **0.81**.
Adding a unit text vector to a unit image vector therefore barely rotates it, the
image-to-image structure wins by a factor of four, and the answer was always the
base image's own nearest neighbours with the order faintly shuffled.

Centring each modality first - the standard remedy - was tried and was not
enough: after centring the image-to-text cosine drops to about 0.01, because the
text direction has almost no component inside the image cloud at all.

What works is to stop doing vector arithmetic. Each term - the starting image,
every idea added, every idea subtracted - is now scored against the whole
collection separately, and those score columns are **standardised** before being
combined. Each term then contributes in units of how much it actually separates
*this* collection, which is the only scale that means anything here. The same
query that used to return the base image's neighbours now returns, for
"+ a ruin", a completely different family of images - and they are ruins.

Two things follow from the change:

- An **idea strength** slider, because the balance between "near this image" and
  "strongly this concept" is a real choice and now has a real effect. At 0 the
  phrases are ignored entirely; at the top end the starting image barely matters.
- Scores are reported in **σ** - standard deviations above the collection's
  average - not as a cosine, because that is what they now are. The interface
  says so rather than showing a number that merely looks like a similarity.

Read the feature as *"images like this one, but more X and less Y"*. It is still
an instrument for thinking rather than a reliable operation.

### Choosing a community left the graph unchanged

Selecting a node dims everything except that node's handful of neighbours.
Highlighting a community dims everything outside the community. Both conditions
were applied at once, so with a node already selected, picking a community showed
their **intersection** - the two or three images that were both - and the graph
appeared not to react at all. The reported symptom was exact: it stayed on the
previously selected nodes.

The two answer different questions - one about a single image, one about the
whole collection - so they are now mutually exclusive. Choosing a community
clears the node selection; selecting a node clears the community. Clicking empty
canvas does not cancel a community (it only deselects a node), and Escape clears
both, being the one gesture that means "back out of everything".

Edges outside a highlighted community now recede as well. They were left at full
strength, so the group's own structure was buried under the links crossing it.

### A page explaining what every function is for

`/guide`, linked from the dashboard and from every view's header. Each feature in
plain language: the question it answers, what comes back, and what it cannot
tell you - including the limits that are properties of the models rather than
gaps in the app.

### Tests

Eight regression tests around the arithmetic, including the three that fail
against the old implementation: different concepts must give different answers,
strength 0 must ignore the phrases entirely, and raising the strength must pull
the ranking towards the concept.

Two more assert the graph fix on the source, there being no JS test runner here:
`selectNode` must drop a community focus, the community handler must drop a node
selection, and `drawEdges` must read the community at all. Each was confirmed to
fail with the fix reverted. 310 tests pass.


## v6.4.0 - asking the model questions, not just reading its answer

Everything before this release used one half of a CLIP-like model: the image
side. Embeddings came out, and the app arranged them. But these models put
images and words in the *same* space with the same coordinates, and search used
exactly one sentence of that shared vocabulary - "which images are nearest this
phrase". This release uses the rest of it.

### Semantic axes: a map with axes you named

UMAP and t-SNE answer "arrange these so that near stays near". The two axes that
come out mean nothing in particular - they are the directions the algorithm
happened to flatten along, and the horizontal one is not a quantity you can name.

A semantic axis is built instead from two phrases. "A portrait of a person"
minus "a landscape" is a direction in the shared space, and every image has a
position along it. Give a second pair for the vertical and the collection is
laid out on two axes you chose and can read - the axis names are drawn on the
plot itself.

Subtracting the poles is what makes it an axis rather than a score. "Portrait"
alone also points at *person*, *face*, *painting*; those components are shared
with the other pole, so the subtraction cancels them and leaves only what
separates the two.

It is **off by default**. The reducer list gains a third entry, *Named axes*,
alongside UMAP and t-SNE; choosing it reveals the pole fields, three presets
(subject / mood / setting) and a **Preview** that recomputes the layout against
the projection already loaded - no job, no re-encoding, so axes can be tried and
discarded in a second. The panel appears only for models that can read text, and
the app warns when the two axes overlap too much and the plot would collapse
onto a diagonal.

### Colour by a concept

The plot could be coloured by cluster, and clusters have no names. Now it can be
coloured by *how much of a concept* each image carries: type "a ruin" and every
image is shaded along a blue-to-red ramp by its own score.

This is a distribution, not a ranking. "Show me the top thirty ruins" and "where
do ruins occur in this collection" are different questions, and the second is
the one the map is good at answering - the concept may turn out to be spread
across two clusters, or concentrated in a corner of one.

The ramp spans the 2nd and 98th percentiles rather than the extremes: raw cosine
scores sit in a narrow band, so a ramp stretched over the theoretical range
would render the whole collection one colour. A legend bottom-left shows the
actual numbers, and the cluster legend steps aside while the concept is active.

Also **off by default**, and a single click restores the cluster colours.

### A new view: Lenses (`/lens`)

The projection, the mosaic and the graph all answer with a *picture of the
collection*. These three answer with something else - a list of words, a derived
query, a ranking of disagreements - so they live in their own view rather than
crowding a workspace built around a plot.

**What separates two groups, in words.** Both groups have a centre in the
model's space; the direction from one to the other is itself a point in that
space, and because words live there too, that direction can be read as language.
Whatever the groups share cancels in the subtraction, which is why the answer
names what *separates* them rather than what they are.

Groups can be two clusters, or - the default, and available on any projection -
the two ends of a concept: score the whole collection against "a landscape" and
contrast the highest with the lowest. On the test collection that returns,
towards A: *landscape, seascape, outdoor scene, cityscape, calm scene, sunset,
ruin*; towards B: *candlelight, sculpture, night, snow, portrait, nude*.

Two honest limits are shown with every answer. The **vocabulary constrains the
result** - the model returns the best terms *you offered*, even when none fits -
so the list is editable, one term per line, with a 52-term default spanning
subject, setting, technique and mood. And the **centroid similarity** says how
far apart the groups actually are: a long list of distinguishing words means
little when the two nearly coincide.

**Concept arithmetic.** An image is a point; so is a phrase. Subtract one, add
another, and the collection's nearest images to the result are the answer. It is
rarely clean - the space encodes web captions, not a grammar - but it is the
most direct demonstration of what an embedding space *is*, and it occasionally
surfaces a relationship nothing else would.

**Where two models disagree.** Generate the same folder twice, once with a
CLIP-family model and once with DINOv2 or Visual Features, and compare. CLIP
groups by what a picture *shows*; DINOv2 by how it *looks*. Neither is wrong -
they answer different questions - and the residue is the interesting part: the
works the two place differently have a gap between their subject and their form.
A landscape painted like a portrait. A sacred scene handled as genre. **Neither
model can point at these alone**, because each sees only its own answer; the
information exists only in the comparison.

The comparison is made on **neighbourhoods, not coordinates**. Two embedding
spaces share no frame - unrelated axes, different scales - so positions cannot
be compared, but "which images does this model consider closest to this one"
means the same thing in both, and the overlap of those two sets is a number that
travels. The code refuses to compare projections covering different images, and
refuses again if they hold the same images in a different order, with distinct
messages: comparing mismatched rows would produce plausible numbers about
nothing.

### Under the hood

- `backend/semantic.py` - the shared space beyond search: `concept_scores`,
  `semantic_axis`, `semantic_axis_preview`, `distinguish`, `concept_arithmetic`,
  and the default vocabulary. Refuses image-only models by name rather than
  failing obscurely later.
- `backend/disagreement.py` - neighbourhood comparison between two projections.
- `reducers.project_semantic_axes()` emits **raw cosine coordinates, unscaled**:
  the numbers on a named axis are meaningful, and normalising them away would
  throw out the only part that is.
- A pole averages several phrasings before normalising, so "a portrait", "a
  painted portrait" and "a head and shoulders" describe the direction more
  reliably than any single wording with its own caption-corpus quirks.
- 37 new tests covering all six features, with the text encoder stubbed so they
  assert what the code computes rather than what CLIP happens to think.


## v6.3.0 - what the graph knows about the collection

### Communities: the division components could not show

"Connected components" only separates images with no path between them at all.
Once k is high enough almost everything lands in one component, and colouring by
it paints the whole screen one colour - which is what most real collections look
like.

**Communities** (Louvain) answer the question components cannot. The algorithm
looks for groups whose members link to each other far more than they link
outward, and keeps refining until no move improves the split. It works inside a
fully connected graph, so it divides the collection even when every image is
reachable from every other.

What that gives you is the graph's own opinion about how the collection is
organised, reached without the 2D projection and without K-Means. Comparing it
with the cluster colours is the useful part: agreement means the grouping is
solid, disagreement means one of the two has distorted something. The
**modularity** figure says how clear-cut the division is - below about 0.3 the
collection is fairly homogeneous, above 0.5 the groups are strongly distinct.

Community is now the default node colouring. On a connected test graph:
components 1, communities 4, modularity 0.736.

### Bridges and load-bearing images

The help panel already told you to look for bridges; nothing helped you find
one. `bridges()` gives the links whose removal would split the graph, exactly
rather than approximately, and `articulation_points()` gives the images whose
removal would do the same. Bridges are drawn heavier in amber, load-bearing
images are ringed in amber, and both are listed with thumbnails.

### Projection stress: where the 2D map lies

In the *Projection coordinates* layout every link already carried two numbers
that ought to agree - its length on screen and the real similarity. Colouring
edges by their disagreement turns the graph into a diagnostic of the projection:
the warm links are strongly similar images that the map pushed far apart. No new
computation; both numbers were already there. The option explains itself and
steps aside in the force layout, where the positions come from the links and the
comparison would be circular.

### Compare two images

A cosine figure means little alone - whether 0.72 is "close" depends on the model
and the collection. The comparison puts three kinds of context around it: where
each image ranks in the other's neighbour list, where the pair sits among all
pairs, and which neighbours the two share. A pair at ranks #4 and #7 is mutual; a
pair at #1 and #200 is one-sided, and the panel says so in words.

Reached from a *Compare* button on any card or in the preview. The picker is a
tray that stays hidden until an image is picked, so nothing is added to the
workspace for a feature that is not in use.

### UI

The graph's right-hand panel now has two tabs, **Selected** and **Structure**.
The second holds the communities, bridges and load-bearing images, which keeps
the left rail from growing and uses space that sat empty whenever nothing was
selected.

### Also

- `networkx` is now declared in requirements-core.txt. It was already present as
  a dependency of scikit-image; relying on that would have let a future
  scikit-image release remove it without warning. Missing it degrades the
  structural reading to "n/a" and leaves the graph itself working.
- `httpx` added to requirements-dev.txt: Starlette's TestClient needs it, so
  without it the API tests could not run in the project venv.
- The dev server launch config now uses the project's own `.venv`.

## v6.2.1 - thumbnails are opaque at 100%

### Fixed: an unclustered projection was treated as "cluster 0"

`Number(null)` is `0`, and `0` is finite, so testing a missing cluster with
`Number.isFinite(Number(row.cluster))` answered "yes, cluster 0". On a projection
generated without clustering that meant:

- the gallery invented a single group and auto-selected it;
- every thumbnail got a blue cluster frame with a 14% blue wash beneath it;
- and because the selection comparison stringifies `null` to `"null"`, which never
  equals `"0"`, *every* image counted as outside the selected cluster and was
  dimmed to 30%.

The result was a plot of pale blue thumbnails that stayed pale at 100% opacity.
Measured against the source pixels, a thumbnail now matches them exactly at 100%.

The same expression appeared in ten further places, printing "Cluster: 0" or
"cluster null" on images with no cluster, in the search cards, the preview modal,
the similar-images header and the neighbor graph. All of them now go through a
guard that rejects null, undefined and empty string while still accepting a
genuine cluster `0`. A test fails the build if the bare expression comes back.

### Fixed: lowering the opacity faded to white instead of to transparent

The card behind each thumbnail was painted at a fixed 78% white, so a dimmed
thumbnail became a white tile rather than a transparent one. The backing now
fades with the slider: at 100% it is hidden behind an opaque image, and at 5% the
plot and the images underneath show through.

## v6.2.0 - four analyses over the stored embeddings

None of these loads a model. They read the vectors the projection already saved,
which makes them instant and - more importantly - makes them work for the
image-only models (DINOv2, Visual Features) that cannot search by text at all.
Those models previously offered a projection and nothing else.

### Similar images

"What else looks like this one?", with an image as the query instead of a phrase.
Double-click any image on the plot, press **Similar** on any card, or use **Find
similar images** in the preview. Results appear in their own dock tab and are
ringed on the plot in accent blue, separate from the white text-search ring so
both can be on screen at once.

### Duplicate and near-duplicate detection

Groups of images that are effectively the same picture, above an adjustable
cosine threshold (0.98 by default - re-saves and light crops land there).
Grouping is transitive, so a chain of re-saves is one group rather than several
overlapping pairs.

Two safeguards, because transitivity can also swallow a tight series whose
internal similarity sits just above the threshold. Groups are ordered by how
alike their members are rather than by size, so the confident ones lead; and a
group whose weakest internal pair falls below the threshold is flagged
**chained** - its members were joined through intermediaries, which is the signal
to look before deleting. The most central member of each group is offered as the
one to keep.

### Outlier ranking

The images least like anything else, ranked by similarity to their k-th nearest
neighbour rather than their nearest - otherwise a pair of duplicates that
resembles nothing else in the collection would read as perfectly typical. The
status line gives the median for comparison, and images well below it are badged.

### Cluster summaries

Each cluster now reports its size, its cohesion (mean similarity to the cluster
centre) and its **medoid** - the member closest to that centre, i.e. a real image
that represents the group. A strip above the gallery shows one chip per cluster
with the medoid thumbnail and a cohesion bar, and clicking a chip selects it.

The gallery itself is ordered most representative first. It previously sorted by
the projection's `cluster_score`, which is one silhouette value for the whole run
and therefore identical on every row - so that sort was alphabetical in practice.

### Also

- Static assets are stamped with the app version, so a stylesheet cached from a
  previous release cannot survive an update and leave the new markup unstyled.
- Dock toolbar buttons size to their label. The base rule gives every button
  `width: 100%`, which claimed a whole line each as soon as the row wrapped -
  latent until the cluster pane gained a Sort control and tipped over.
- The dock's pane switching is data-driven and lives in one place; it previously
  named its two panes explicitly in two different files.
- Removed the redundant `renderSearchResults` monkey-patch in `redesign_ui.js`:
  the search already selects and expands its pane.

## v6.1.0 — readability floor and an explained graph

### Thumbnail opacity now reaches 5%

The floor was 15%, which was not faint enough to read cluster colours through a
crowded region. It is 5% now — a ghost rather than nothing, since 0 would make
the plot look broken — on both the projection canvas and the graph view.

### The neighbor graph explains itself

The view answers a different question from the scatter plot, and the difference
is not self-evident from the controls. A **How to read this** panel (topbar
button, the link in the sidebar, the empty state, or the `?` key) now covers:

- what the view answers that the 2D map cannot, and why the projection distorts
  distances in the first place;
- how the graph is built, step by step, from vectors to filtered edges;
- how to read the picture — knots, hubs, bridges, isolated nodes, islands;
- what each setting changes, and when to reach for it;
- the interactions and the limits worth knowing.

Escape closes the panel before it touches the selection.

### A tooltip on every control

Every interactive control on the graph page (16 of them) and every view control
on the projection page now carries a `title` explaining what it does and when it
helps — including the statistics in the header bar, which read as bare numbers
otherwise, and the mosaic assignment methods, whose trade-off was previously
documented only in the source.

## v6.0.1 — second audit pass

A full re-read of the v6.0 changes, with each finding reproduced before being
fixed. Everything here was found by auditing rather than by a report.

### Fixed: one bad embedding vector degraded other images' neighbours

NaN compares false against everything, so `argpartition` could select a corrupt
vector into *any* node's top-k; the edge was then dropped by the finiteness
check and that node silently ended up with fewer neighbours than requested.
Non-finite rows are now zeroed before the similarity pass — similarity 0 never
outranks a real neighbour — and reported to the UI as `unusable_vectors`, which
the graph view surfaces.

### Fixed: an infinite coordinate collapsed the whole mosaic

`_normalize` used `nanmin`/`nanmax`, which ignore NaN but not infinity. One
infinite coordinate made the span infinite, mapping every real coordinate to 0,
and emitted a `RuntimeWarning` on stderr. Bounds are now taken from finite
values only, and non-finite coordinates are placed at the middle of their axis.

### Fixed: the model cache barely cached anything

Processors, tokenizers and weights shared one two-entry LRU. A single
Transformers model needs three of those, so the cheap objects evicted each other
on every call and the expensive checkpoint sat one step from eviction. Weights
and auxiliary objects now have separate budgets (2 checkpoints, 12 processors).

### Fixed: a large mosaic spun on thumbnail loading

A mosaic can put every cell on screen at once. Above the resident-bitmap cap,
each frame evicted exactly what the next frame requested — an endless load loop.
The mosaic now draws colored cells instead of thumbnails when a cell is below
12px, or when the visible cells would not fit the cache.

### Fixed: "Download TSV" handed over the wrong file

`ingestRows` merged the incoming job id with the previous one
(`jobId || state.jobId`), so after loading a saved projection the button still
pointed at the last job's result. The id is now assigned, and a projection
loaded from disk downloads through the `/output` mount.

### Fixed: a malformed config.json broke the dashboard

A section replaced by a value of the wrong type propagated into the running
configuration, so `dict(CONFIG["limits"])` raised and `/api/config` answered
500. The merge now discards any value whose shape does not match its default,
per key and at every depth, while still preserving unknown keys.

### Fixed: Escape threw away search results

Closing an image preview with Escape also called `clearSearch()`. Escape now
dismisses one layer at a time — preview, drawer, options panel, search focus —
and clears the results only when nothing else is open.

### Fixed: an error dialog that never appeared

`installer_ui.py` passed `exc` into a `lambda` scheduled on the Tk thread, but
Python unbinds the name when the `except` block ends, so the handler raised
`NameError` and the user saw nothing.

### Smaller fixes

- Cluster frames and focus rings were drawn outside the thumbnail with a fixed
  expansion, so in the mosaic they covered neighbouring cells.
- `startJob` left the previous mosaic painted behind the empty-state message.
- **Fit** measured the thumbnail slider even in mosaic mode, where the mark is
  the cell.
- `drawTooltip` and `focusSearchResult` called `getBoundingClientRect()`
  directly, forcing a layout on every hover frame and defeating the rect cache.
- The thumbnail LRU rewrote its Map for every drawn row on every frame;
  recency now lives in a side map of frame numbers.
- `layoutFromProjection` spread the whole node array into `Math.min(...)`.
- Cluster defaults were snapshotted at import, so `reload_config()` changed
  every request field except those.
- Removed `nodeRadius`'s duplicate branch and an unused `active_job_count`.

### Tests

202 tests, up from 150. New files cover degenerate numeric input, the
cached-embedding reuse path end to end (stubbed encoder, no torch needed), the
encoder cache's eviction behaviour, and config.json type tolerance.

## v6.0.0 — two new views, and a maintenance pass

### New: grid mosaic view

A scatter of thumbnails hides its own dense regions: images overlap, small
groups disappear behind large ones, and the space an image occupies means
nothing. The mosaic gives every image an equal cell of a rectangular grid while
keeping neighbours in the projection neighbours in the grid.

- `backend/layouts.py` — two assignment strategies. **Optimal** solves a true
  linear assignment (`scipy.optimize.linear_sum_assignment`, Jonker-Volgenant
  style) minimising total squared distance to cell centres; used automatically up
  to 2000 images. **Recursive split** halves the grid along its longer side and
  splits the images along the matching axis in the same proportion; O(n log n),
  no extra dependency, used for larger sets and selectable manually.
- The grid's shape follows the canvas aspect, and the column count can be pinned.
- Reached from the plot toolbar (`img` / `pts` / `grid`) or the `M` key.

### New: nearest-neighbor graph view

A standalone view at `/graph`. Edges are computed in the **full embedding
space**, not the 2D projection, so an edge means real similarity rather than an
accident of the flattening.

- `backend/neighbors.py` — chunked cosine k-NN, undirected de-duplicated edges
  flagged as mutual or one-directional, per-node degree, and connected
  components via union-find.
- Controls for k, a minimum-similarity threshold, and reciprocal-only edges.
- Two layouts: the projection's own coordinates (instant), or a force-directed
  relaxation seeded from them with grid-approximated repulsion.
- Clicking a node dims everything except its neighbourhood and lists the nearest
  neighbours with their similarity scores.

### New: thumbnail opacity

Thumbnails can be dimmed (15–100%) so overlapping images stay readable. Present
in both the projection canvas and the graph view, and composed with the existing
search/cluster dimming rather than overriding it.

### Fixed: embeddings could be attributed to the wrong images

Encoders skip files they cannot read, so the encoded set can be a strict subset
of a directory scan. The manifest previously stored only a count, and both the
cached-embedding path and text search re-derived the order from a fresh scan. One
added file plus one deleted file left the counts equal and silently shifted every
row onto the wrong image. Manifests now record the exact encoded paths, and a
manifest without them is re-encoded rather than trusted.

### Fixed: models were reloaded on every call

Only the Sentence-Transformers encoder cached its model. Every text query
reloaded a multi-GB checkpoint to encode one sentence, and every projection job
paid the load cost again even when the embeddings themselves were cached. All
encoders now share a small LRU (`backend/encoders/model_cache.py`).

### Fixed: unbounded growth

- In-memory job history is capped (`limits.max_jobs_in_memory`), oldest finished
  jobs first; running jobs are never dropped and stale cancel requests are cleared.
- The thumbnail cache is trimmed to `limits.thumb_cache_max_mb`, oldest first.
- `output/logs` is trimmed to `limits.max_log_files`.
- Successful searches no longer write a debug JSON each; result files are opt-in
  via `semantic_search.save_search_results` (now off by default).
- Concurrent embedding jobs are queued (`limits.max_concurrent_jobs`, default 1)
  instead of loading two checkpoints into memory at once.

### Fixed: config.json is now load-bearing

Only `host`, `port` and `block_start_if_dependencies_missing` were ever read. The
clustering, semantic-search and debug sections are now wired to the code that
they describe, unused keys were removed, `limits` and `views` sections were added,
and a new `/api/config` endpoint feeds the UI its defaults. A malformed file falls
back to defaults instead of blocking startup.

### Fixed: one version, one name

`backend/version.py` is the single source of truth. Previously the version was
stated five different ways (`5.5-dev`, `0.6.0-dev`, `v6.0`, `V53`, `v5.2.2`) and
the product alternated between "ImagePlot-CLIP" and "ImageCluster".

### Frontend fixes

- Hover now honours the same filters as the canvas; previously a tooltip could
  describe an image the cluster filter had removed from view.
- Thumbnails load on demand for the rows actually drawn, a few at a time, with a
  cap on resident bitmaps. Previously one `Image` was created per row up front.
- The wheel zooms towards the cursor instead of the canvas centre.
- **Fit** frames the rows currently visible instead of being an alias for **Reset**.
- Opacity read from `localStorage` defaulted to fully transparent, because
  `Number(null)` is `0` and passed the `isFinite` guard.
- Removed `drawClusterLegend` and `drawPoint`, both dead since the legend became
  HTML.

### Dependencies

- Removed `pandas` (declared required, never imported) and `python-multipart`
  (no uploads).
- Added `scipy` for the optimal mosaic assignment; a greedy fallback runs without it.
- All requirements now carry version bounds. `requirements-dev.txt` added.

### Tests and tooling

Test count went from 8 to 150. New coverage for path containment and traversal,
reducers and clustering edge cases, manifest alignment, exports, job lifecycle,
config loading, housekeeping, mosaic layout, neighbor graph and both view
endpoints. Added `pyproject.toml` with pytest and ruff configuration.

Export filenames are now deterministic: the collision suffix used `hash()`, which
Python randomises per process.

## V-5.5 — multimodal embedding models via Sentence-Transformers

### New models

Two new multimodal embedding models added as optional dependencies, using the
**Sentence-Transformers API** (`model.encode`) rather than the CLIP-style
`get_image_features`. This required a dedicated encoder provider, not just
registry wiring.

- **Qwen3-VL Embedding · 2B** — `Qwen/Qwen3-VL-Embedding-2B`. Joint text/image
  embedding space (2048-dim). Requires `sentence-transformers`, `transformers>=4.57`,
  `qwen-vl-utils`. Large GPU.
- **Jina v5 Omni · Small** — `jinaai/jina-embeddings-v5-omni-small` (~1.74B,
  1024-dim, Matryoshka). Requires `sentence-transformers`, `transformers>=4.57`,
  `peft`. GPU recommended. **License: CC BY-NC 4.0 (non-commercial use only).**

Both models follow the optional-dependency pattern: they are listed in the model
selector but disabled until the matching packages are installed. A double-click
installer in the project root activates them without requiring terminal knowledge.

### New encoder

- `backend/encoders/sentence_transformers_encoder.py` — new provider handling image
  and text encoding, per-process model cache, batching, progress reporting, and
  per-image error resilience.
- `backend/encoders/base.py` — routing for `provider = "sentence_transformers"` in
  both `encode_images()` and `encode_texts()`.

### New installer scripts

- `Install additional models (Windows).bat` — root shortcut (double-click)
- `Install additional models (macOS).command` — root shortcut (double-click)
- `bootstrap/windows/install_optional_multimodal.bat`
- `bootstrap/macos/install_optional_multimodal.command`

All install: `sentence-transformers transformers>=4.57 qwen-vl-utils peft`

### Other changes

- Version string updated to `5.5-dev` in `backend/main.py`.
- Registry dependency key `peft` added to `IMPORT_NAMES`.
- Tests extended: `test_v55_multimodal_models_are_registered()` covers both new models.

> Verification status: registry, encoder wiring, and UI behaviour verified
> structurally (12/12 tests). Real load + encode must be confirmed on a GPU machine
> after installing the optional deps.

---

## V-5.4 — model registry expansion (Wave A)

Three roadmap models promoted from `planned` to `experimental` — wired to existing
encoders, no new encoder code required.

### Promoted models

- **EVA-CLIP · L/14** — OpenCLIP path, `EVA02-L-14` / `merged2b_s4b_b131k`.
  GPU recommended.
- **EVA-CLIP · bigE/14** — OpenCLIP path, `EVA02-E-14` / `laion2b_s4b_b115k`.
  Large GPU; multi-GB checkpoint, not a default choice.
- **MetaCLIP 2 · Worldwide H/14** — Transformers MetaCLIP 2 path,
  `facebook/metaclip-2-worldwide-huge-quickgelu`. Large GPU. Reuses the proven
  MetaCLIP B/32 sibling loader.

All three support text search (joint image/text embedding space).

### Still planned

- **LaCLIP** — architecture is standard CLIP, but official weights ship as `.pt`
  checkpoints in the `LijieFan/LaCLIP` GitHub repo, not as an OpenCLIP pretrained
  tag or a Hugging Face Transformers repo. Needs a verified `hf-hub:` mirror or a
  checkpoint-file loader before promotion.
- HQ-CLIP B/16, Long-CLIP B/32, MetaCLIP 2 2B, CLOC — see
  `docs/PLANNED_MODELS_FEASIBILITY.md`.

### How to verify Wave A models

With torch + open_clip + transformers installed (GPU recommended):
1. Open `/models`, filter by **Experimental** — confirm the three cards appear.
2. In `/clip`, scan `img`, select each model, click **Generate** — first run
   downloads the checkpoint.
3. Run a text search query for each — all three are text-capable.
4. Promote to `stable` once load, projection, and search are verified and hardware
   requirements are acceptable.

---

## V-5.3 — performance and UI consolidation (frozen baseline)

### Rendering performance (`app/js/clip_projection.js`)

- **Memoized bounds** — data extent computed once per projection load (O(n)
  single-pass), invalidated on new data.
- **Cached canvas rect** — `getBoundingClientRect()` called once and reused across
  mouse events instead of once per point per event.
- **rAF coalescence** — `scheduleDraw()` prevents more than one redraw per animation
  frame regardless of event rate.
- **Viewport culling** — off-screen points are skipped in `draw()`.
- **Level-of-detail** — colored points at zoom < 2× when the collection has more than
  500 images; thumbnails at zoom ≥ 2×. Restores smooth interaction at 2,500+ images.
- **`/api/thumb` endpoint** — PIL resize, SHA1 disk cache in `output/thumb_cache/`,
  path-traversal protected, ~20× smaller than originals. `Cache-Control: max-age=86400`.
- **Progressive thumbnail loading** — non-blocking; each thumbnail triggers a
  `scheduleDraw()` on load so the canvas fills in without freezing.

### UI redesign (production stack)

- Three-column `/clip` shell: left rail (workflow steps), centre (canvas), right panel
  (search + cluster gallery).
- Run-state machine: `idle → scanned → ready → running → done` drives step badges
  and the Generate button.
- Capability-aware semantic search bar: 3-state indicator (ok / warn / danger) with
  tooltip distinguishing capability from runtime availability.
- Step panels are collapsible (click the step header).
- Collapsible setup-help box in the dashboard (replaces inline install-advice block
  that was leaking into the `/clip` left rail).
- Token-first CSS pass; legacy dark control leakage fixed (selects, checkboxes,
  range inputs, modal, model detail buttons).
- `localStorage` namespace: `imageplot.*` → `imagecluster.*`.
- Download filenames: `imageplot-*` → `imagecluster-*`.
- Keyboard shortcuts: `G` (generate), `F` (fit), `R` (reset zoom), `C` (cluster
  gallery toggle), `Esc` (close modal) — guarded against INPUT/SELECT focus.

### Removed

- `/original` route and the `imageplot_original.html` page.
- `renderInstallAdvice()` from the clip projection module.
