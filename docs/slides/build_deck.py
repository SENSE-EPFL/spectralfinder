"""Build the Monday deck: docs/slides/point_12_octobre.pptx.

    uv run docs/slides/make_figures.py            # figures (docs/slides/fig/)
    uv run --with python-pptx docs/slides/build_deck.py

The "what changed" slide reads the effect of each correction from the newest
results/variants*.csv (uv run compare_variants.py V0 ... V9), not typed in by hand.
"""
import csv
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "fig"
OUT = HERE / "point_12_octobre.pptx"

# ---------------------------------------------------------------- style -----
INK, TEAL, ACCENT = "1E2430", "1F7A8C", "E8702A"
LIGHT, MUTED, WHITE, SOFT = "F3F5F8", "5B6472", "FFFFFF", "C9D3E3"
TAG = {"FILTRE": "C22B2B", "CLASSEMENT": "1F5FBF", "AFFICHÉ": "6B6B6B"}
GOOD, BAD = "1A7D3A", "C22B2B"
HEAD, BODY = "Cambria", "Calibri"
N_SLIDES = 10
FOOTER = "Léan Bruttin · Sélection de fenêtres spectrales · 12.10.2026"


def rgb(h):
    return RGBColor.from_string(h)


def nb(t):
    """French typography: never break before a unit ('2,6 %' must not end on a lone '%')."""
    for unit in (" %", " ‰", " cm⁻¹", " ppm", " °C", " K)", " km", " mbar"):
        t = t.replace(unit, "\u00a0" + unit[1:])
    for sign in ("> ", "≥ ", "≤ ", "< "):  # nor right after a comparison sign
        t = t.replace(sign, sign[0] + "\u00a0")
    return t


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
LAYOUT_TITLE_ONLY = prs.slide_layouts[5]
LAYOUT_BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, paragraphs, size=15, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, space_after=6, margin=0.0):
    """paragraphs: str, or list of paragraphs; a paragraph is a str or a list of
    (text, {size, color, bold, italic, font}) runs."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    if isinstance(paragraphs, str):
        paragraphs = [paragraphs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = [(para, {})] if isinstance(para, str) else para
        for t, opt in runs:
            r = p.add_run()
            r.text = nb(t)
            f = r.font
            f.name = opt.get("font", font)
            f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold)
            f.italic = opt.get("italic", italic)
            f.color.rgb = rgb(opt.get("color", color))
    return box


def shape(slide, kind, x, y, w, h, fill, line=None, radius=None):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def card(slide, x, y, w, h, fill=LIGHT):
    return shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, fill, radius=0.06)


def chip(slide, x, y, label, w=1.15):
    s = shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, 0.3, TAG[label], radius=0.3)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = BODY, Pt(10), True
    r.font.color.rgb = rgb(WHITE)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return s


def badge(slide, x, y, label, d=0.5, fill=TEAL, size=15, color=WHITE):
    """The deck's motif: a number (or short label) in a coloured circle."""
    s = shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = BODY, Pt(size), True
    r.font.color.rgb = rgb(color)
    return s


def picture(slide, path, x, y, w=None, h=None, alt=""):
    """Image fitted inside w x h (keeps its aspect ratio), anchored top-left."""
    iw, ih = Image.open(path).size
    if w and h:
        scale = min(w / iw, h / ih)
        w, h = iw * scale, ih * scale
    elif w:
        h = w * ih / iw
    else:
        w = h * iw / ih
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    pic._element.nvPicPr.cNvPr.set("descr", alt)
    return pic, w, h


def content_slide(n, title, message, notes):
    slide = prs.slides.add_slide(LAYOUT_TITLE_ONLY)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.8)
    tf = t.text_frame
    tf.margin_left = tf.margin_right = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.name, r.font.size, r.font.bold = HEAD, Pt(34), True
    r.font.color.rgb = rgb(INK)
    text(slide, 0.6, 1.12, 12.1, 0.5, message, size=17, color=TEAL, italic=True)
    text(slide, 0.6, 7.02, 9.0, 0.3, FOOTER, size=10, color=MUTED)
    text(slide, 11.7, 7.02, 1.0, 0.3, f"{n} / {N_SLIDES}", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


def dark_slide(notes):
    slide = prs.slides.add_slide(LAYOUT_BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(INK)
    slide.notes_slide.notes_text_frame.text = notes
    return slide


# ------------------------------------------------------- comparison data ----

STEP_LABELS = {  # variant -> what that step adds (compare_variants.py)
    "V1": "Pic lu à la raie, pas au max de la fenêtre",
    "V2": "Ailes des raies sommées sur 5 cm⁻¹",
    "V3": "Grille de calcul 5× plus fine",
    "V4": "Interférence (seulement affichée)",
    "V5": "Température (seulement affichée)",
    "V6": "Filtre de saturation",
    "V7": "Sommet de la courbe affiné",
    "V8": "Écart à la plage mesuré sur τ",
    "V9": "Raie choisie pour son absorption",
}


def comparison():
    """{variant: number of its top-10 windows NOT in the previous step's top 10},
    {variant: rank of the SubOcean window or None}."""
    order = ["V0"] + list(STEP_LABELS)
    top = {}
    # the newest results/variants*.csv that has every step (uv run compare_variants.py [V0 ... V9])
    for path in sorted((ROOT / "results").glob("variants*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        top = {}
        for r in csv.DictReader(open(path)):
            top.setdefault(r["variant"], []).append(r)
        if all(k in top for k in order):
            print(f"comparison data: {path.name}")
            break
    changed, ref = {}, {}
    for prev, k in zip(order, order[1:]):
        if k not in top or prev not in top:  # comparison not (re)run yet with this step
            changed[k] = 0
            continue
        prev_c = [float(r["center"]) for r in top[prev][:10]]
        cur = top[k][:10]
        changed[k] = sum(all(abs(float(r["center"]) - c) >= 0.05 for c in prev_c) for r in cur)
    for k in order:
        ref[k] = next((int(r["rank"]) for r in top.get(k, []) if float(r["lo"]) <= 4398.15 <= float(r["hi"])), None)
    return changed, ref


# ================================================================ slides ======

# 1 -- title ------------------------------------------------------------------
s = dark_slide("Présenter le but en une phrase : un outil qui choisit automatiquement la fenêtre de "
               "balayage laser où tous les gaz se mesurent en une seule mesure. Le fond : le vrai spectre "
               "calculé autour de la fenêtre SubOcean (CH4 en bleu, N2O en orange, H2O en vert).")
picture(s, FIG / "hero.png", 6.73, 0, w=6.6, alt="Spectre calculé de CH4, N2O et H2O autour de 4398 cm-1")
text(s, 0.7, 1.75, 6.0, 2.4, "Sélection automatique des fenêtres spectrales", size=40, color=WHITE,
     bold=True, font=HEAD, anchor=MSO_ANCHOR.BOTTOM, space_after=0)
text(s, 0.7, 4.3, 6.0, 0.9, "Workflow du code, évolutions et premiers tests", size=20, color=SOFT)
text(s, 0.7, 5.6, 6.0, 0.4, "Léan Bruttin · Projet de semestre MA3 · EPFL", size=14, color=WHITE)
text(s, 0.7, 6.0, 6.0, 0.4, "Point du lundi 12 octobre 2026", size=14, color=ACCENT, bold=True)
text(s, 0.7, 6.55, 6.0, 0.4, [[("CH4", {"color": "3B8FD4", "bold": True}), ("   ·   ", {"color": SOFT}),
                               ("N2O", {"color": "FF9A3C", "bold": True}), ("   ·   ", {"color": SOFT}),
                               ("H2O", {"color": "3CB44B", "bold": True})]], size=12)

# 2 -- workflow (+ objective) -------------------------------------------------------------------
s = content_slide(2, "Le workflow en cinq étapes",
                  "Objectif : la fenêtre laser où tous les gaz se mesurent, en un seul balayage",
                  "Le but : pour un instrument donné (L, pression, température, largeur de balayage), trouver la "
                  "fenêtre où tous les gaz se mesurent ensemble. Vue d'ensemble, détaillée dans les deux slides "
                  "suivantes. Les étiquettes disent le rôle de chaque "
                  "étape : FILTRE élimine des fenêtres, CLASSEMENT décide de l'ordre, AFFICHÉ est seulement "
                  "reporté. L'étape 1 travaille sur la liste des raies (rapide), l'étape 2 calcule le vrai "
                  "spectre, seulement sur 200 fenêtres.")
steps = [
    ("Données", "gaz + concentrations,\ninstrument, raies HITRAN", []),
    ("Étape 1\nprésélection", "liste des raies,\n2000–9000 cm⁻¹", ["FILTRE"]),
    ("Étape 2\nvrai spectre", "profil de Voigt,\npic = sommet adapté", ["FILTRE"]),
    ("Classement", "gaz dans la plage\n1–20 % d'absorption", ["CLASSEMENT"]),
    ("Vérification", "interférence ≤ 5 %,\nsaturation, température", ["FILTRE", "AFFICHÉ"]),
]
x0, colw = 0.6, 12.1 / 5
line = shape(s, MSO_SHAPE.RECTANGLE, x0 + colw / 2, 2.43, colw * 4, 0.05, "B8C2CF")
for i, (head, desc, tags) in enumerate(steps):
    cx = x0 + colw * i + colw / 2
    badge(s, cx - 0.45, 2.0, str(i + 1), d=0.9, size=24)
    text(s, cx - colw / 2 + 0.1, 3.1, colw - 0.2, 0.85, head, size=17, bold=True, align=PP_ALIGN.CENTER,
         font=HEAD, space_after=0)
    text(s, cx - colw / 2 + 0.1, 3.95, colw - 0.2, 0.8, desc, size=13.5, color=MUTED, align=PP_ALIGN.CENTER)
    total = len(tags) * 1.15 + (len(tags) - 1) * 0.1
    for j, tag in enumerate(tags):
        chip(s, cx - total / 2 + j * 1.25, 4.85, tag)
card(s, 0.6, 5.55, 12.1, 0.75, fill="E6F0F2")
text(s, 0.95, 5.55, 11.5, 0.75, [[("Résultat  ", {"bold": True, "color": TEAL}),
                                  ("les 10 meilleures fenêtres, avec pour chaque gaz son pic, son absorption, "
                                   "son interférence et sa sensibilité à la température", {})]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)
legend = [("FILTRE", "fenêtre éliminée si la condition n'est pas remplie"),
          ("CLASSEMENT", "décide de l'ordre"), ("AFFICHÉ", "reporté, n'influence pas le choix")]
lx = 0.6
for tag, desc in legend:
    chip(s, lx, 6.55, tag)
    width = 0.072 * len(desc) + 0.15
    text(s, lx + 1.25, 6.52, width, 0.35, desc, size=11, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    lx += 1.25 + width + 0.3

# 3 -- selection: three rules ---------------------------------------------------------------
s = content_slide(3, "Sélection · trois règles",
                  "Elles décident quelles fenêtres restent, et dans quel ordre",
                  "1 : chaque gaz doit avoir une raie dans la fenêtre, et les raies des gaz doivent être séparées "
                  "d'au moins 3 largeurs, sinon on ne peut pas attribuer le signal. 2 : sur le vrai spectre "
                  "(profil de Voigt), le pic retenu est le sommet de la courbe — affiné entre les points de "
                  "calcul, choisi pour son absorption, jamais l'aile d'une raie hors fenêtre. 3 : chaque gaz "
                  "doit absorber entre 1 et 20 % à son pic ; le seuil de 1 % vient de ton mail (SNR 1000). "
                  "On classe par nombre de gaz dans la plage, puis par l'écart à la plage.")
rules = [
    ("sketch_A_small.png", "1 · Une raie de chaque gaz", "FILTRE",
     "Sinon la fenêtre est éliminée. Les raies des gaz doivent être séparées d'au moins 3 largeurs."),
    ("sketch_C_small.png", "2 · Le vrai sommet", None,
     "Sur le spectre réel (Voigt) : le sommet le mieux adapté, jamais l'aile d'une raie hors fenêtre."),
    ("sketch_D_small.png", "3 · Entre 1 et 20 %", "CLASSEMENT",
     "1 % = SNR requis (10) / SNR du spectre (1000). On classe par nombre de gaz dans la plage."),
]
for i, (img, head, tag, desc) in enumerate(rules):
    x = 0.6 + i * 4.1
    card(s, x, 1.9, 3.9, 4.6, fill=WHITE).line.color.rgb = rgb("D5DBE3")
    picture(s, FIG / img, x + 0.15, 2.0, w=3.6, h=2.65, alt=head)
    text(s, x + 0.25, 4.75, 3.45, 0.45, head, size=17, bold=True, font=HEAD)
    text(s, x + 0.25, 5.25, 3.45, 1.2, desc, size=14, color=MUTED)

# 4 -- verification ---------------------------------------------------------------------
s = content_slide(4, "Vérification · interférence, saturation, T",
                  "Pour les meilleures fenêtres : un filtre sur l'interférence, le reste est affiché",
                  "E : l'interférence est l'absorption de tous les autres gaz (atmosphère, isotopes, autres "
                  "gaz mesurés) à la position du pic, divisée par celle du gaz mesuré ; au-dessus de 5 % la "
                  "fenêtre est éliminée — seuil réglable comme tu le proposais. F : la saturation n'est plus "
                  "un filtre par défaut, mais reste affichée. G : coefficient en %/K, et pour un rapport "
                  "isotopique l'erreur en pour mille pour une stabilité de 0,1 K.")
cards = [
    ("sketch_E.png", "Interférence", "FILTRE",
     "Absorption des autres gaz au pic du gaz mesuré. Au-dessus de 5 % (réglable) : fenêtre éliminée."),
    ("sketch_F.png", "Saturation", "AFFICHÉ",
     "Transmission minimale sur tout le balayage. Filtre optionnel, désactivé par défaut."),
    ("sketch_G.png", "Température", "AFFICHÉ",
     "Variation du pic en %/K ; rapport isotopique : erreur en ‰ pour ±0,1 K."),
]
for i, (img, head, tag, desc) in enumerate(cards):
    x = 0.6 + i * 4.1
    card(s, x, 1.9, 3.9, 4.45, fill=WHITE).line.color.rgb = rgb("D5DBE3")
    picture(s, FIG / img, x + 0.15, 2.05, w=3.6, h=2.4, alt=head)
    text(s, x + 0.25, 4.6, 2.4, 0.45, head, size=17, bold=True, font=HEAD)
    chip(s, x + 2.55, 4.66, tag)
    text(s, x + 0.25, 5.1, 3.45, 1.55, desc, size=14, color=MUTED)

# 5 -- isotopes: two roles ---------------------------------------------------------------------
s = content_slide(5, "Isotopes · deux rôles",
                  "Comptés automatiquement comme interférents, ou mesurés comme un gaz à part entière",
                  "1 : pour chaque gaz, tous les isotopes répertoriés dans HITRAN sont comptés comme "
                  "interférents, à leur abondance naturelle — c'est par exemple HDO, dans 30 % de H2O, qui "
                  "sature près de certaines raies. Tu as confirmé que c'est suffisant. 2 : un isotope peut "
                  "aussi être un gaz à mesurer ; on donne la concentration de la molécule et l'abondance "
                  "HITRAN donne sa part (1,1 % pour 13CH4, donc environ 0,11 ppm dans 10 ppm de CH4) ; elle "
                  "est modifiable pour simuler un autre delta 13C. 3 : pour deux isotopes mesurés, le code "
                  "donne la sensibilité du rapport à la température, l'erreur en pour mille pour 0,1 K, et "
                  "signale les fenêtres au-delà de 1 pour mille.")
picture(s, FIG / "isotopes.png", 0.5, 1.95, w=6.0, h=3.85, alt="Schéma : raie de 12CH4 et de 13CH4")
text(s, 0.6, 5.95, 5.9, 0.8, "Même raie : ¹³CH4 est ~90× plus faible que ¹²CH4. Pour un δ¹³C, il faut des "
     "raies qui absorbent toutes les deux dans la plage.", size=12.5, color=MUTED)
iso_rows = [
    ("Comme interférent : automatique",
     "tous les isotopes HITRAN de chaque gaz, à leur abondance naturelle (HDO, H2¹⁸O, ¹³CH4…)"),
    ("Comme gaz à mesurer",
     "on choisit l'isotope ; concentration de la molécule, part donnée par l'abondance HITRAN "
     "(modifiable, ex. δ¹³C)"),
    ("Rapport isotopique et température",
     "coefficient du rapport en %/K ; erreur en ‰ pour ±0,1 K, signalée au-delà de 1 ‰"),
]
for i, (h1, h2) in enumerate(iso_rows):
    yy = 2.0 + i * 1.55
    badge(s, 6.95, yy, str(i + 1), d=0.55, size=16)
    text(s, 7.7, yy - 0.05, 5.0, 1.45, [[(h1, {"bold": True, "size": 17, "font": HEAD})],
                                        [(h2, {"color": MUTED})]], size=14.5, space_after=3)

# 6 -- the code: who does what ------------------------------------------------------------------
s = content_slide(6, "Le code · qui fait quoi",
                  "Deux façons de lancer, un moteur en huit fichiers, des sorties",
                  "Le plus simple pour tester : l'interface, « uv run streamlit run app.py », qui s'ouvre dans "
                  "le navigateur — gaz et isotopes, instrument, plage, critères, puis le tableau des fenêtres et "
                  "le spectre interactif de chacune. Sinon, modifier run.py et lancer « uv run run.py ». Le moteur est le dossier hitran_tool : les fichiers sont dans "
                  "l'ordre du workflow, windows.py est le cœur (étapes 1 et 2, classement, filtres). Les raies "
                  "HITRAN sont téléchargées une seule fois puis gardées dans data/.")
MONO = "Courier New"
columns = [
    (0.6, 3.3, "Tu lances", TEAL, [
        ("app.py", "l'interface : tout se règle dans le navigateur"),
        ("run.py", "les mêmes paramètres, dans un fichier"),
        ("studies/", "une étude = un script (ex. CH4 haute concentration)"),
        ("compare_variants.py", "rejoue les versions pour mesurer chaque changement"),
    ]),
    (10.05, 2.68, "Données / sorties", TEAL, [
        ("atm_standard.csv", "gaz de fond (42 molécules)"),
        ("data/", "cache des raies HITRAN"),
        ("results/", "tableaux CSV"),
        ("window_details/", "un graphique par fenêtre"),
    ]),
]
for x, w, head, color, items in columns:
    card(s, x, 1.9, w, 4.55)
    text(s, x + 0.25, 2.05, w - 0.4, 0.45, head, size=18, bold=True, font=HEAD, color=color)
    step = 3.75 / len(items)
    for i, (name, desc) in enumerate(items):
        text(s, x + 0.25, 2.65 + i * step, w - 0.4, step, [[(name, {"font": MONO, "bold": True, "size": 13})],
                                                          [(desc, {"color": MUTED, "size": 12.5})]],
             size=13, space_after=1)
engine = [
    ("species.py", "molécules, isotopes, unités (ppm, %)"),
    ("lines.py", "raies HITRAN, téléchargées une fois"),
    ("broadening.py", "largeurs : Doppler, pression, Voigt"),
    ("spectrum.py", "vrai spectre α(σ), profil de Voigt"),
    ("windows.py", "le cœur : étapes, classement, filtres"),
    ("interference.py", "interférents, interférence au pic"),
    ("temperature.py", "%/K, rapport isotopique"),
    ("report.py", "tableau, graphiques, CSV"),
]
card(s, 4.25, 1.9, 5.45, 4.55, fill="E6F0F2")
text(s, 4.5, 2.05, 5.2, 0.45, [[("Le moteur · ", {}), ("hitran_tool/", {"font": MONO, "size": 17})]],
     size=18, bold=True, font=HEAD, color=TEAL)
for i, (name, desc) in enumerate(engine):
    yy = 2.62 + i * 0.47
    core = name == "windows.py"
    text(s, 4.5, yy, 1.8, 0.42, name, size=12.5, bold=True, font=MONO, color=ACCENT if core else INK,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.3, yy, 3.3, 0.42, desc, size=13, color=INK if core else MUTED, bold=core,
         anchor=MSO_ANCHOR.MIDDLE)
for ax in (3.95, 9.74):
    shape(s, MSO_SHAPE.RIGHT_ARROW, ax, 3.95, 0.27, 0.4, ACCENT)
text(s, 0.6, 6.6, 12.1, 0.35, [[("Pour l'utiliser : ", {"bold": True, "color": TEAL}),
                                ("uv run streamlit run app.py", {"font": MONO, "bold": True}),
                                ("  (interface dans le navigateur), ou modifier run.py puis  ", {}),
                                ("uv run run.py", {"font": MONO, "bold": True})]], size=13.5)

# 7 -- what changed, and what it changes ------------------------------------------------------
changed, ref = comparison()


def severity(n):
    return BAD if n >= 3 else ACCENT if n >= 1 else "9AA4B1"


s = content_slide(7, "Ce qui a changé depuis le début",
                  "Des corrections motivées par des défauts observés, puis tes retours",
                  "À gauche, les corrections de la méthode avec leur effet mesuré : on rejoue le cas de "
                  "validation en ajoutant chaque correction et on compte combien des 10 meilleures fenêtres "
                  "changent. La lecture du pic était la plus importante : l'ancien numéro 1 était un artefact, "
                  "l'aile d'une raie hors fenêtre. À droite, ce qui a été ajouté, et ce qui a changé après ton "
                  "mail (seuil de détection, filtre d'interférence, saturation optionnelle, 40 °C). Le choix "
                  "de la raie par son absorption vient de l'étude A et change autant le top 10 que la lecture "
                  "du pic.")
fixes = [
    ("Pic lu à la raie, puis au sommet", "le max tombait sur l'aile d'une raie hors fenêtre", changed["V1"]),
    ("Ailes des raies sur 5 cm⁻¹", "avant coupées à ~0,3 cm⁻¹ (ailes de HDO saturé)", changed["V2"]),
    ("Sommet affiné, grille 5× plus fine", "pic sous-estimé entre deux points de calcul",
     max(changed["V3"], changed["V7"])),
    ("Écart à la plage mesuré sur τ", "trop faible ~40× moins pénalisé que trop fort", changed["V8"]),
    ("Raie choisie pour son absorption", "avant toujours la plus forte, qui peut saturer", changed["V9"]),
]
card(s, 0.6, 1.9, 6.0, 4.85)
text(s, 0.95, 2.05, 5.4, 0.45, "Corrections de la méthode", size=19, bold=True, font=HEAD, color=TEAL)
for i, (what, why, n) in enumerate(fixes):
    yy = 2.6 + i * 0.74
    pill = shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.95, yy + 0.05, 0.95, 0.42, severity(n), radius=0.5)
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run()
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    r.text = f"{n}/10"
    r.font.name, r.font.size, r.font.bold = BODY, Pt(13), True
    r.font.color.rgb = rgb(WHITE)
    text(s, 2.1, yy - 0.02, 4.35, 0.72, [[(what, {"bold": True})], [(why, {"color": MUTED, "size": 12.5})]],
         size=14.5, space_after=0)
text(s, 0.95, 6.38, 5.5, 0.32, "x/10 = fenêtres du top 10 qui changent avec cette correction", size=11,
     color=MUTED)
adds = [
    ("Interférences", "42 gaz de tes slides, isotopes HITRAN, autres gaz mesurés"),
    ("Température", "%/K par gaz ; rapport isotopique en ‰"),
    ("Isotopes", "un isotope peut être un gaz à mesurer"),
    ("Seuil de détection 1 %", "SNR requis 10 / SNR du spectre 1000 (ton mail)"),
    ("Filtre d'interférence 5 %", "réglable (ton mail) ; saturation optionnelle ; 40 °C"),
]
card(s, 6.75, 1.9, 5.95, 4.85)
text(s, 7.1, 2.05, 5.3, 0.45, "Ajouts et suite à ton mail", size=19, bold=True, font=HEAD, color=ACCENT)
for i, (what, why) in enumerate(adds):
    yy = 2.7 + i * 0.8
    badge(s, 7.1, yy + 0.08, str(i + 1), d=0.38, size=12, fill=ACCENT)
    text(s, 7.65, yy, 4.9, 0.66, [[(what, {"bold": True, "size": 14}), ("  " + why, {"color": MUTED, "size": 12.5})]],
         size=14, anchor=MSO_ANCHOR.MIDDLE, space_after=0)

# 8 -- validation ---------------------------------------------------------------------------------
s = content_slide(8, "Validation · la fenêtre SubOcean retrouvée",
                  "Avec des critères réalistes, le code retombe sur le choix des expérimentateurs",
                  "Le graphique : la fenêtre réelle de SubOcean telle que le code la trouve, avec le pic "
                  "retenu de chaque gaz et son absorption. Dans les 7 cm-1 du laser DFB, c'est la seule des "
                  "8 fenêtres candidates sous 5 % d'interférence. Le tableau montre pourquoi elle remonte : "
                  "surtout le seuil de détection à 1 % (N2O et H2O comptent alors), puis le filtre "
                  "d'interférence qui élimine la seule fenêtre devant elle.")
picture(s, FIG / "subocean_window.png", 0.5, 1.9, w=6.6, h=4.1, alt="Fenêtre SubOcean calculée par le code")
text(s, 0.6, 6.15, 6.5, 0.6, "Fenêtre 4397,72–4398,27 cm⁻¹ à 40 °C : les trois gaz dans la plage, "
     "interférence au plus 2,6 %", size=12, color=MUTED)
for i, (big, small) in enumerate([("1re", "dans les 7 cm⁻¹ du laser DFB — seule des 8 fenêtres sous 5 % "
                                          "d'interférence"),
                                   ("7e", "sur toute la gamme 2000–9000 cm⁻¹ (avant : après la 30e place)")]):
    yy = 1.9 + i * 1.05
    text(s, 7.5, yy, 1.4, 0.95, big, size=40, bold=True, color=ACCENT, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.95, yy, 3.8, 0.95, small, size=13.5, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
trace = [("Réglage", "Rang", "Gaz dans la plage"),
         ("Avant : 296 K, plage 5–20 %", "2e / 8", "1 / 3"),
         ("+ seuil de détection 1 %", "3e / 8", "3 / 3"),
         ("+ 40 °C", "2e / 8", "3 / 3"),
         ("+ filtre interférence 5 %", "1re / 1", "3 / 3")]
tbl = s.shapes.add_table(len(trace), 3, Inches(7.5), Inches(4.0), Inches(5.2), Inches(2.45)).table
for j, wd in enumerate([2.5, 1.0, 1.7]):
    tbl.columns[j].width = Inches(wd)
for i, row in enumerate(trace):
    for j, val in enumerate(row):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = rgb(TEAL if i == 0 else (LIGHT if i % 2 else WHITE))
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = c.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = nb(val)
        r.font.name, r.font.size = BODY, Pt(13)
        r.font.bold = i == 0 or (i == len(trace) - 1 and j > 0)
        r.font.color.rgb = rgb(WHITE if i == 0 else INK)
text(s, 7.5, 6.55, 5.2, 0.3, "rang dans les 7 cm⁻¹ du laser, en ajoutant les réglages un par un", size=11,
     color=MUTED)

# 9 -- study A ------------------------------------------------------------------------------------
s = content_slide(9, "Étude A · CH4 à 1000 ppm et plus",
                  "Avec ce laser, la même détection devient impossible quand CH4 dépasse ~1000 ppm",
                  "Ta demande : autour de la fenêtre actuelle, dans les 7 cm-1 d'accord du laser DFB, peut-on "
                  "faire la même détection avec 1000 ou 10 000 ppm de CH4 ? À 10 ppm, la fenêtre SubOcean "
                  "sort première. À 1000 ppm, CH4 lui-même reste mesurable sur des raies plus faibles, mais "
                  "ses raies et celles de 13CH4 couvrent tous les pics de N2O et de H2O : près de 300 raies "
                  "de CH4 absorbent plus de 1 % sur ces 7 cm-1. Dans la meilleure fenêtre, l'interférence "
                  "reste d'au moins 60 % ; aucune fenêtre ne passe le filtre.")
picture(s, FIG / "high_ch4.png", 0.5, 1.85, w=8.1, h=4.55, alt="Épaisseur optique de CH4, N2O et H2O sur 7 cm-1")
text(s, 0.6, 6.45, 8.0, 0.4, "Épaisseur optique τ = α·L à 40 °C ; bande verte : plage visée (1–20 % d'absorption)",
     size=12, color=MUTED)
facts = [
    ("10 ppm", GOOD, "la fenêtre SubOcean sort 1re"),
    ("1000 ppm", ACCENT, "CH4 reste mesurable, sur des raies plus faibles (I/I0 0,83–0,94)"),
    ("0 fenêtre", BAD, "CH4 + ¹³CH4 sous chaque pic de N2O et H2O : interférence d'au moins 60 %, souvent > 1000 %"),
]
for i, (big, color, small) in enumerate(facts):
    yy = 1.9 + i * 1.62
    card(s, 8.85, yy, 3.85, 1.45)
    text(s, 9.1, yy + 0.1, 3.4, 0.6, big, size=26, bold=True, color=color, font=HEAD)
    text(s, 9.1, yy + 0.7, 3.45, 0.75, small, size=13.5, color=MUTED)

# 10 -- questions --------------------------------------------------------------------------------------
s = dark_slide("Les questions à discuter, de la plus importante à la moins urgente. La 1 : la valeur de 3,8 % "
               "que tu as obtenue correspond, chez nous, à la raie de N2O à 4397,997 ; H2O y est à 1,8 % à 40 °C "
               "— vérifier quelle raie et quelle longueur L. Ensuite l'étude B (SWIS) attend L et la fraction "
               "d'eau.")
text(s, 0.7, 0.5, 12.0, 0.9, "Questions pour aujourd'hui", size=36, bold=True, color=WHITE, font=HEAD,
     anchor=MSO_ANCHOR.MIDDLE)
questions = [
    ("Ta simulation à 3,8 % : quelle longueur L ?",
     "chez nous, 3,8 % c'est N2O à 4397,997 ; H2O est à 1,8 % à 40 °C"),
    ("SWIS : longueur L (ringdown) et fraction d'eau", "nécessaires pour lancer l'étude B (isotopes de l'eau)"),
    ("Seuil d'interférence : 5 % ?", "à 10 %, il resterait 3 fenêtres au lieu d'une près de 4398 cm⁻¹"),
    ("Départager les fenêtres où tous les gaz sont dans la plage",
     "par le score d'interférence ? par la sensibilité à T ?"),
    ("CH4 à haute concentration", "chercher dans une autre région spectrale, avec un autre laser ?"),
]
for i, (q, d) in enumerate(questions):
    yy = 1.65 + i * 0.98
    badge(s, 0.75, yy + 0.05, str(i + 1), d=0.55, size=17, fill=ACCENT)
    text(s, 1.55, yy, 7.2, 0.95, [[(q, {"bold": True, "color": WHITE, "size": 17})],
                                  [(d, {"color": SOFT, "size": 13.5})]], size=17, space_after=2)
card(s, 9.2, 1.65, 3.5, 3.3, fill="2A3242")
text(s, 9.5, 1.85, 3.0, 0.5, "Prochaines étapes", size=19, bold=True, color=ACCENT, font=HEAD)
text(s, 9.5, 2.5, 2.95, 3.8, ["Étude B : isotopes de l'eau (SWIS), dès que L est connu",
                              "Départager les fenêtres ex aequo",
                              "Rapport d'étude : SubOcean et SWIS"], size=15, color=WHITE, space_after=12)

prs.save(OUT)
print(f"saved {OUT}")
