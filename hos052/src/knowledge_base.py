"""
Plant disease knowledge base and rule-based farming chatbot.

Contains:
  - KB: information about 26 plant diseases (cause, symptoms, spread)
  - Plant-care suggestions and prevention checklists for each disease type
  - diagnosis_report(): builds a readable report for a predicted disease
  - chat_answer(): answers simple farming questions using keyword matching
"""

# ---------------------------------------------------------------------------
# Raw disease data
# Each row: (plant, key, title, type, cause, symptoms, how it spreads)
# "plant" and "key" are matched against the model's class labels.
# ---------------------------------------------------------------------------
KB_RAW = [
 ("apple","apple_scab","Apple Scab","fungal","Venturia inaequalis (fungus)","Olive-green to brown velvety spots on leaves; rough, cracked scabby spots on fruit","Spores from fallen infected leaves spread by wind and rain splash in spring; worse in cool, wet weather"),
 ("apple","black_rot","Apple Black Rot","fungal","Botryosphaeria obtusa (fungus)","Small round leaf spots with purple edges (frog-eye spots); fruit rots and turns black and dried","Spores from infected branches, wounds and mummified fruit spread by rain and wind"),
 ("apple","cedar","Cedar Apple Rust","fungal","Gymnosporangium juniperi-virginianae (fungus); its life cycle needs a juniper or cedar tree nearby","Bright yellow-orange spots on leaves, later small tube-like structures under the spots","Windborne spores travel from juniper/cedar trees to apple trees in spring"),
 ("cherry","powdery_mildew","Cherry Powdery Mildew","fungal","Podosphaera clandestina (fungus)","White powdery coating on leaves; leaves may curl","Windborne spores; grows in moderate temperatures and humid conditions"),
 ("corn","cercospora","Corn Gray Leaf Spot","fungal","Cercospora zeae-maydis (fungus)","Rectangular tan to gray lesions between the leaf veins","Spores from old crop residue spread by wind and rain splash; worse in humid weather"),
 ("corn","common_rust","Corn Common Rust","fungal","Puccinia sorghi (fungus)","Small reddish-brown pustules on both sides of the leaf","Windborne spores; worse in cool, wet weather"),
 ("corn","northern","Corn Northern Leaf Blight","fungal","Exserohilum turcicum (fungus)","Long cigar-shaped gray-green to brown lesions on leaves","Spores from crop residue spread by wind and rain; favoured by moderate temperatures and humidity"),
 ("grape","black_rot","Grape Black Rot","fungal","Guignardia bidwellii (fungus)","Brown round leaf spots with dark borders; berries shrivel into hard black mummies","Spores from mummified berries and old vines spread by rain"),
 ("grape","esca","Grape Esca (Black Measles)","fungal","A complex of several wood-attacking fungi","Tiger-stripe yellow-brown patches on leaves; small dark spots on berries","Spores enter through pruning wounds; the fungi can stay in vines for years"),
 ("grape","isariopsis","Grape Leaf Blight (Isariopsis Leaf Spot)","fungal","Pseudocercospora vitis (fungus)","Irregular dark brown leaf spots; leaves may drop early","Spores from infected leaves spread by wind and rain in humid conditions"),
 ("orange","greening","Orange Citrus Greening (Huanglongbing)","hlb","Candidatus Liberibacter bacteria","Uneven yellow blotches on leaves; small, lopsided, bitter fruit; tree slowly declines","Spread by the Asian citrus psyllid insect and by infected grafted plants"),
 ("peach","bacterial_spot","Peach Bacterial Spot","bacterial","Xanthomonas arboricola pv. pruni (bacteria)","Small angular water-soaked leaf spots that later fall out as holes; pitted spots on fruit","Spread by rain splash and wind; survives on infected twigs"),
 ("pepper","bacterial_spot","Pepper Bacterial Spot","bacterial","Xanthomonas species (bacteria)","Small water-soaked leaf spots that turn brown; raised rough spots on fruit","Infected seed and seedlings, rain splash, and hands or tools during work"),
 ("potato","early_blight","Potato Early Blight","fungal","Alternaria solani (fungus)","Brown spots with concentric rings (target-like) on older leaves","Spores from crop residue and soil spread by wind and rain; worse on stressed plants"),
 ("potato","late_blight","Potato Late Blight","oomycete","Phytophthora infestans (fungus-like organism)","Dark water-soaked leaf spots with white mold underneath; brown rot in tubers","Spreads very fast by wind in cool, wet weather; infected tubers can also carry it"),
 ("squash","powdery_mildew","Squash Powdery Mildew","fungal","Podosphaera xanthii (fungus)","White powdery patches on leaves; leaves later dry out","Windborne spores; worse in crowded, humid conditions"),
 ("strawberry","leaf_scorch","Strawberry Leaf Scorch","fungal","Diplocarpon earlianum (fungus)","Small purple spots on leaves; leaves later look scorched","Rain splash and infected leaves or plants"),
 ("tomato","bacterial_spot","Tomato Bacterial Spot","bacterial","Xanthomonas species (bacteria)","Small dark water-soaked spots on leaves and fruit; rough scabby spots on fruit","Infected seed and seedlings, rain splash, and hands or tools"),
 ("tomato","early_blight","Tomato Early Blight","fungal","Alternaria solani (fungus)","Brown spots with concentric rings (target-like) on lower, older leaves; stems and fruit may also show spots","Spores from crop residue and soil spread by wind and rain; worse on stressed plants"),
 ("tomato","late_blight","Tomato Late Blight","oomycete","Phytophthora infestans (fungus-like organism)","Large dark water-soaked patches on leaves and stems; fruit becomes firm, brown and rots","Spreads very fast by wind in cool, wet weather"),
 ("tomato","leaf_mold","Tomato Leaf Mold","fungal","Passalora fulva (fungus)","Yellow patches on the upper leaf surface with olive-brown mold underneath","Windborne spores in high humidity, especially in greenhouses"),
 ("tomato","septoria","Tomato Septoria Leaf Spot","fungal","Septoria lycopersici (fungus)","Many small round spots with dark edges and gray centers containing tiny black dots","Rain splash and crop residue; usually starts on the lower leaves"),
 ("tomato","spider_mites","Tomato Two-spotted Spider Mite","mite","Two-spotted spider mite (Tetranychus urticae), a tiny pest","Tiny yellow speckles on leaves, fine webbing; leaves dry out and drop","Multiplies quickly in hot, dry weather; spreads by wind, clothing and infested plants"),
 ("tomato","target_spot","Tomato Target Spot","fungal","Corynespora cassiicola (fungus)","Brown spots with concentric rings on leaves; fruit may also show spots","Spores from crop residue spread in humid weather"),
 ("tomato","yellow_leaf_curl","Tomato Yellow Leaf Curl Virus (TYLCV)","viral","Begomovirus (TYLCV)","Leaves turn yellow and curl upward; plants are stunted and yield drops","Spread by whiteflies; infected seedlings can also carry it"),
 ("tomato","mosaic_virus","Tomato Mosaic Virus (ToMV)","viral","Tomato mosaic virus (a tobamovirus)","Light and dark green mosaic-like mottling; distorted leaves","Spreads easily by hands, tools, infected seed and plant sap; the virus survives for a long time"),
]

# Readable names for each disease type
TYPE_NAME = {"fungal":"Fungal disease","oomycete":"Oomycete (fungus-like) disease","bacterial":"Bacterial disease",
             "viral":"Viral disease","mite":"Mite (pest) damage","hlb":"Bacterial disease (insect-vectored)"}

# ---------------------------------------------------------------------------
# Prevention checklists and plant-care suggestions, grouped by disease type
# ---------------------------------------------------------------------------
_FUNGAL_PREV = ["Use disease-free seed or seedlings and resistant varieties where possible",
                "Collect and remove fallen infected leaves and crop residue from the field",
                "Keep enough space between plants and improve air flow",
                "Water at the base of the plant, not over the leaves, and preferably in the morning",
                "Rotate crops instead of growing the same crop in the same field every season",
                "Use balanced fertilizer and avoid too much nitrogen",
                "Inspect the tops and undersides of leaves regularly, at least 2-3 times a week"]
PREVENT = {
 "fungal": _FUNGAL_PREV,
 "oomycete": _FUNGAL_PREV + ["Watch the weather forecast closely for cool, wet periods", "Use healthy, certified seed tubers or planting material"],
 "bacterial": ["Use disease-free seed or seedlings",
               "Avoid working in the field or touching plants when they are wet",
               "Clean hands and tools before and after work",
               "Avoid overhead (sprinkler) irrigation and water at the base",
               "Remove and destroy infected plant parts and rotate crops",
               "Do not leave crop residue in the field"],
 "viral": ["Use disease-free seed or seedlings and resistant varieties where possible",
           "Control insect vectors (such as whiteflies and aphids) and use nets in nurseries",
           "Remove and destroy infected plants so the virus does not spread",
           "Clean hands and tools before and after work",
           "Keep the field and its surroundings free of weeds"],
 "mite": ["Check the underside of leaves regularly",
          "Reduce dust and water stress by irrigating properly",
          "Remove heavily infested leaves",
          "Protect beneficial insects (natural enemies of mites)",
          "Inspect new seedlings before bringing them into the field"],
 "hlb": ["Use only certified disease-free planting material",
         "Monitor and control psyllid insects regularly",
         "Report suspicious trees to the local agricultural authority",
         "Follow official government guidance for infected trees"],
}
CARE = {
 "fungal": ["Carefully cut off and destroy infected leaves or parts (do not put them in compost)",
            "Avoid keeping leaves wet and improve air flow",
            "Check every day whether the disease is spreading to other plants"],
 "bacterial": ["Remove and destroy infected leaves or parts",
               "Do not handle plants when they are wet and clean tools after work",
               "Stop overhead irrigation"],
 "viral": ["There is no cure for plant viruses; remove and destroy severely infected plants",
           "Keep insect vectors (whiteflies, aphids) under control",
           "Clean hands and tools before touching healthy plants"],
 "mite": ["Remove heavily infested leaves",
          "Reduce water stress and dust around the plants",
          "Keep checking the underside of leaves"],
 "hlb": ["There is no cure for this disease; if you suspect it, report it to the local agricultural authority quickly",
         "Do not move trees or planting material elsewhere without official guidance"],
}
CARE["oomycete"] = CARE["fungal"] + ["Risk is higher in cool, wet weather and the disease can spread very fast, so contact a local agricultural officer quickly"]

# The system never suggests specific chemicals or dosages (safety)
SAFETY = ("**Safety note:** This system does not guess specific pesticide names or dosages. "
          "Before using any chemical, follow approved agricultural guidance or ask your local agricultural extension officer.")

# ---------------------------------------------------------------------------
# Keyword lists used by the chatbot
# ---------------------------------------------------------------------------
ALIAS_EXTRA = {
 "cercospora":["gray leaf spot","grey leaf spot"], "northern":["northern leaf blight"],
 "common_rust":["common rust","rust"], "cedar":["cedar apple rust","cedar","rust"],
 "greening":["citrus greening","huanglongbing","hlb"], "spider_mites":["spider mite","mite"],
 "yellow_leaf_curl":["yellow leaf curl","tylcv","leaf curl"], "mosaic_virus":["mosaic"],
 "leaf_mold":["leaf mold","leaf mould"], "powdery_mildew":["powdery mildew"],
 "bacterial_spot":["bacterial spot"], "apple_scab":["scab"], "black_rot":["black rot"],
 "esca":["black measles"], "isariopsis":["grape leaf blight"], "target_spot":["target spot"],
 "leaf_scorch":["leaf scorch","scorch"],
}
PLANT_WORDS = {
 "tomato":["tomato"], "potato":["potato"], "apple":["apple"], "grape":["grape"], "corn":["corn","maize"],
 "pepper":["pepper","capsicum"], "peach":["peach"], "cherry":["cherry"], "squash":["squash"],
 "strawberry":["strawberry"], "orange":["orange","citrus"],
}
INTENTS = {
 "prevent":["prevent","avoid","protect","stop it","prevention"],
 "symptom":["symptom","sign","look like","identify","recognize","recognise"],
 "cause":["cause","why does","why do","pathogen"],
 "spread":["spread","transmit","transmission"],
 "treat":["treat","cure","what to do","manage","remedy","care for"],
 "chem":["pesticide","fungicide","bactericide","insecticide","spray","dosage","dose","chemical","medicine"],
}

# Build the searchable knowledge base list
KB = []
for plant, key, title, typ, cause, sym, spr in KB_RAW:
    KB.append(dict(plant=plant, key=key, title=title, type=typ, cause=cause, symptoms=sym, spread=spr,
                   aliases=[key.replace("_", " ")] + ALIAS_EXTRA.get(key, [])))

HEALTHY = dict(title="Healthy leaf", type="healthy")


def pretty(label):
    """Convert a raw class label to a readable one, e.g. 'Tomato___Late_blight' -> 'Tomato - Late blight'."""
    return label.replace("___", " - ").replace("_", " ").strip()


def get_info(label):
    """Find the knowledge base entry that matches a class label (None if not found)."""
    l = label.lower()
    for e in KB:
        if e["plant"] in l and e["key"] in l:
            return e
    return HEALTHY if l.endswith("healthy") else None


def diagnosis_report(label, conf=None):
    """Build a markdown report (disease info, care suggestions, prevention checklist)."""
    e = get_info(label)
    if e is None:
        return f"**{pretty(label)}**\n\nNo information about this disease is available in the knowledge base."
    head = f"## {pretty(label)}" + (f"  (confidence: {conf*100:.0f}%)" if conf is not None else "")
    if e["type"] == "healthy":
        return (head + "\n\nThe leaf looks healthy. Keep monitoring regularly, check the underside of leaves, "
                "and keep watering and fertilizing balanced.")
    # Warn the user when the model is not confident
    low = conf is not None and conf < 0.5
    out = [head]
    if low:
        out.append("**The model is not sure about this result.** Please upload a clearer, closer photo and consult a local agricultural officer before deciding.")
    out.append(f"### Disease information\n**Disease:** {e['title']} ({TYPE_NAME[e['type']]})\n\n"
               f"**Cause:** {e['cause']}\n\n**Symptoms:** {e['symptoms']}\n\n**How it spreads:** {e['spread']}")
    out.append("### General plant-care suggestions\n" + "\n".join("- " + x for x in CARE[e["type"]]))
    out.append("### Prevention checklist\n" + "\n".join("- [ ] " + x for x in PREVENT[e["type"]]))
    out.append(SAFETY)
    return "\n\n".join(out)


def detect_intents(t):
    """Find what the user is asking about (prevention, symptoms, cause, ...)."""
    return {k for k, ws in INTENTS.items() if any(w in t for w in ws)}


def supported_list():
    """Return all supported disease names as one text line."""
    return ", ".join(sorted({e["title"] for e in KB}))


def chat_answer(msg, last_label=None):
    """Answer a farming question by matching plant, disease and intent keywords."""
    t = msg.lower()
    plants = [p for p, ws in PLANT_WORDS.items() if any(w in t for w in ws)]
    intents = detect_intents(t)
    hits = [e for e in KB if any(a in t for a in e["aliases"])]
    if hits and plants:
        f = [e for e in hits if e["plant"] in plants]
        if not f:
            return "I do not have information about that disease for this crop. Supported diseases: " + supported_list()
        hits = f
    # If no disease is named, use the disease from the last diagnosis
    if not hits and not plants and last_label and intents:
        e = get_info(last_label)
        if e and e["type"] != "healthy":
            hits = [e]
    # Chemical / dosage questions always get the safety answer, even without a disease name
    if not hits and "chem" in intents:
        return ("I cannot recommend specific pesticides or dosages, because a wrong choice can harm "
                "your crop, your health and the environment. Please follow approved guidance from your "
                "local agricultural extension officer.\n\nI can still explain the cause, symptoms, spread, "
                "prevention and general care of a disease. Just ask with a disease name.")
    if not hits and plants:
        own = [e["title"] for e in KB if e["plant"] in plants]
        if own:
            return "For this crop I can answer about: " + "; ".join(own) + ". Which one would you like to know about?"
    if not hits:
        return ("Sorry, your question does not match the diseases I support. "
                "Please ask with a disease name (for example: How can I prevent Early Blight?).\n\nSupported diseases: " + supported_list())
    parts = []
    for e in hits[:3]:
        p = [f"**{e['title']}**"]
        if "chem" in intents:
            p.append("I cannot recommend specific pesticides or dosages. Please follow approved guidance from your local agricultural extension officer.")
        if "cause" in intents or not intents:
            p.append(f"Cause: {e['cause']}")
        if "symptom" in intents or not intents:
            p.append(f"Symptoms: {e['symptoms']}")
        if "spread" in intents:
            p.append(f"How it spreads: {e['spread']}")
        if "prevent" in intents or "chem" in intents:
            p.append("Prevention:\n" + "\n".join("- " + x for x in PREVENT[e["type"]]))
        if "treat" in intents or "chem" in intents:
            p.append("General care:\n" + "\n".join("- " + x for x in CARE[e["type"]]))
        parts.append("\n\n".join(p))
    return "\n\n---\n\n".join(parts)
