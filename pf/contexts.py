"""Diverse natural contexts for key names. Every context has a whitespace-separated word right after
the key name (the mode word), so the anchor token is identical across the 12 concepts."""
MAJOR_CONTEXTS = [
    "The piece is written in the key of {x} major",
    "The symphony is in {x} major and",
    "She composed the sonata in {x} major for",
    "The second movement modulates to {x} major before",
    "A prelude in {x} major opens",
    "The song was recorded in {x} major with",
    "This etude is in {x} major and requires",
    "The concerto in {x} major was premiered",
    "Transpose the melody to {x} major so",
    "The key signature of {x} major has",
    "The scale of {x} major contains",
    "The chorus shifts to {x} major while",
    "The quartet in {x} major is",
    "Most of the album is in {x} major or",
    "The hymn is usually sung in {x} major by",
    "The organ fugue in {x} major begins",
    "He improvised in {x} major over",
    "The finale returns to {x} major after",
    "The mass in {x} major dates from",
    "The overture in {x} major opens with",
    "The exposition stays in {x} major until",
    "The tune is in {x} major, which",
    "Written in {x} major, the piece",
    "The string quintet in {x} major was",
]
MINOR_CONTEXTS = [c.replace(" major", " minor") for c in MAJOR_CONTEXTS]
