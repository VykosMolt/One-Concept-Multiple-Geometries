"""Concept families with fixed, recorded orderings. NEVER reorder to improve a spectrum."""
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# semitone order, x = 0..11 (C=0)
PC_CANON_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]   # brief's canonical set
PC_CANON_MINOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]   # conventional minor spellings
PC_ALL_SHARP   = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC_ALL_FLAT    = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PC_WORDS_MAJOR = ["C", "D-flat", "D", "E-flat", "E", "F", "F-sharp", "G", "A-flat", "A", "B-flat", "B"]
PC_UNICODE_MAJOR = ["C", "D♭", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]

FAMILIES = {
    "months":   {"concepts": MONTHS,   "templates": ["The month of the year is {x}", "{x}", "It happened in {x}, which", "The month of the year is {x}."]},
    "weekdays": {"concepts": WEEKDAYS, "templates": ["The day of the week is {x}", "{x}", "It happened on {x}, which"]},
    "major_canon": {"concepts": PC_CANON_MAJOR, "templates": [
        "The piece is written in the key of {x} major",
        "The piece is written in the key of {x} major.",
        "{x} major",
        "Symphony in {x} major, which",
        "The key is {x} major, and the piece",
        "In music, the key of {x} major has a key signature of",
    ]},
    "minor_canon": {"concepts": PC_CANON_MINOR, "templates": [
        "The piece is written in the key of {x} minor",
        "The piece is written in the key of {x} minor.",
        "{x} minor",
        "Symphony in {x} minor, which",
    ]},
    "major_sharp": {"concepts": PC_ALL_SHARP, "templates": ["The piece is written in the key of {x} major", "{x} major"]},
    "major_flat":  {"concepts": PC_ALL_FLAT,  "templates": ["The piece is written in the key of {x} major", "{x} major"]},
    "major_words": {"concepts": PC_WORDS_MAJOR, "templates": ["The piece is written in the key of {x} major"]},
    "major_unicode": {"concepts": PC_UNICODE_MAJOR, "templates": ["The piece is written in the key of {x} major"]},
    "tonic_canon": {"concepts": PC_CANON_MAJOR, "templates": ["The note is {x}", "The tonic of the scale is {x}", "The musical note {x}, which"]},
    "chord_major": {"concepts": PC_CANON_MAJOR, "templates": ["The chord is {x} major", "Play a {x} major chord, then"]},
}
