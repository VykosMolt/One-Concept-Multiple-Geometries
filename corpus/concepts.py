"""Concept families and regexes. All matching is case-sensitive on raw text.

Pitch-class index (semitone): C=0, C#/Db=1, D=2, D#/Eb=3, E=4, F=5, F#/Gb=6, G=7, G#/Ab=8, A=9, A#/Bb=10, B=11.
Every spelling is counted separately; aggregation to pitch classes happens at analysis time so
that enharmonic choices are explicit and recorded (see analysis code / RESEARCH_LOG).
"""
import re

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_FORMS = ["#", "\u266f", "-sharp", " sharp"]
FLAT_FORMS = ["b", "\u266d", "-flat", " flat"]

# spelling name -> pitch class
SPELLINGS = {}
for L, pc in LETTER_PC.items():
    SPELLINGS[L] = pc
    SPELLINGS[L + "#"] = (pc + 1) % 12
    SPELLINGS[L + "b"] = (pc - 1) % 12

# canonical 12-key set in chromatic order (major keys), as given in the brief
CANON_MAJOR = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# canonical minor set in chromatic order (conventional spellings for minor keys)
CANON_MINOR = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

def _acc_alternation():
    s = "|".join(re.escape(a) for a in SHARP_FORMS)
    f = "|".join(re.escape(a) for a in FLAT_FORMS)
    return s, f

_S, _F = _acc_alternation()
# group 'letter', 'acc', 'mode'
KEY_RE = re.compile(r"(?<![A-Za-z0-9\-])(?P<letter>[A-G])(?P<acc>%s|%s)?[ \u00a0](?P<mode>major|minor)(?![A-Za-z])" % (_S, _F))
MONTH_RE = re.compile(r"(?<![A-Za-z])(?P<month>%s)(?![A-Za-z])" % "|".join(MONTHS))
WEEKDAY_RE = re.compile(r"(?<![A-Za-z])(?P<weekday>%s)(?![A-Za-z])" % "|".join(WEEKDAYS))
# note names in explicit note contexts (bonus family, noisy): "the note C", "note of C#"
NOTE_RE = re.compile(r"(?<![A-Za-z0-9\-])(?:note|notes|pitch|key of|tonic|dominant)[ ](?:of[ ])?(?P<letter>[A-G])(?P<acc>%s|%s)?(?![A-Za-z0-9#\u266f\u266d])" % (_S, _F))

COF_PHRASES = ["circle of fifths", "cycle of fifths", "perfect fifth", "perfect fifths", "circle of fourths"]

def normalize_acc(acc):
    if acc is None: return ""
    if acc in SHARP_FORMS: return "#"
    if acc in FLAT_FORMS: return "b"
    raise ValueError(acc)

def key_name(m):
    return m.group("letter") + normalize_acc(m.group("acc"))

SENTENCE_INITIAL_PREV = set(".!?\n:\"'\u201c(\u2018[")

def is_sentence_initial(text, start):
    """True if match is at text start or preceded (ignoring spaces) by a sentence-final / opening char."""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or text[i] in SENTENCE_INITIAL_PREV
