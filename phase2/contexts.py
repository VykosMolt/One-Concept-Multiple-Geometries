"""Balanced factorial context families for the 15-key experiments. Every template contains '{x}' followed by ' major'
and at least one more word so that the key token is never prompt-final; family E/C/D/B templates END at a position
whose next token should be a key name (predicting position). Written BEFORE any Phase-II result was seen."""
FAMILIES = {
    "A_spelling": [
        "The key signature of {x} major has",
        "In {x} major, the accidentals written in the key signature are",
        "Spelled correctly, the scale of {x} major uses the notes",
        "The number of sharps or flats in the key signature of {x} major is",
    ],
    "B_enharmonic": [
        "On the piano, the tonic of {x} major is the same key as the tonic of",
        "In equal temperament, {x} major sounds identical to",
        "The enharmonic equivalent of {x} major is",
        "Played on a keyboard, a piece in {x} major uses the same keys as a piece in",
    ],
    "C_harmonic": [
        "The dominant key of {x} major is",
        "The subdominant key of {x} major is",
        "A key closely related to {x} major is",
        "The relative minor of {x} major shares its key signature with",
    ],
    "D_chord": [
        "In the key of {x} major, the tonic chord is usually followed by the chord of",
        "A typical chord progression in {x} major moves from the tonic chord to the chord of",
        "In {x} major, the perfect cadence resolves from the dominant chord to the chord of",
        "Improvising over a song in {x} major, the guitarist played the chord of",
    ],
    "E_modulation": [
        "The piece modulates from {x} major to",
        "The first movement is in {x} major and the second movement is in",
        "The song, originally in {x} major, was later transposed to",
        "The first song on the album is in {x} major. The second song is in",
    ],
    "F_generic": [
        "She said the piece was in {x} major and everyone",
        "The recording of the sonata in {x} major was released in",
        "Most listeners preferred the version in {x} major because",
        "The manuscript of the symphony in {x} major is kept in",
    ],
}
