# Product Requirements: Tapped Rhythm Analysis UX

## 1. User Personas

### 1A. Beginner Drummer ("What am I playing?")
- Taps on a table, desk, or practice pad
- Knows "4/4" and "3/4" from basic lessons, but not much beyond that
- Wants a simple answer: "You're playing in 3" or "You're in 4"
- Does NOT know what "numerator", "denominator", or "hypothesis" means
- Likely to be confused by 2/4 when they think they're playing 4/4

### 1B. Experienced Musician ("Am I in 7/8 or 4/4?")
- Already knows music theory vocabulary
- Wants precise, detailed output including grouping (2+2+3 vs 3+2+2)
- May intentionally play odd meters and wants the tool to confirm
- Values accuracy and confidence scores; can interpret ambiguity
- Wants to see competing hypotheses and why the engine chose one

### 1C. Music Teacher ("Show the class what 5/4 feels like")
- Uses the tool as a teaching aid
- Needs the metronome -> analyze -> verify loop to be seamless
- Wants to demonstrate: play pattern, show result, play metronome to confirm
- Appreciates the educational descriptions (song examples like "Take Five")

### 1D. Casual Listener ("What's the time signature of this song?")
- Claps or taps along to recorded music
- Their tapping will be less precise (following a song, not performing)
- Mostly expects 4/4 or 3/4; will be surprised by anything exotic
- Needs a plain-language explanation if the result is unusual

---

## 2. Input Characteristics: What Makes Tapped Rhythm Special

### Signal properties
- **Pure percussion**: No pitch, harmony, or melody. Only onset timing + relative volume.
- **Accent = volume**: The ONLY signal for meter detection is which taps are louder. This means the RMS-based accent detection (which already works well per MEMORY.md) is the right approach.
- **Short duration**: Users will typically tap for 5-15 seconds. The engine needs to work with as few as 4 bars.
- **Imprecise timing**: Human groove means beats won't be machine-perfect. Timing tolerance of +/-30ms is realistic for trained musicians, +/-80ms for beginners.
- **Background noise**: Room reverb, table resonance, other sounds. The onset detector needs a minimum energy threshold.
- **Warm-up beats**: The first 2-4 taps are often irregular as the user finds their groove. The engine should discard or down-weight the first ~2 seconds.

### Minimum input requirements
| Meter | Minimum bars needed | Minimum duration at 120 BPM |
|-------|--------------------|-----------------------------|
| 4/4   | 3 bars             | 6 seconds                   |
| 3/4   | 4 bars             | 6 seconds                   |
| 5/4   | 3 bars             | 7.5 seconds                 |
| 7/8   | 4 bars             | 7 seconds                   |

**Recommendation**: Require at least 6 seconds of input with at least 8 detected onsets before attempting meter analysis. Show "Keep tapping..." until this threshold is met.

---

## 3. Output Design

### 3A. Primary result: Simple, human-readable

**Current state**: The app shows "4/4" with a description like "Standard rock/pop (e.g. Billie Jean, Hey Jude)" and a confidence percentage. This is good for the experienced user but needs refinement for beginners.

**Proposed tiered display**:

```
+------------------------------------------+
|                                          |
|              4/4                         |  <-- Big, bold signature
|                                          |
|  "Four beats per bar"                    |  <-- Plain English, always shown
|  Standard rock/pop beat                  |  <-- Familiar description
|                                          |
|  Confidence: 85%                         |  <-- Keep but de-emphasize
|                                          |
+------------------------------------------+
```

For odd meters, add grouping visualization:

```
+------------------------------------------+
|                                          |
|              7/8                         |
|                                          |
|  "Seven beats per bar,                   |
|   grouped 2 + 2 + 3"                    |
|                                          |
|  [**] [**] [***]                        |  <-- Visual grouping
|                                          |
|  Like "Money" by Pink Floyd              |
|  Confidence: 72%                         |
|                                          |
+------------------------------------------+
```

### 3B. Plain-language meter descriptions (new)

Replace technical jargon with musician-friendly descriptions:

| Meter | Current description | Proposed description |
|-------|-------------------|---------------------|
| 2/4   | "March, polka"    | "Two beats per bar - like a march" |
| 3/4   | "Waltz"           | "Three beats per bar - waltz feel" |
| 4/4   | "Standard rock/pop" | "Four beats per bar - the most common meter" |
| 5/4   | "Take Five"       | "Five beats per bar - grouped 3+2 or 2+3" |
| 6/8   | "Tarantella"      | "Six beats per bar in two groups of three - a swaying feel" |
| 7/8   | "Money"           | "Seven beats per bar - often grouped 2+2+3" |

Keep the song examples as secondary context (e.g. "Like 'Take Five' by Dave Brubeck"), but lead with the structural description.

### 3C. Confidence display strategy

| Confidence range | Display | UX behavior |
|-----------------|---------|-------------|
| 80-100%         | "Confident" (green) | Show result normally |
| 60-79%          | "Likely" (yellow) | Show result + "Could also be X/Y" |
| 40-59%          | "Uncertain" (orange) | Show top 2 hypotheses side by side |
| < 40%           | "Not sure" (red) | "I couldn't determine the meter clearly. Try tapping longer or with clearer accents." |

**Important**: Never show raw floating-point confidence to beginners. Use words. The percentage can remain visible but smaller/secondary.

---

## 4. Key Product Decisions

### 4A. Should we distinguish 2/4 from 4/4?

**Recommendation: Default to 4/4, explain the ambiguity.**

Rationale:
- The current benchmark shows the engine often reports 2/4 for what users would call 4/4 (Rock Beat, Drum Beat, Drum Cadence A all detected as 2/4).
- For tapped rhythm, distinguishing 2/4 from 4/4 requires detecting a "super-accent" every 4 beats (beat 1 louder than beat 3). This is subtle and unreliable.
- Most musicians conceptualize "4/4" even when the mathematical pattern is 2/4.
- **Implementation**: When the engine returns 2/4 with less than 80% confidence advantage over 4/4, present as "4/4 (or possibly 2/4)" rather than asserting 2/4. Only show 2/4 as primary when the accent pattern clearly favors it (e.g., strong march-like 1-2-1-2).

### 4B. Should the denominator matter for tapped input?

**Recommendation: Hide the denominator for live/tapped mode; show it for file upload.**

Rationale:
- When tapping, the user provides quarter-note-level beats. There's no subdivided information to distinguish /4 from /8.
- Show "in 7" instead of "7/8" for tapped input, with a small note: "Assuming eighth-note pulse based on tempo."
- For file uploads (recorded music), the full time signature is meaningful because the audio contains subdivision information.

**Update**: After further consideration, showing "7/8" is probably clearer than "in 7" since even beginners recognize time signature notation. But add the plain-language description as the primary label.

### 4C. Should the metronome auto-start after detection?

**Recommendation: No auto-start. Offer a prominent "Hear it" button.**

Rationale:
- Auto-playing audio is a universally hated UX pattern.
- But the verify loop (tap -> analyze -> listen to metronome -> confirm) is extremely valuable.
- After showing results, display a large "Hear it" button that starts the metronome at the detected tempo and meter. One-tap verification.

### 4D. How to handle the warm-up period?

**Current state**: The live mode has a warm-up progress bar. Good foundation.

**Enhancement for tapped input**:
1. Show each detected tap as a visual pulse (already done with onset dots - good).
2. After 6+ taps, show a preliminary "I'm hearing ~120 BPM" text.
3. After enough bars (8+ taps with clear accent pattern), show full results.
4. If the user keeps tapping, update results every 2-3 bars (but don't flicker - smooth transitions).

---

## 5. Edge Cases and Graceful Handling

### 5A. User taps irregularly at first, then settles
- **Solution**: Use a sliding window that weights recent beats more heavily. Discard the first 2 seconds or first 4 taps (whichever is longer).
- Show "Getting your groove..." during the warm-up phase.

### 5B. User stops and restarts
- **Solution**: Detect a gap > 2x the expected beat interval. Reset the analysis window but keep the previous result visible (greyed out) with a "Tap again to update" message.

### 5C. User changes meter mid-session
- **Solution**: For live mode, track the most recent 10-second window. If meter changes, show the new result with a subtle transition. Do NOT show both meters simultaneously (confusing).
- For file upload, the sections timeline already handles this.

### 5D. Genuinely ambiguous patterns (6/8 vs 3/4)
- **Already handled**: The disambiguation hints in i18n.js are excellent. The 6/8 vs 3/4 hint explaining "two groups of 3" vs "three equal beats" is clear and educational.
- **Enhancement**: Add a visual comparison mode: show both patterns side-by-side with the beat grid, let the user tap along to each to see which feels right.

### 5E. User expects 4/4 but is actually playing 5/4
- **Solution**: When detecting an unexpected meter, phrase it positively: "Sounds like you're playing in 5! That's an asymmetric meter - think 'Take Five' by Dave Brubeck." Not: "ERROR: Expected 4/4, got 5/4."
- If confidence is moderate, offer: "This could be 5/4 or possibly 4/4 with some extra beats. What do you think?"

---

## 6. Current Accuracy Assessment

From the benchmark (22 tests):
- **Engine meter accuracy**: 68% (15/22)
- **BeatNet raw meter**: 55% (12/22)
- **Tempo accuracy**: 91% (20/22)

### Key failure patterns relevant to tapped rhythm:
1. **3/4 waltzes are the weakest area**: Bubbles, Chopin, Blue Danube all fail (detected as 5/4 or 11/4). This is critical because waltz is one of the most common meters beginners want to identify.
2. **Shuffles and syncopated patterns**: 4/4 shuffle detected as 7/4 - the engine is confused by swing/syncopation.
3. **2/4 vs 4/4 ambiguity**: Most 4/4 rock patterns are detected as 2/4 - functionally correct but will confuse users.

### Priority fixes for tapped rhythm use case:
1. **P0**: Fix 3/4 waltz detection (currently 2/5 waltzes correct = 40%)
2. **P0**: Handle 2/4-vs-4/4 presentation (report as "4/4" by default for tapped input)
3. **P1**: Improve shuffle/syncopation handling (don't let swing feel create false odd meters)
4. **P2**: Add tapped-specific preprocessing (onset-only, no spectral features needed)

---

## 7. Prioritized Recommendations

### P0 - Must Have (before tapped rhythm is usable)

1. **Fix waltz (3/4) detection** - Currently failing 60% of the time. This is a showstopper for the most common non-4/4 meter.

2. **Smart 2/4 -> 4/4 presentation** - When engine says 2/4 and confidence gap vs 4/4 is small, present as 4/4 with a note. Add a `tapped_input` flag to the API so the backend can apply this heuristic.

3. **Minimum input guard** - Don't attempt meter analysis until 6+ seconds / 8+ onsets. Show "Keep tapping..." with a progress indicator.

4. **Low confidence fallback message** - When confidence < 40%, show "I'm not sure yet. Try tapping a bit longer with clearer accents on beat 1." instead of showing a possibly wrong result.

### P1 - Should Have (significant UX improvement)

5. **Plain-language descriptions** - Replace "Meter Hypotheses" heading with "What meter are you playing?". Lead with structural description ("Three beats per bar"), follow with examples.

6. **"Hear it" verification button** - After showing results, prominent button to start metronome at detected tempo/meter. Lets user instantly verify.

7. **Warm-up beat discarding** - Down-weight or discard first 2 seconds of input to avoid irregular warm-up beats skewing analysis.

8. **Onset dot visualization refinement** - The current onset dots (audio-capture.js:129-145) are good. Add size variation based on accent strength to give visual feedback on accent pattern.

### P2 - Nice to Have (polish)

9. **Visual grouping for odd meters** - For 7/8, show `[**] [**] [***]` beat boxes in addition to the circular beat grid. More intuitive for beginners than the clock visualization.

10. **Comparative mode for ambiguous results** - When top 2 hypotheses are close, show both with "Tap along to check" interactive comparison.

11. **Meter override learning** - When user manually overrides the detected meter, log this as training signal for improving the engine.

12. **Localized rhythm names** - Beyond EN/PL, add culturally relevant meter examples (e.g., for Middle Eastern users: "Saidi" instead of "march" for 4/4).

---

## 8. UI Copy Recommendations

### Section headings (current -> proposed)
- "Meter Hypotheses" -> "Detected Meter" (or "What meter are you playing?" for live mode)
- "Beat Grid" -> "Beat Pattern"
- "Onset detection" / "Beat tracking" -> "Listening for your taps..." / "Finding the beat..."

### Confidence labels (current -> proposed)
- "Confidence: 85%" -> "Very likely 4/4" or keep percentage but add word label
- "Confidence: 52%" -> "Possibly 3/4 - could also be 6/8"

### Error/edge case messages
- Mic denied: Keep current (clear and actionable)
- Too short: "Keep tapping! I need a few more bars to figure out the meter."
- Very low confidence: "I'm hearing beats but can't pin down the meter yet. Try accenting beat 1 more strongly."
- No onsets detected: "I don't hear any taps yet. Make sure your mic is working and tap firmly."

---

## 9. Technical Notes for Implementation

- The live WebSocket mode already streams audio and returns analysis results. The tapped rhythm use case is essentially the live mode with percussion-only input.
- The `compute_beat_energies()` function with +/-30ms RMS window (per MEMORY.md) is well-suited for tapped input where accent = volume.
- Consider adding a `source_type: "tapped" | "recorded"` parameter to the analysis API to enable tapped-specific heuristics (2/4->4/4 mapping, denominator hiding, warm-up discarding).
- The AudioWorklet processor already captures raw PCM chunks - no changes needed on the capture side.
