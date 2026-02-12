# Meter Classification System: Comprehensive Musicological & ML Analysis

**Date**: 2026-02-12
**System under analysis**: BeatMeter rhythm analyzer
**Current categories**: [3, 4, 5, 7, 9, 11] with multi-label sigmoid output
**Benchmark**: METER2800 — 532/700 (76%) on test split

---

## Table of Contents
1. [Are Our Categories Optimal?](#1-are-our-categories-optimal)
2. [Additive/Aksak Meters and Decomposition](#2-additiveaksak-meters-and-decomposition)
3. [Polyrhythm vs Mixed/Changing Meter](#3-polyrhythm-vs-mixedchanging-meter)
4. [Soft Labels for Ambiguous Meters](#4-soft-labels-for-ambiguous-meters)
5. [Practical Output Format](#5-practical-output-format)
6. [Proposed Musicologically-Informed Soft Label Scheme](#6-proposed-musicologically-informed-soft-label-scheme)
7. [Concrete Recommendations](#7-concrete-recommendations)
8. [References](#8-references)

---

## 1. Are Our Categories Optimal?

### 1.1 What the Literature Uses

The research community has no consensus on meter taxonomy for classification. The landscape looks like this:

| System / Paper | Classes | Notes |
|---|---|---|
| **METER2800** (Ferreira et al., 2023) | 4: {3, 4, 5, 7} | Numerator only; 6/8→3, 12/8→4 |
| **ResNet18** (Ferreira et al., 2024) | 4: {3, 4, 5, 7} | Same as METER2800; 88% binary 3/4 vs 4/4 |
| **Holzapfel & Stylianou** (Turkish music) | 6: {3/4, 4/4, 5/8, 8/8, 9/8, 10/8} | Actual time signatures, not numerator classes |
| **SSM-based** (Greek music, 95.5%) | 3: {2/4, 3/4, 4/4} | Simple meters only |
| **BeatNet** (ISMIR 2021) | 2: {duple, triple} | Implicit via downbeat tracking |
| **Time Signature Survey** (Kapoor et al., 2021) | Varies | Most papers use 2-4 classes |
| **Our system** | 6: {3, 4, 5, 7, 9, 11} | Multi-label with polyrhythm support |

**Key observation**: Nearly all published work restricts itself to {3, 4} or {3, 4, 5, 7}. Our 6-class system with 9 and 11 is **more ambitious than anything in the published literature**. This is both a strength (covers more musical ground) and a weakness (extremely sparse training data for 9 and 11).

### 1.2 Should We Distinguish Compound from Simple?

**The musicological case is strong.** 3/4 and 6/8 are fundamentally different:

- **3/4** (simple triple): THREE beats, each divisible into 2. Stress pattern: **S**-w-w. Feel: waltz, mazurka, sarabande.
- **6/8** (compound duple): TWO beats, each divisible into 3. Stress pattern: **S**-w-w-**s**-w-w. Feel: jig, tarantella, barcarolle.

The *sesquialtera* phenomenon (documented extensively by Cano et al. in their TISMIR study of Colombian bambuco) shows that the 3/4 vs 6/8 distinction is **perceptually real but genuinely ambiguous** in many musical traditions. In their perception study, 10 Colombian musicians annotated the same bambucos in **five different metric interpretations**, confirming that the same music can legitimately be heard as 3/4, 6/8, or alternating between them.

**However, for classification, this is dangerous.** If we separate 6/8 from 3/4:
- We split an already-small class (category 3 has only 300 test files in METER2800)
- Many pieces genuinely sit on the boundary (sesquialtera, hemiola passages)
- Ground truth annotations would become even more contested

**Recommendation**: Keep 6/8 within category 3 for the primary classification output, but add a **compound/simple sub-classification** as a secondary output. The sigmoid output for class 3 already tells us "this is triple-grouped." A second head or post-processing step can estimate "is this compound (6/8) or simple (3/4)?" based on subdivision analysis. This mirrors the hierarchical approach described in the Time Signature Survey: tatum → tactus → measure.

### 1.3 Should We Distinguish 2/4 from 4/4?

**Less important than 6/8 vs 3/4, but still musicologically distinct.**

2/4 has a lighter, more march-like feel (polka, samba). 4/4 has the strong-weak-medium-weak pattern of rock/pop. However:

- From an audio signal perspective, 2/4 is essentially a sub-period of 4/4. Our code already handles this with `SUBPERIOD_4_4_THRESHOLD` and `SUBPERIOD_3_4_THRESHOLD`.
- The "Skip That Beat" paper (Morais et al., ISMIR LBD 2024) specifically addresses 2/4 vs 4/4 in their data augmentation, removing beats from 4/4 to create 2/4 tracks. Their results show the distinction is learnable but only marginally so.
- In METER2800, 2/4 is lumped with 4/4 (both → class 4).

**Recommendation**: Keep 2/4 within category 4. If needed, add as sub-classification. The primary challenge is detecting duple vs triple vs irregular — the 2/4 vs 4/4 distinction is secondary.

### 1.4 Categories to Add/Remove/Merge

**Keep as-is**: 3, 4, 5, 7
**Keep but acknowledge data scarcity**: 9, 11
**Consider adding**: None at primary level
**Consider removing**: None

The 6-class {3, 4, 5, 7, 9, 11} system is sound. The real issue is not the taxonomy but the **data imbalance** and **within-class heterogeneity**. Category 3 contains waltzes AND jigs AND compound duple — these are very different rhythmically but share the "triple grouping" property.

---

## 2. Additive/Aksak Meters and Decomposition

### 2.1 The Aksak Problem

Aksak (Turkish for "limping") refers to meters built from **unequal subdivision groups** — alternations of groups of 2 and 3 at the fastest metrical level. This is fundamental to Balkan, Turkish, Greek, and some West African music.

The key insight: **the numerator alone is insufficient to characterize aksak meters.** Consider 7/8:

| Grouping | Feel | Example |
|---|---|---|
| 2+2+3 | "short-short-long" | Most Bulgarian rachenitsa |
| 3+2+2 | "long-short-short" | Some Macedonian music |
| 2+3+2 | "short-long-short" | Less common, some Greek dances |

These feel **completely different** when performed. A Bulgarian musician would immediately tell you 2+2+3 is not the same as 3+2+2. Yet our system currently assigns them the same label: category 7.

### 2.2 The 9/8 Compound vs Aksak Distinction

This is the most critical subdivision issue in our system. Two completely different meters share the same numerator:

- **9/8 compound** (3+3+3): This is rhythmically equivalent to **3/4 with triplet subdivision**. The Moonlight Sonata's third movement, "Ride of the Valkyries" in 9/8 compound — these feel like triple meter. Our system correctly maps this to category 3.
- **9/8 aksak** (2+2+2+3): "Blue Rondo a la Turk" by Dave Brubeck. This is a genuine asymmetric meter. It should NOT be in category 3. Our system correctly treats this as category 9.

**Current handling is correct**: compound 9/8 → category 3, aksak 9/8 → category 9. But the detection of which type of 9/8 is playing is the hard part.

### 2.3 Should Decomposition Be Reflected in Soft Labels?

The argument for: 11/8 = 5+6 means the piece has a "5-ness" and a "6-ness" (which decomposes to 3+3, relating to "3-ness"). So `[5→0.3, 11→1.0]` captures structural similarity.

The argument against: This conflates **subdivision structure** with **meter classification**. A piece in 11/8 is not "partly in 5/4" — it is firmly in 11/8. The 5+6 decomposition describes internal grouping, not meter ambiguity.

**My assessment**: Soft labels for decomposition are **not the right mechanism**. Instead:

1. Use hard labels for the meter class (11 is 11, not "partly 5")
2. Predict subdivision grouping as a **separate output** (see Section 5)
3. Reserve soft labels for genuine **perceptual ambiguity** (see Section 4)

### 2.4 How to Handle Decomposition

The `GROUPINGS` dict in `meter.py` already captures the right information:

```python
GROUPINGS = {
    (5, 4): ["3+2", "2+3"],
    (7, 8): ["2+2+3", "3+2+2", "2+3+2"],
    (9, 8): ["2+2+2+3", "3+3+3", "3+2+2+2"],
    (11, 8): ["3+3+3+2", "2+2+3+2+2", "3+2+3+3"],
}
```

This is a good start. The challenge is **detecting** the grouping from audio. Approaches:

- **Onset accent analysis**: For each possible grouping, check if onsets/accents align with the predicted strong positions (first beat of each group).
- **Template matching**: Create rhythmic templates for each grouping variant and cross-correlate with the detected onset pattern.
- **Learned approach**: Multi-task head that jointly predicts meter class + grouping. Requires labeled training data with grouping annotations (which METER2800 does not have).

**Recommendation**: Grouping detection is a secondary priority. Get the meter class right first. When the system reports 7/8, include the most likely grouping from accent analysis as metadata.

---

## 3. Polyrhythm vs Mixed/Changing Meter

### 3.1 Taxonomy of Complex Rhythmic Structures

The 2026 paper "Challenging Beat Tracking" (Springer) provides a useful taxonomy based on three musical examples of increasing complexity:

| Phenomenon | Definition | Example | Duration |
|---|---|---|---|
| **Polyrhythm** | Two different rhythmic patterns over the **same metric framework** | Uruguayan Candombe (3 interlocking drum parts) | Continuous |
| **Polymeter** | Two different meters **simultaneously** | Colombian Bambuco (3/4 + 6/8) | Continuous |
| **Polytempo** | Two different tempos **simultaneously** | Steve Reich's "Piano Phase" | Continuous |
| **Mixed meter** | Meter **changes** between sections | Stravinsky's "Rite of Spring" | Sequential |
| **Metric modulation** | Tempo/meter shifts via reinterpretation of note values | Elliott Carter compositions | Transitional |

### 3.2 What Should Our System Output?

For each of these, the appropriate output differs:

**Polyrhythm** (e.g., Afrobeat, Candombe):
- The *meter* is typically well-defined (4/4 for Afrobeat). The polyrhythm exists within the metric framework.
- Output: primary meter + "polyrhythmic" flag.
- Our multi-label sigmoid is well-suited: `[3→0.0, 4→0.95]` — the meter is clear, the complexity is in the rhythmic layer.

**Polymeter / Sesquialtera** (e.g., Bambuco, hemiola passages):
- Two meters genuinely coexist. The bambuco study showed 5 different metric interpretations from 10 expert listeners.
- Output: multi-label `[3→0.8, 4→0.6]` — both meters are genuinely present.
- Our system already supports this with `[3,4]` polyrhythmic output. This is correct.

**Mixed meter** (e.g., prog rock, Stravinsky):
- The meter changes over time. A single global label is misleading.
- Output: **temporal meter map** — a sequence of (time_range, meter) pairs.
- Our current system does NOT support this. It outputs a single global meter.

**Metric modulation**:
- This is a performance/composition technique, not a meter category. The notated meter may not change even though the perceived pulse does.
- Output: Flag as "metric modulation detected" if beat tracking shows systematic tempo ratio changes (e.g., dotted quarter = quarter).
- Low priority for implementation.

### 3.3 Literature Recommendations

The academic literature (particularly the "Challenging Beat Tracking" work and the Bambuco study) converges on these points:

1. **Global meter estimation is insufficient for complex music.** Most real-world music may be in 4/4 throughout, but the interesting cases (and failure cases) involve meter changes or ambiguity.
2. **Human-in-the-loop adaptation** can help for edge cases, but autonomous systems need to at least **flag** when they are uncertain.
3. **Beat tracking and meter estimation should be treated as a joint problem**, not sequential. This is what BeatNet and Beat This! already do.

### 3.4 Practical Impact for Our System

Mixed meter detection would be a **major architectural change** — moving from "one label per file" to "a sequence of labels over time." This is:
- High value (many meter-5 and meter-7 files in METER2800 probably contain passages of 4/4)
- High effort (requires windowed analysis, segment boundaries, temporal consistency)
- Not supported by METER2800 annotations (which are global labels)

**Recommendation**: Phase this in gradually:
1. **Near-term**: Add a "meter stability" score to the output (is the meter consistent throughout, or do signals disagree at different time points?)
2. **Medium-term**: Windowed analysis with 8-16 bar windows, outputting a temporal meter map
3. **Long-term**: True mixed-meter detection with segment boundaries

---

## 4. Soft Labels for Ambiguous Meters

### 4.1 Literature Support

The idea of using soft training labels that encode inter-class relationships is **well-supported** in ML literature, though not (to my knowledge) specifically applied to meter classification:

- **Class-Similarity Based Label Smoothing** (Li et al., 2020, arXiv:2006.14028): Demonstrates that distributing label smoothing mass proportionally to class similarity (rather than uniformly) improves both accuracy and calibration. "More similar classes should result in closer probability values."
- **Relation-Aware Label Smoothing (RAS-KD)** (2024, Springer): Uses inter-class relationships between class representative vectors to generate soft labels, outperforming uniform label smoothing.
- **Label Smoothing++ (2025, arXiv:2509.05307)**: Enhanced label regularization that accounts for semantic relationships between classes.
- **Multi-label music genre classification** (Oramas et al., ISMIR 2017): Music genre shares the "fuzzy boundary" problem with meter. Their multi-label approach with soft targets improved genre classification.
- **Metrical multistability** (Desain & Honing, 2003, Cognition): Demonstrates that human perception of meter is genuinely multistable — the same rhythmic pattern can be heard in multiple meters, providing cognitive science justification for soft labels.

### 4.2 Why Soft Labels Make Musicological Sense for Meter

Meter perception is **demonstrably multistable**. The cognitive science literature (Desain & Honing, 2003) shows that listeners can adopt different metrical interpretations of the same rhythmic sequence, and these aren't just cognitive judgments but genuine perceptual differences reflected in sensorimotor synchronization.

The bambuco study (Cano et al., TISMIR 2021) provides the strongest empirical evidence: 10 expert musicians produced **5 different metric interpretations** of the same music. This is not annotator noise — it is genuine perceptual ambiguity.

This means that for some pieces, a hard label of "3" is **wrong** — the music genuinely has properties of both 3 and 4. A soft label `[3→0.8, 4→0.5]` better captures the ground truth.

### 4.3 Proposed Soft Label Scheme

There are two distinct reasons to soften labels, and they should be handled differently:

#### Type A: Genuine Perceptual Ambiguity
These are cases where **multiple metric interpretations are musicologically valid**:

| Situation | Soft Label | Rationale |
|---|---|---|
| 6/8 with hemiola | [3→1.0, 4→0.3] | The hemiola creates momentary 3+3→2+3 or 3+3→3+2 groupings |
| Bambuco-style sesquialtera | [3→0.9, 4→0.7] | Both 3/4 and 6/8 genuinely present simultaneously |
| 12/8 shuffle | [3→0.4, 4→1.0] | Felt as 4 beats with triplet subdivision |
| Fast 3/4 heard as 1-in-a-bar | [3→1.0] | No ambiguity — still triple |
| Slow 6/8 with clear duple feel | [3→1.0, 4→0.15] | Mild duple tendency from compound grouping |

#### Type B: Structural Relationships (NOT recommended as soft labels)
These are cases where meters share structural properties but are NOT perceptually ambiguous:

| Situation | Why NOT to soften |
|---|---|
| 7/8 = 3+4 | Nobody hears 7/8 as "partly 3/4 and partly 4/4" |
| 11/8 = 5+6 | The 11 is the meter; the 5+6 is internal grouping |
| 5/4 = 3+2 | 5/4 is its own thing, not a mix of 3 and 2 |

**The distinction is critical**: Type A reflects genuine perceptual ambiguity (the listener could reasonably assign either meter). Type B reflects mathematical decomposition (the listener knows what meter they are hearing).

### 4.4 How to Implement

For the multi-label sigmoid architecture, soft labels integrate naturally:

```
# Current (hard labels)
target = [0, 0, 0, 0, 0, 0]  # one-hot for category indices
target[class_idx] = 1.0

# Proposed (soft labels for Type A ambiguity)
target = [0, 0, 0, 0, 0, 0]
target[class_idx] = 1.0
# Add soft targets for related classes
for related_class, weight in ambiguity_map[meter_type]:
    target[related_class] = weight
```

With BCE loss (which we already use), this works directly — each sigmoid output is trained independently, so a target of `[3→1.0, 4→0.3]` simply means "this piece is definitely 3, and somewhat 4."

### 4.5 Practical Considerations

1. **Annotation effort**: Soft labels require musicological judgment for each training example, or at least per-category rules. METER2800 has hard labels only.
2. **Rule-based approximation**: We can derive soft labels from rules (e.g., "all 6/8 files get 4→0.15") without per-file annotation. This is a good starting point.
3. **Learned soft labels**: Train a teacher model, use its output probabilities as soft targets for a student model (knowledge distillation). The teacher's "confusion" between 3 and 4 for 6/8 files is exactly the signal we want.
4. **Calibration**: Soft labels should improve calibration (sigmoid outputs will be more meaningful as probabilities) because the model learns that some pieces genuinely have multi-class membership.

---

## 5. Practical Output Format

### 5.1 Recommended Output Structure

Based on the analysis above, here is the recommended output format, ordered by priority:

#### Tier 1: Always output (current system, enhanced)
```json
{
  "primary_meter": {"numerator": 7, "denominator": 8},
  "confidence": 0.82,
  "class_scores": {
    "3": 0.12, "4": 0.08, "5": 0.15, "7": 0.82, "9": 0.03, "11": 0.01
  },
  "description": "Money (Pink Floyd) - grupowanie 2+2+3 lub 3+2+2"
}
```

#### Tier 2: Add subdivision and ambiguity info
```json
{
  "primary_meter": {"numerator": 7, "denominator": 8},
  "confidence": 0.82,
  "grouping": "2+2+3",
  "grouping_confidence": 0.65,
  "ambiguity_score": 0.18,
  "alternative_meters": [
    {"meter": {"numerator": 4, "denominator": 4}, "confidence": 0.08}
  ]
}
```

- **ambiguity_score**: entropy of the class_scores distribution. High entropy = genuinely ambiguous meter. This is cheap to compute from existing sigmoid outputs.
- **grouping**: most likely subdivision pattern for irregular meters.

#### Tier 3: Temporal meter map (future)
```json
{
  "global_meter": {"numerator": 7, "denominator": 8},
  "meter_stability": 0.91,
  "temporal_map": [
    {"start_bar": 1, "end_bar": 16, "meter": "4/4", "confidence": 0.95},
    {"start_bar": 17, "end_bar": 24, "meter": "7/8", "confidence": 0.82},
    {"start_bar": 25, "end_bar": 48, "meter": "4/4", "confidence": 0.93}
  ],
  "meter_changes_detected": true
}
```

### 5.2 The Ambiguity Score

Define as normalized entropy of the sigmoid outputs:

```
ambiguity = -sum(p * log(p) + (1-p) * log(1-p)) / (N * log(2))
```

where p_i are the sigmoid outputs and N is the number of classes. This gives:
- 0.0 for a completely certain prediction (one sigmoid at 1.0, rest at 0.0)
- 1.0 for maximum uncertainty (all sigmoids at 0.5)

In practice, most pieces will score 0.05-0.20 (clear meter), and genuinely ambiguous pieces will score 0.30+.

### 5.3 Compound/Simple Sub-Classification

For category 3 (triple), add a secondary output:

```json
{
  "primary_meter": {"numerator": 3, "denominator": 4},
  "compound_type": "simple",
  "compound_confidence": 0.72,
  "note": "If compound, equivalent to 6/8"
}
```

Detection approach: analyze the **subdivision** level. In 3/4 (simple), beats subdivide into 2 (eighth notes). In 6/8 (compound), the "beats" (dotted quarters) subdivide into 3. Check onset density between beats:
- Regular pairs of onsets between beats → simple (3/4)
- Regular triplets between beats → compound (6/8)
- Mixed → sesquialtera

---

## 6. Proposed Musicologically-Informed Soft Label Scheme

### 6.1 Rule-Based Soft Labels for METER2800

Since METER2800 has hard labels and we cannot re-annotate 2800 files, here is a rule-based scheme to derive soft labels from the existing annotations plus audio features:

#### Static Rules (apply to all files in a category)

| Ground Truth | Target Vector [3, 4, 5, 7, 9, 11] | Rationale |
|---|---|---|
| Pure 4/4 (rock/pop) | [0, 1.0, 0, 0, 0, 0] | Unambiguous |
| Pure 3/4 (waltz) | [1.0, 0, 0, 0, 0, 0] | Unambiguous |
| 6/8 | [1.0, 0.15, 0, 0, 0, 0] | Compound duple has mild 4-grouping affinity |
| 12/8 | [0.25, 1.0, 0, 0, 0, 0] | Felt as 4 with triple subdivision |
| 2/4 | [0, 1.0, 0, 0, 0, 0] | Duple = category 4 |
| 5/4 or 5/8 | [0, 0, 1.0, 0, 0, 0] | Unambiguous irregular |
| 7/4 or 7/8 | [0, 0, 0, 1.0, 0, 0] | Unambiguous irregular |
| 9/8 aksak | [0, 0, 0, 0, 1.0, 0] | Unambiguous irregular |
| 9/8 compound | [1.0, 0, 0, 0, 0.1, 0] | Primarily triple, trace of 9-grouping |
| 11/8 | [0, 0, 0, 0, 0, 1.0] | Unambiguous irregular |

#### Dynamic Rules (require audio analysis)

For files where the system or an existing model is uncertain, use the **teacher model's confusion** as soft labels:

1. Run the current 7-signal engine on each training file
2. If the engine outputs `[3→0.7, 4→0.5]` for a file labeled "3", this suggests genuine ambiguity
3. Use a weighted combination: `target = 0.8 * hard_label + 0.2 * engine_prediction`

This is a form of **self-knowledge distillation** (Yang et al., ICCV 2023) — using the model's own predictions to soften the training labels.

### 6.2 Expected Impact

Based on the class-similarity label smoothing literature:

- **Accuracy**: Modest improvement (1-3pp) from better calibration
- **Calibration**: Significant improvement — sigmoid outputs become more meaningful probabilities
- **Robustness**: Better handling of boundary cases (6/8 hemiola, 12/8 shuffle)
- **Regression risk**: Low, since the primary label is unchanged; only secondary labels are added

### 6.3 Implementation Priority

1. **Start simple**: Static rules only (the table above). Requires no new audio analysis.
2. **Validate**: Compare BCE loss with soft targets vs hard targets on the tuning split.
3. **If promising**: Add dynamic rules using engine predictions as soft label component.
4. **If very promising**: Full knowledge distillation with a teacher model.

---

## 7. Concrete Recommendations

### 7.1 High Priority (Do Now)

1. **Keep the 6-class system {3, 4, 5, 7, 9, 11}**. It is well-justified and more comprehensive than any published benchmark. Do not add or remove classes.

2. **Implement static soft labels** for training (Section 6.1). This requires only modifying the training script — no new data, no new annotations. Expected gain: 1-3pp from better handling of compound meters and boundary cases.

3. **Add an ambiguity score** to the output. This is free — just compute entropy of existing sigmoid outputs. Immediately useful for flagging unreliable predictions.

4. **Report grouping for irregular meters**. The `GROUPINGS` dict already exists. Use accent analysis to pick the most likely grouping and include it in the output.

### 7.2 Medium Priority (Next Round)

5. **Compound/simple sub-classification for category 3**. Add a secondary head or post-processing step that distinguishes 3/4 from 6/8 based on subdivision analysis. This addresses a genuine musical distinction without complicating the primary taxonomy.

6. **Meter stability score**. Run the analysis on multiple overlapping windows and compute agreement. A piece where all windows agree on 4/4 gets stability=1.0; a piece where some windows say 4/4 and others say 7/8 gets stability=0.5.

7. **Knowledge distillation soft labels**. Use the full engine's predictions on training data to create soft targets for the MERT/ResNet models. This transfers the engine's "confusion patterns" (which reflect real musical ambiguity) to the learned model.

### 7.3 Low Priority (Future)

8. **Temporal meter map**. Windowed analysis with segment boundaries. High value but high effort. Requires rethinking the evaluation protocol (METER2800 has global labels only).

9. **Subdivision grouping detection for aksak meters**. Template matching or learned approach to distinguish 2+2+3 from 3+2+2 in 7/8. Requires labeled training data that does not currently exist.

10. **Mixed meter / metric modulation detection**. Detect where the meter changes within a piece. Requires a fundamentally different architecture (sequential labeling rather than classification).

### 7.4 Do NOT Do

- **Do NOT separate 6/8 into its own primary class.** This would split already-sparse data and create annotator disagreement. Handle it as a sub-classification within category 3.
- **Do NOT use decomposition-based soft labels** (7 = 3+4 → soften 3 and 4). This conflates internal grouping with metric identity. A 7/8 piece is not "partly in 3."
- **Do NOT add categories for extremely rare meters** (13/8, 15/8, etc.). There is essentially zero training data, and these are better handled as "irregular/other" with a freeform output.

---

## 8. References

### Datasets & Benchmarks
1. Ferreira, G. et al. (2023). "METER2800: A novel dataset for music time signature detection." *Data in Brief*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10700346/) | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923008053) | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/0CLXBQ)

### Time Signature Detection
2. Ferreira, G. et al. (2024). "Music time signature detection using ResNet18." *EURASIP J. Audio Speech Music Process.* [Springer](https://link.springer.com/article/10.1186/s13636-024-00346-6)
3. Kapoor, S. et al. (2021). "Time Signature Detection: A Survey." *Sensors*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8512143/) | [MDPI](https://www.mdpi.com/1424-8220/21/19/6494)

### Beat Tracking & Meter Estimation
4. Foscarin, F. et al. (2024). "Beat this! Accurate beat tracking without DBN postprocessing." *ISMIR 2024*. [arXiv](https://arxiv.org/abs/2407.21658) | [GitHub](https://github.com/CPJKU/beat_this)
5. Morais, G. et al. (2024). "Skip That Beat: Augmenting Meter Tracking Models for Underrepresented Time Signatures." *LAMIR 2024 & ISMIR LBD 2024*. [arXiv](https://arxiv.org/abs/2502.12972) | [GitHub](https://github.com/giovana-morais/skip_that_beat)
6. Heydari, M. et al. (2021). "BeatNet: CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter Tracking." *ISMIR 2021*. [arXiv](https://arxiv.org/abs/2108.03576) | [GitHub](https://github.com/mjhydri/BeatNet)

### Complex Meters & Polyrhythm
7. "Challenging Beat Tracking: Tackling Polyrhythm, Polymetre, and Polytempo with Human-in-the-Loop Adaptation." (2026). *Springer*. [Link](https://link.springer.com/chapter/10.1007/978-3-032-02042-0_35)
8. Cano, E. et al. (2021). "Sesquialtera in the Colombian Bambuco: Perception and Estimation of Beat and Meter – Extended version." *TISMIR*. [Link](https://transactions.ismir.net/articles/10.5334/tismir.118)
9. Holzapfel, A. & Stylianou, Y. "In Search of Automatic Rhythm Analysis Methods for Turkish and Indian Art Music." [ResearchGate](https://www.academia.edu/45299384/In_Search_of_Automatic_Rhythm_Analysis_Methods_for_Turkish_and_Indian_Art_Music)

### Aksak & Irregular Meters
10. Goldberg, D. et al. "Pattern and Variation in the Timing of Aksak Meter." *Empirical Musicology Review*. [Link](https://emusicology.org/article/view/4883)
11. "Aksak." *Wikipedia*. [Link](https://en.wikipedia.org/wiki/Aksak)
12. Complex meters. *Chromatone.center*. [Link](https://chromatone.center/theory/rhythm/meter/complex/)

### Soft Labels & Label Smoothing
13. Li, Z. et al. (2020). "Class-Similarity Based Label Smoothing for Confidence Calibration." [arXiv](https://arxiv.org/abs/2006.14028)
14. "Relation-Aware Label Smoothing for Self-KD." (2024). *Springer*. [Link](https://link.springer.com/chapter/10.1007/978-981-97-2253-2_16)
15. Chen, W. et al. (2025). "Label Smoothing++: Enhanced Label Regularization for Training Neural Networks." [arXiv](https://arxiv.org/html/2509.05307)
16. Yang, L. et al. (2023). "From Knowledge Distillation to Self-Knowledge Distillation: A Unified Approach." *ICCV 2023*. [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_From_Knowledge_Distillation_to_Self-Knowledge_Distillation_A_Unified_Approach_with_ICCV_2023_paper.pdf)

### Music Perception & Cognition
17. Desain, P. & Honing, H. (2003). "Hearing a melody in different ways: Multistability of metrical interpretation." *Cognition*. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010027706000540)
18. Pearce, M. (2023). "Music Perception." *Oxford Research Encyclopedias*. [PDF](https://www.marcus-pearce.com/assets/papers/Pearce2023.pdf)

### Music Representation Learning
19. Li, Y. et al. (2023). "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training." *ICLR 2024*. [arXiv](https://arxiv.org/abs/2306.00107) | [GitHub](https://github.com/yizhilll/MERT)

### Multi-Label Music Classification
20. Oramas, S. et al. (2017). "Multi-label music genre classification from audio, text, and images using deep features." *ISMIR 2017*. [PDF](https://ccrma.stanford.edu/~urinieto/MARL/publications/ismir2017.pdf)
