# Limitations and responsible interpretation

This is computational replication and extension, not a published CruxRNN paper
or a human-subject study. The original article remains attributed to its authors.

- The initial cue contains the full target. The task measures decoding and
  sequential unfolding, not inference of withheld movement intentions.
- Scalar 1–4 cues impose ordinal geometry on arbitrary categories. Relabeling,
  one-hot/partial cues and temporal representations would be distinct studies.
- Exact Hamming manipulation does not balance action frequencies or repeated
  runs. The generator changes more than an abstract psychological similarity.
- Fixed SGD settings and budgets compare a protocol, not the best achievable
  performance of each architecture. Gate counts and parameter counts differ;
  the h=3 comparison is approximate matching, not identical capacity.
- Larger models may solve this easy full-cue task differently. A change in
  schedule advantage is not a proof of a particular representation mechanism.
- D is randomly selected and its distance is observed, not manipulated. Raw
  forgetting has baseline/floor effects and is not a model of offline biological
  consolidation, sleep, synapses or rehabilitation.
- Run-level resampling and sign-flip assumptions cover this model/generator.
  They do not cover biological validity, every optimizer or every possible task.
  Pointwise CIs are distinct from corrected hypothesis-test families.
- Exploratory comparisons have 20 pairs per original distance, versus 100 in
  the expanded main study. Their uncertainty and selection scope differ.
- v1 outcomes were seen before v2 planning. Earlier pre-confirmatory runs also
  lack producing Git/timestamp provenance; those gaps remain disclosed.
- Numerical reproducibility depends on software/backend. CPU is the reference;
  CUDA is optional and unvalidated on this CPU-only machine. Across-version or
  GPU results should not be promised bit-identical to the recorded environment.
- AI-assisted implementation, documentation and interpretation require the
  owner's independent scientific and code review before v2 release.

Climbing supplies motivation, not measurements. These simulations do not prove
anything about climbing performance or establish a mechanism in the human brain.
