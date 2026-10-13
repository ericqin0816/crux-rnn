# CruxRNN: questions to discuss with a professor

1. **Why use Hamming distance?** It counts differing action positions directly and gives exact equal distances among A/B/C. It does not measure biomechanics, timing, chunks, or difficulty.
2. **Why pair blocked and interleaved models?** They start from the same pretrained weights and see identical training examples; their within-run difference isolates presentation order for that family.
3. **Why isn't a noisy copy an independent subject?** Copies share a base, model, and training history. The independent random realization is the simulation run.
4. **What does “generalization” mean here?** Reproducing an unseen seven-action base from its complete numeric cue without further training. It does not mean transfer to unobserved movement dynamics or climbing routes.
5. **Why can't this establish biological consolidation?** There is no time-dependent offline process, sleep, synaptic biology, brain measurement, or human outcome in this model.
6. **What would a more realistic follow-up require?** A justified action/timing representation, partial sequential cues, mechanisms such as replay or plasticity constraints, and independent comparison with behavioral or neural data. Added complexity needs testable predictions.
7. **Could action frequencies or repeated-action patterns confound Hamming distance?** Yes. They are recorded but not matched; a follow-up could balance these while holding pairwise distance fixed. Current results describe this exact generation rule.
8. **Why is D secondary?** Its similarity is observed rather than experimentally varied. Random D training tests interference under this protocol, but cannot identify an effect of D distance.
9. **Why 30 independent runs?** The requested fixed sample gives five repeats of each blocked order. It is a design choice, not a power guarantee; uncertainty intervals quantify variability conditional on the model and generator.
10. **What would falsify the hypothesis?** If the proposed claim is that increasing dissimilarity increases the IP advantage, consistently flat or negative run-level trends under this frozen design would contradict it. Categorical nonlinearity could also contradict a simple monotonic account. The study does not assume that IP must win.
11. **Does lower blocked training loss mean better learning?** It describes the currently presented examples. Fixed A/B/C probes and held-out final performance reveal whether earlier sequences are still produced correctly.
12. **What does a bootstrap CI cover?** Variation from resampling independent simulation IDs under this model/generator. It does not cover misspecified architecture, biological uncertainty, or generalization to humans.
13. **Why feed the full sequence in the initial cue?** That is the audited task. The network learns to decode and unfold a complete cue; it is not predicting an action whose identity was never supplied.
14. **Is every action category equally hard?** Numeric cue values impose an ordinal geometry on categories. Arbitrary relabeling or one-hot inputs would be useful sensitivity checks, but would change this reproduction's architecture.
