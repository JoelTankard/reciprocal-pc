# Current Findings

1. Short runs understate PC progress. Both canonical PC models continue
   improving through 50k, although improvement becomes shallow.
2. ReciprocalPC nearly matches the exact-credit oracle at 50k. Its remaining
   Blue gap is `0.006653` BPB and `0.001711` accuracy.
3. Most remaining distance is shared by both PC models. GPT leads Blue by
   `0.081856` BPB and `0.030401` accuracy, so learned reciprocal credit alone is
   not the main remaining bottleneck.
4. Blue is diagnostic, not strict PC. It has zero global loss backprops but uses
   exact local autograd VJPs.
5. ReciprocalPC is the strict target. It uses learned local reciprocal paths,
   no exact block VJPs, and records zero global backprops through 50k.
6. GPT schedule horizon changes its curve. Compare against the fresh 50k GPT
   schedule only; never splice a 20k schedule or draw endpoint lines.
7. Curves must contain every measured 100-step row. Interpolation and mixed run
   horizons previously produced misleading comparisons.

Next target: change shared PC representation/inference dynamics while keeping
ReciprocalPC credit local. Beating Blue first would optimize the wrong gap.
