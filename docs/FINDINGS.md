# Current Release Findings

1. Phase827 ReciprocalPC reaches lower validation BPB than its matched GPT
   control at 5,000 updates: `1.4867165192` versus `1.5625470909`.
2. Matched GPT retains a small exact-token accuracy lead: `0.2420239928` versus
   `0.2393999365`, a difference of `0.262406` percentage points.
3. The deficit is concentrated in ranking common short-token alternatives.
   ReciprocalPC does use context strongly; broad context blindness is not the
   measured cause.
4. Exact recurrent/FFN credit improves accuracy only slightly at this horizon,
   so local credit transport is not the sole bottleneck.
5. Fixed output calibration can move individual decisions but has not removed
   the gap robustly. Added recall mechanisms and split lexical roles made the
   measured result worse.
6. Several later representation and objective interventions were duplicate,
   contract-incompatible, or negative. This release therefore preserves the
   accepted Phase827 model without speculative additions.

Next engineering target: first package, reproduce, and profile this exact model.
Then run one preregistered representation/ranking experiment at a time. Promote
only changes that preserve strict credit and improve the measured BPB/accuracy
Pareto frontier on the same sealed rows.
