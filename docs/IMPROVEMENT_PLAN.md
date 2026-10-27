# Audit and next experiments

## Implemented in this iteration

- Removed the second `get_redundancy_mask` definition that silently replaced
  the stack-based teacher. The adjacent optimiser now rebuilds from the same
  mask, including nested cancellations. Standard gate classes are checked so
  a custom operation named `x` is not mistaken for the Pauli X operation.
- Bounded numerical operator verification to eight qubits, added explicit
  equivalent / not_equivalent / inconclusive results, preserved legacy boolean
  compatibility, allowed barriers, and rejected unsupported non-unitary inputs.
- Added a wire-dependency graph with ordered operand ports, inverse-pair
  cancellation across disjoint operations, identity removal, and same-axis
  rotation merging. Rewrites preserve global phase and barrier constraints.
- Added a PyTorch message-passing GNN that ranks legal rewrites instead of
  deleting gates independently. Training uses small synthetic circuits and
  bounded greedy-lookahead targets. This is an experimental policy, not a new
  general-purpose optimiser or a proven learned advantage.
- Added seeded family fixtures, training fingerprints, a held-out entangling
  family, dependency versions, separate optimisation/verification timings,
  common-basis metrics, and accepted-output metrics to benchmark reports.
- Added a Graph Lab page with numerical verification, rewrite traces, and JSON
  report download. Existing CNN workflow remains available.
- Removed fabricated UI fallback metrics, labelled the saved CNN report as
  historical, resolved model paths relative to the repository, and eliminated
  duplicate inference in the CNN interface.
- Rejected oversized CNN input rather than silently truncating it. The legacy
  generator now stops before an entire generated block would exceed 50 gates.
- Changed CNN training from full-dataset steps to mini-batches and used explicit
  weights-only checkpoint loading.
- Added regression, training, benchmark, checkpoint and Streamlit interaction tests.

## First experiment (development run)

Python 3.12; Qiskit 2.6.1; PyTorch 2.14.1; seed 42; 24 training circuits;
5 epochs; 96 training states. Benchmark seed 10000; 16 synthetic circuits
(four per family); three qubits; six generated blocks. The entangling family
was excluded from GNN training. Exact training-circuit fingerprint overlap is
rejected; this does not establish independence from every related pattern.

| Method | Equivalent candidates | Accepted common-basis gate reduction |
| --- | ---: | ---: |
| Adjacent rules | 16/16 | 10.5% |
| Graph greedy | 16/16 | 39.5% |
| Experimental graph GNN | 16/16 | 39.5% |
| Legacy CNN | 12/16 | 0.0% |
| Qiskit level 3 | 16/16 | 61.8% |

The GNN has **not** beaten greedy selection. Qiskit is stronger on this suite.
The CNN has no documented training-corpus provenance, so these circuits cannot
be called unseen for the saved CNN. Rejected proposals retain the original
circuit and contribute zero accepted reduction. Raw candidate reductions are
not counted as successful optimisations.

These deliberately structured fixtures validate the workflow, not commercial
superiority. Counts are normalised to a logical `u,cx` basis; there is no hardware
target, routing comparison, calibration model or experimental fidelity result.
Training time is not included in inference timing. The old saved metrics have
not been overwritten or republished as results from the new implementation.

## Next: evidence before extra architecture

1. Add larger, externally sourced algorithm families (with recorded licences,
   versions and fingerprints); hold out whole families and size ranges.
2. Compare bounded optimisation budgets and multiple seeds; report per-family
   results, uncertainty, two-qubit gates, depth, memory and runtime. Capture
   checkpoint hash and dataset manifest for every published result.
3. Improve the action space with tested commutation identities, additional
   synthesis rewrites and well-defined hardware objectives. The current rewrite
   set is too simple to establish the need for a learned selector.
4. Compare the same rewrite engine with random, greedy and GNN policies under
   equal budgets; include supervised validation and checkpoint selection. Try
   reinforcement learning only if longer rewrite sequences justify it.
5. Add MQT QCEC or another appropriate verification backend behind the same
   status interface. Do not remove numerical size limits without a replacement.
   Measurements, reset, control flow and ancilla/layout semantics require a
   separate specification and appropriate verification techniques.
6. Record CNN checkpoint/data provenance and rerun evaluation after teacher fixes.
   Legacy evaluation still uses the old report schema and terminology; the new
   benchmark CLI is the source of truth for cross-method comparisons.
7. Add target-specific compilation and evaluate actual hardware results with a
   lab partner before making noise, cost, fidelity or business-value claims.
8. Profile candidate enumeration and graph rebuilding before increasing circuit
   sizes. The Python prototype intentionally prioritises correctness over scale.

## OKF and NCM

No OKF specification has been supplied, and text-generation NCM has not been
defined for this circuit task. Neither is implemented or advertised. First
establish a measurable planning or rule-management problem, then specify and
compare the proposed method against a simpler baseline.
