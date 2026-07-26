# F18 Simple-to-Strict-JSBSim Transfer Preflight

The frozen matrix is four tasks, three formal seeds `[0,1,2]`, one episode per seed, and the three declared main methods. A valid paired transfer record must retain task, seed, method, policy/gain artifact hash, control frequency, action contract, metric definitions, and complete backend provenance.

A strict JSBSim record is admissible only when requested/final backend is `jsbsim`, `strict_backend=true`, and `backend_fallback_occurred=false`. A fallback result is never transfer evidence. The corresponding simple run must finish on `simple` with the same non-backend contract.

Static preflight confirms the declared main policy checkpoints are absent, so no execution is attempted and no transfer claim is made. F18 remains `needs_more_evidence`; restore artifact inputs, then run paired evaluations and portable bundles before a transfer conclusion or candidate promotion.