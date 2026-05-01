Training:
base seeds 1–100
episode seeds deterministisch via SeedSequence(base_seed, worker_index, vector_index, episode_counter)

Evaluation:
explizite eval seeds 101–500
keine Wrapper-Überschreibung

Final Test:
separater Seedbereich, z.B. 501–1000

### Seeding Strategy and Reproducibility

To ensure reproducibility and avoid bias due to stochasticity, a strict separation of random seeds was enforced across training, validation, and testing phases.

#### Seed Separation

Three disjoint seed ranges were used:

* **Training base seeds:** [1–100]
  Each training run is initialized with a distinct base seed.

* **Evaluation (validation) seeds:** [101–500]
  Used exclusively for model selection and validation.

* **Test seeds:** [501–1000]
  Reserved for final performance evaluation and never used during training or model selection.

This separation ensures that no information leakage occurs between training and evaluation phases.

---

#### Episode-Level Seed Generation

During training, each episode is initialized with a unique deterministic seed derived from the training base seed. Episode seeds are generated using NumPy’s `SeedSequence`, ensuring reproducibility and independence across parallel environments.

For each environment instance, the episode seed is computed as:

* Inputs:

  * `base_seed` (training run seed)
  * `worker_index` (RLlib rollout worker ID)
  * `vector_index` (environment index within worker)
  * `episode_counter` (incremented per episode)

* Derivation:

```python
ss = np.random.SeedSequence([
    base_seed,
    worker_index,
    vector_index,
    episode_counter,
])
episode_seed = int(ss.generate_state(1, dtype=np.uint32)[0])
```

This guarantees:

* Deterministic seed sequences per training run
* No seed collisions across workers or environments
* High diversity of environment realizations

During evaluation, no such derivation is applied. Instead, seeds are passed explicitly via `env.reset(seed=...)`, ensuring full control over evaluation conditions.

---

#### Reproducibility Verification

A series of smoke tests were conducted to verify correctness:

* **Determinism:**
  Re-running training with the same base seed produces identical episode seed sequences and identical behavior.

* **Seed Diversity:**
  Different base seeds result in distinct episode seed sequences.

* **Worker Independence:**
  Parallel workers generate different episode seeds even for identical episode indices.

* **Evaluation Integrity:**
  Evaluation seeds are used exactly as specified and are not modified by the wrapper logic.

These tests confirm that the seeding strategy is both reproducible and free of unintended correlations.

---

This design ensures that stochasticity is properly controlled while preserving sufficient variability for robust policy learning and evaluation.
