pub struct RingBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    write_idx: usize,
    read_idx: usize,
    size: usize,
}

impl<T, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self {
            buffer: core::array::from_fn(|_| None),
            write_idx: 0,
            read_idx: 0,
            size: 0,
        }
    }
}

impl<T, const N: usize> RingBuffer<T, N> {
    pub fn new() -> Self {
        Self {
            buffer: [const { None }; N],
            write_idx: 0,
            read_idx: 0,
            size: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == N
    }

    /// Physical slot index for the `i`-th oldest item (`0` = oldest).
    fn physical_index(&self, from_oldest: usize) -> Option<usize> {
        if from_oldest >= self.size {
            return None;
        }
        Some((self.read_idx + from_oldest) % N)
    }

    /// `0` = oldest, `len()-1` = newest.
    pub fn get(&self, from_oldest: usize) -> Option<&T> {
        let idx = self.physical_index(from_oldest)?;
        self.buffer[idx].as_ref()
    }

    /// `0` = newest, `1` = previous, etc.
    pub fn get_newest(&self, from_newest: usize) -> Option<&T> {
        if from_newest >= self.size {
            return None;
        }
        self.get(self.size - 1 - from_newest)
    }

    pub fn latest(&self) -> Option<&T> {
        self.get_newest(0)
    }

    pub fn push(&mut self, item: T) {
        if self.size == N {
            // Buffer is full; advance read index to overwrite the oldest item
            self.read_idx = (self.read_idx + 1) % N;
            self.size -= 1;
        }

        self.buffer[self.write_idx] = Some(item);
        self.write_idx = (self.write_idx + 1) % N;
        self.size += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let item = self.buffer[self.read_idx].take();
        self.read_idx = (self.read_idx + 1) % N;
        self.size -= 1;
        item
    }

    pub fn clear(&mut self) {
        self.buffer = [const { None }; N];
        self.write_idx = 0;
        self.read_idx = 0;
        self.size = 0;
    }

    /// Sum of all stored values (chronological order is irrelevant for a sum).
    pub fn sum(&self) -> f32
    where
        T: Copy + Into<f32>,
    {
        let mut total = 0.0;
        for i in 0..self.size {
            if let Some(v) = self.get(i) {
                total += (*v).into();
            }
        }
        total
    }
}

impl<T: Copy, const N: usize> RingBuffer<T, N> {
    /// Copy values oldest → newest into `dst`. Returns number of items written.
    pub fn copy_chronological(&self, dst: &mut [T]) -> usize {
        let n = self.size.min(dst.len());
        for i in 0..n {
            dst[i] = *self.get(i).expect("index in range");
        }
        n
    }
}

/// Population size for one CEM generation (completed episodes).
const POP_SIZE: usize = 12;
/// Wall-clock steps per episode with fixed sampled parameters
/// (includes refractory steps so generation rate stays predictable).
const EPISODE_LEN: usize = 8;
/// Number of top episodes used for the distribution update.
const ELITE_COUNT: usize = 4;

/// Soft-update rates toward elite statistics (0 = freeze, 1 = hard replace).
const LR_MEAN: f32 = 0.55;
const LR_STD: f32 = 0.40;
/// Numerical floor only (not an exploration floor); std may shrink toward this.
const STD_EPS: f32 = 1e-4;
/// Upper clamp to avoid MC explosion.
const STD_MAX: f32 = 2.0;
/// Extra cost when a spike resets the membrane during tracking.
const SPIKE_PENALTY: f32 = 0.2;
/// Mild preference for small |v_rest| (steady-state V ≈ v_rest + I tracks I best at 0).
const V_REST_L2: f32 = 0.01;
/// Monte Carlo draws of (v_rest, v_threshold) per step when noise is active (odd).
const MC_SAMPLES: usize = 5;
/// Below this search stddev, skip MC and use the trial means (fast path).
const MC_DET_THRESH: f32 = 0.04;
/// Fraction of search-dist stddev used as MC noise (keeps features stable).
const MC_NOISE_SCALE: f32 = 0.25;
/// Initial CEM exploration scale (smaller ⇒ cleaner early tracking).
const INIT_STD: f32 = 0.25;
/// On spike, reset both search-dist stddevs to this value (re-opens exploration).
const SPIKE_STD_RESET: f32 = 0.45;
/// Sparse recurrent taps per ensemble unit.
const REC_TAPS: usize = 4;

/// Legacy alias used by a few tests that need a small positive stddev seed.
const STD_MIN: f32 = 0.05;

/// Hidden LIF units in the reservoir ensemble.
const ENSEMBLE_N: usize = 48;
/// Normalized-LMS step size for the linear readout (decays mildly over a run).
const READOUT_LR: f32 = 0.45;
/// L2 weight decay on readout weights.
const READOUT_L2: f32 = 5e-5;
/// Spectral radius target for sparse recurrent weights.
const REC_RADIUS: f32 = 0.5;
/// Clamp membrane features / drives so a spike storm cannot blow up SGD.
const FEATURE_CLIP: f32 = 4.0;
/// Clamp absolute readout weights.
const WEIGHT_CLIP: f32 = 5.0;
/// Online passes over each series (last pass is scored).
const TRAIN_PASSES: usize = 5;
/// Max state dimension across benchmarks / ensemble I/O (one-hot, Lorenz, LM embeds, …).
const MAX_DIMS: usize = 16;

/// Diagonal Gaussian over one learnable scalar.
#[derive(Clone, Copy, Debug)]
pub struct GaussianParam {
    pub mean: f32,
    pub stddev: f32,
}

impl GaussianParam {
    pub fn new(mean: f32, stddev: f32) -> Self {
        Self {
            mean,
            // No exploration floor: only keep stddev positive and bounded above.
            stddev: stddev.clamp(STD_EPS, STD_MAX),
        }
    }

    pub fn sample(&self, z: f32) -> f32 {
        self.mean + z * self.stddev
    }

    /// Set stddev (clamped to [STD_EPS, STD_MAX]).
    pub fn set_stddev(&mut self, stddev: f32) {
        self.stddev = stddev.clamp(STD_EPS, STD_MAX);
    }

    /// CEM soft update from rank-weighted elite samples.
    pub fn update_from_elites(&mut self, elite_values: &[f32], weights: &[f32]) {
        debug_assert_eq!(elite_values.len(), weights.len());
        if elite_values.is_empty() {
            return;
        }

        let mut w_mean = 0.0;
        let mut w_sum = 0.0;
        for (&x, &w) in elite_values.iter().zip(weights.iter()) {
            w_mean += w * x;
            w_sum += w;
        }
        w_mean /= w_sum.max(1e-8);

        let mut w_var = 0.0;
        for (&x, &w) in elite_values.iter().zip(weights.iter()) {
            let d = x - w_mean;
            w_var += w * d * d;
        }
        w_var /= w_sum.max(1e-8);
        let elite_std = w_var.sqrt();

        let old_mean = self.mean;
        self.mean = (1.0 - LR_MEAN) * self.mean + LR_MEAN * w_mean;
        // Follow elite spread and mean travel; no minimum-std floor — exploration
        // is restored on spike via [`SPIKE_STD_RESET`] instead.
        let travel = (self.mean - old_mean).abs();
        let target_std = elite_std.max(travel);
        self.stddev =
            ((1.0 - LR_STD) * self.stddev + LR_STD * target_std).clamp(STD_EPS, STD_MAX);
    }
}

/// Log-rank weights for elites (best index 0). Higher weight on better ranks.
/// Classic CEM/CMA positive weights: w_i ∝ log(λ+½) − log(i+1), normalized.
fn elite_rank_weights(elite_n: usize, out: &mut [f32]) {
    assert!(elite_n <= out.len());
    if elite_n == 0 {
        return;
    }
    let mut sum = 0.0;
    for i in 0..elite_n {
        let w = (elite_n as f32 + 0.5).ln() - ((i + 1) as f32).ln();
        out[i] = w.max(0.0);
        sum += out[i];
    }
    if sum <= 0.0 {
        let u = 1.0 / elite_n as f32;
        for i in 0..elite_n {
            out[i] = u;
        }
    } else {
        for i in 0..elite_n {
            out[i] /= sum;
        }
    }
}

// Define the Leaky Integrate-and-Fire neuron
pub struct LifNeuron {
    pub v_membrane: f32,
    pub v_reset: f32,
    pub tau_m: f32,
    pub is_refractory: bool,

    /// Search distribution over resting potential. Per-step Monte Carlo draws
    /// `v_rest ~ N(trial_v_rest, v_rest_dist.stddev)` (CEM particle × learned scale).
    pub v_rest_dist: GaussianParam,
    /// Search distribution over spike threshold. Paired with `v_rest_dist` in the
    /// per-step Monte Carlo: fire only if a majority of sampled LIF steps spike.
    pub v_threshold_dist: GaussianParam,

    /// Parameters active for the current episode.
    pub trial_v_rest: f32,
    pub trial_v_threshold: f32,
    /// Antithetic partner for the next episode (reduces gradient noise).
    pending_antithetic: Option<(f32, f32)>,

    /// Per-step I/O history (for inspection / scoring).
    pub input: RingBuffer<f32, POP_SIZE>,
    pub output: RingBuffer<f32, POP_SIZE>,
    /// Completed-episode fitness population for CEM.
    pub episode_fitness: RingBuffer<f32, POP_SIZE>,
    pub episode_v_rest: RingBuffer<f32, POP_SIZE>,
    pub episode_v_threshold: RingBuffer<f32, POP_SIZE>,

    /// Accumulators for the open episode.
    episode_error_sum: f32,
    episode_spike_count: u32,
    episode_step: u32,
    pub generation: u64,
    /// Last generation's mean episode fitness (for monitoring).
    pub last_gen_fitness: f32,
    pub rng: Rand,
}

impl LifNeuron {
    pub fn new(v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32) -> Self {
        let mut neuron = Self {
            v_membrane: v_rest,
            v_reset,
            tau_m,
            is_refractory: false,
            v_rest_dist: GaussianParam::new(v_rest, INIT_STD),
            v_threshold_dist: GaussianParam::new(v_threshold, INIT_STD),
            trial_v_rest: v_rest,
            trial_v_threshold: v_threshold,
            pending_antithetic: None,
            input: RingBuffer::new(),
            output: RingBuffer::new(),
            episode_fitness: RingBuffer::new(),
            episode_v_rest: RingBuffer::new(),
            episode_v_threshold: RingBuffer::new(),
            episode_error_sum: 0.0,
            episode_spike_count: 0,
            episode_step: 0,
            generation: 0,
            last_gen_fitness: f32::INFINITY,
            rng: Rand::new(1),
        };
        neuron.resample_trial();
        neuron
    }

    pub fn v_rest(&self) -> f32 {
        self.v_rest_dist.mean
    }

    pub fn v_threshold(&self) -> f32 {
        self.v_threshold_dist.mean
    }

    pub fn v_rest_stddev(&self) -> f32 {
        self.v_rest_dist.stddev
    }

    pub fn v_threshold_stddev(&self) -> f32 {
        self.v_threshold_dist.stddev
    }

    /// Start a new episode: antithetic pair if available, else fresh Gaussian sample.
    fn resample_trial(&mut self) {
        if let Some((rest, thr)) = self.pending_antithetic.take() {
            self.trial_v_rest = rest;
            self.trial_v_threshold = thr;
            return;
        }

        let (z0, z1) = self.rng.g();
        self.trial_v_rest = self.v_rest_dist.sample(z0);
        self.trial_v_threshold = self.v_threshold_dist.sample(z1).max(self.v_reset + 0.05);

        // Antithetic twin mirrors the noise for the following episode.
        self.pending_antithetic = Some((
            self.v_rest_dist.sample(-z0),
            self.v_threshold_dist
                .sample(-z1)
                .max(self.v_reset + 0.05),
        ));
    }

    /// Episode cost: mean squared tracking error + spike disruption + mild prior.
    fn episode_fitness_value(&self) -> f32 {
        let steps = self.episode_step.max(1) as f32;
        let mse = self.episode_error_sum / steps;
        let spike_cost = SPIKE_PENALTY * self.episode_spike_count as f32 / steps;
        let prior = V_REST_L2 * self.trial_v_rest * self.trial_v_rest;
        mse + spike_cost + prior
    }

    fn finish_episode(&mut self) {
        let fitness = self.episode_fitness_value();
        self.episode_fitness.push(fitness);
        self.episode_v_rest.push(self.trial_v_rest);
        self.episode_v_threshold.push(self.trial_v_threshold);

        self.episode_error_sum = 0.0;
        self.episode_spike_count = 0;
        self.episode_step = 0;

        if self.episode_fitness.len() >= POP_SIZE {
            self.cem_update();
        }

        self.resample_trial();
    }

    /// Cross-entropy method with log-rank elite weights and soft Gaussian updates.
    fn cem_update(&mut self) {
        let n = self.episode_fitness.len();
        if n == 0 {
            return;
        }

        let mut order = [0usize; POP_SIZE];
        for i in 0..n {
            order[i] = i;
        }

        // Sort ascending: lower fitness is better.
        let mut swapped = true;
        while swapped {
            swapped = false;
            for i in 0..n.saturating_sub(1) {
                let fa = *self
                    .episode_fitness
                    .get(order[i])
                    .unwrap_or(&f32::INFINITY);
                let fb = *self
                    .episode_fitness
                    .get(order[i + 1])
                    .unwrap_or(&f32::INFINITY);
                if fa > fb {
                    order.swap(i, i + 1);
                    swapped = true;
                }
            }
        }

        let elite_n = ELITE_COUNT.min(n).max(1);
        let mut weights = [0.0f32; POP_SIZE];
        elite_rank_weights(elite_n, &mut weights);

        let mut elite_rest = [0.0f32; POP_SIZE];
        let mut elite_thr = [0.0f32; POP_SIZE];
        for i in 0..elite_n {
            let idx = order[i];
            elite_rest[i] = *self.episode_v_rest.get(idx).unwrap_or(&0.0);
            elite_thr[i] = *self.episode_v_threshold.get(idx).unwrap_or(&0.0);
        }

        self.v_rest_dist
            .update_from_elites(&elite_rest[..elite_n], &weights[..elite_n]);
        self.v_threshold_dist
            .update_from_elites(&elite_thr[..elite_n], &weights[..elite_n]);
        // Keep threshold above reset so the neuron can still spike when useful.
        if self.v_threshold_dist.mean < self.v_reset + 0.1 {
            self.v_threshold_dist.mean = self.v_reset + 0.1;
        }

        // Monitor: average fitness of the whole generation.
        self.last_gen_fitness = self.episode_fitness.sum() / n as f32;
        self.generation += 1;

        // Fresh population next generation.
        self.episode_fitness.clear();
        self.episode_v_rest.clear();
        self.episode_v_threshold.clear();
        // Drop a pending antithetic that was drawn under the old distribution.
        self.pending_antithetic = None;
    }

    /// Score membrane against `score_target` (may differ from the drive current).
    fn record_step(&mut self, drive: f32, score_target: f32) {
        let err = score_target - self.v_membrane;
        self.episode_error_sum += err * err;
        self.episode_step += 1;
        self.input.push(drive);
        self.output.push(self.v_membrane);

        if self.episode_step >= EPISODE_LEN as u32 {
            self.finish_episode();
        }
    }

    /// Drive with `i_input` and score tracking of the same value.
    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
        self.step_with_target(i_input, i_input, dt)
    }

    /// LIF step under drive `drive`, score membrane vs `score_target`.
    ///
    /// Uses a **deterministic** trial-mean update when both search stddevs are
    /// below [`MC_DET_THRESH`]; otherwise Monte Carlo majority vote over
    /// `MC_SAMPLES` draws from
    /// `N(trial_*, v_*_dist.stddev * MC_NOISE_SCALE)`.
    ///
    /// On spike, stddevs reset to [`SPIKE_STD_RESET`].
    pub fn step_with_target(&mut self, drive: f32, score_target: f32, dt: f32) -> bool {
        // Refractory: hold at reset, still score so spikes that wreck the
        // trajectory are charged to the active trial parameters.
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            self.record_step(drive, score_target);
            return false;
        }

        let (v_mean, spiked) = if self.v_rest_dist.stddev <= MC_DET_THRESH
            && self.v_threshold_dist.stddev <= MC_DET_THRESH
        {
            self.deterministic_step(drive, dt)
        } else {
            self.monte_carlo_step(drive, dt)
        };
        self.v_membrane = v_mean;
        if spiked {
            self.is_refractory = true;
            self.episode_spike_count += 1;
            // Re-open parameter exploration after a spike (replaces a min-std floor).
            self.v_rest_dist.set_stddev(SPIKE_STD_RESET);
            self.v_threshold_dist.set_stddev(SPIKE_STD_RESET);
        }

        self.record_step(drive, score_target);
        spiked
    }

    /// Exact subthreshold step with fixed trial parameters (no MC).
    #[inline]
    fn deterministic_step(&self, drive: f32, dt: f32) -> (f32, bool) {
        let decay = (-dt / self.tau_m.max(1e-3)).exp();
        let v_rest = self.trial_v_rest;
        let v_thr = self.trial_v_threshold.max(self.v_reset + 0.05);
        let v_inf = v_rest + drive;
        let v_new = v_inf + (self.v_membrane - v_inf) * decay;
        (v_new, v_new >= v_thr)
    }

    /// Run `MC_SAMPLES` LIF micro-steps with params drawn from
    /// `v_rest_dist` / `v_threshold_dist` (trial mean, scaled learned stddev).
    /// Returns `(mean_post_voltage, majority_fired)`.
    fn monte_carlo_step(&mut self, drive: f32, dt: f32) -> (f32, bool) {
        let v0 = self.v_membrane;
        let decay = (-dt / self.tau_m.max(1e-3)).exp();
        let thr_floor = self.v_reset + 0.05;

        // Episode law: CEM particle as mean; MC noise is a fraction of search std.
        // Stddevs may shrink freely; a spike resets them to SPIKE_STD_RESET.
        let rest_std = (self.v_rest_dist.stddev * MC_NOISE_SCALE).max(STD_EPS);
        let thr_std = (self.v_threshold_dist.stddev * MC_NOISE_SCALE).max(STD_EPS);
        let rest_mean = self.trial_v_rest;
        let thr_mean = self.trial_v_threshold;

        let mut fire_votes = 0usize;
        let mut v_sum = 0.0f32;

        for _ in 0..MC_SAMPLES {
            let (z_rest, z_thr) = self.rng.g();
            let v_rest = rest_mean + z_rest * rest_std;
            let v_thr = (thr_mean + z_thr * thr_std).max(thr_floor);

            // Exact subthreshold step: τ V' = -(V - v_rest) + I
            let v_inf = v_rest + drive;
            let v_new = v_inf + (v0 - v_inf) * decay;
            v_sum += v_new;
            if v_new >= v_thr {
                fire_votes += 1;
            }
        }

        let v_mean = v_sum / MC_SAMPLES as f32;
        let spiked = fire_votes * 2 > MC_SAMPLES;
        (v_mean, spiked)
    }
}

// ---------------------------------------------------------------------------
// Multi-neuron reservoir + linear readout
// ---------------------------------------------------------------------------

/// Bank of LIF units with random input/recurrent projections and an online
/// NLMS readout. Hidden units keep CEM on (v_rest, v_threshold). The readout is
///
///   y = b + W_h V + W_x x
///
/// so linear next-step maps (rotation, one-hot permutation) are learnable
/// even when the reservoir is a pure encoder of the current input.
pub struct LifEnsemble {
    pub units: Vec<LifNeuron>,
    /// w_in[h][d]: input dim d → hidden unit h.
    w_in: Vec<[f32; MAX_DIMS]>,
    /// Sparse recurrent indices: unit h reads `prev_v[rec_idx[h][t]]`.
    rec_idx: Vec<[usize; REC_TAPS]>,
    /// Sparse recurrent weights aligned with `rec_idx`.
    rec_w: Vec<[f32; REC_TAPS]>,
    /// Hidden readout: contribution of membrane V_h to output dim d.
    w_out: Vec<[f32; ENSEMBLE_N]>,
    /// Input skip: contribution of x_i to output dim d (critical for next-step).
    w_skip: Vec<[f32; MAX_DIMS]>,
    /// One-step delay skip: contribution of x_i[t-1] (temporal context).
    w_delay: Vec<[f32; MAX_DIMS]>,
    /// Two-step delay skip: x_i[t-2] (helps multi-sine / slow dynamics).
    w_delay2: Vec<[f32; MAX_DIMS]>,
    bias: [f32; MAX_DIMS],
    prev_v: [f32; ENSEMBLE_N],
    prev_x: [f32; MAX_DIMS],
    prev2_x: [f32; MAX_DIMS],
    in_dims: usize,
    out_dims: usize,
    lr: f32,
    step_count: u32,
}

impl LifEnsemble {
    pub fn new(in_dims: usize, out_dims: usize, seed: u32) -> Self {
        assert!(in_dims >= 1 && in_dims <= MAX_DIMS);
        assert!(out_dims >= 1 && out_dims <= MAX_DIMS);

        let mut rng = Rand::new(seed.max(1));
        let mut units = Vec::with_capacity(ENSEMBLE_N);
        let mut w_in = Vec::with_capacity(ENSEMBLE_N);
        let mut rec_idx = Vec::with_capacity(ENSEMBLE_N);
        let mut rec_w = Vec::with_capacity(ENSEMBLE_N);

        // Diverse membrane time constants (exact integrator is stable for all τ).
        let taus = [3.0f32, 5.0, 7.0, 10.0, 12.0, 16.0, 22.0, 30.0];
        let in_scale = 1.0 / (in_dims as f32).sqrt();

        for h in 0..ENSEMBLE_N {
            let tau = taus[h % taus.len()];
            // Slightly different seeds / inits so CEM populations diverge.
            let v_rest0 = 0.35 * rng.signed();
            let thr0 = 2.0 + 1.0 * rng.u();
            let mut n = LifNeuron::new(v_rest0, thr0, 0.0, tau);
            n.rng = Rand::new(seed.wrapping_mul(2654435761).wrapping_add(h as u32 + 1));
            n.resample_trial();
            units.push(n);

            let mut row_in = [0.0f32; MAX_DIMS];
            for d in 0..in_dims {
                row_in[d] = in_scale * rng.signed();
            }
            // Pure identity taps so tracking tasks stay easy.
            if h < in_dims {
                row_in = [0.0; MAX_DIMS];
                row_in[h] = 1.0;
            }
            // Delayed-style taps: second block copies input with opposite sign /
            // scale for phase-sensitive features.
            if h >= in_dims && h < 2 * in_dims {
                let d = h - in_dims;
                row_in = [0.0; MAX_DIMS];
                row_in[d] = 0.5;
            }
            w_in.push(row_in);

            let mut idx = [0usize; REC_TAPS];
            let mut wts = [0.0f32; REC_TAPS];
            for t in 0..REC_TAPS {
                idx[t] = (rng.u32() as usize) % ENSEMBLE_N;
                wts[t] = 0.4 * rng.signed();
            }
            rec_idx.push(idx);
            rec_w.push(wts);
        }

        // Rescale sparse recurrence toward REC_RADIUS (row L1 proxy).
        let mut max_abs = 1e-6f32;
        for wts in &rec_w {
            let s: f32 = wts.iter().map(|w| w.abs()).sum();
            if s > max_abs {
                max_abs = s;
            }
        }
        let rec_scale = REC_RADIUS / max_abs;
        for wts in &mut rec_w {
            for w in wts.iter_mut() {
                *w *= rec_scale;
            }
        }

        let mut w_out = Vec::with_capacity(out_dims);
        let mut w_skip = Vec::with_capacity(out_dims);
        let mut w_delay = Vec::with_capacity(out_dims);
        let mut w_delay2 = Vec::with_capacity(out_dims);
        for d in 0..out_dims {
            let mut row = [0.0f32; ENSEMBLE_N];
            // Warm-start identity hidden taps toward corresponding outputs.
            if d < ENSEMBLE_N {
                row[d] = 0.35;
            }
            w_out.push(row);

            let mut skip = [0.0f32; MAX_DIMS];
            // Warm-start skip as identity (good for track; next-step adapts).
            if d < in_dims {
                skip[d] = 0.55;
            }
            w_skip.push(skip);
            w_delay.push([0.0; MAX_DIMS]);
            w_delay2.push([0.0; MAX_DIMS]);
        }

        // Start quiet; stddevs re-open to SPIKE_STD_RESET when a unit spikes.
        for n in &mut units {
            n.v_rest_dist.set_stddev(STD_EPS);
            n.v_threshold_dist.set_stddev(STD_EPS);
            n.trial_v_rest = n.v_rest_dist.mean;
            n.trial_v_threshold = n.v_threshold_dist.mean.max(n.v_reset + 0.1);
        }

        Self {
            units,
            w_in,
            rec_idx,
            rec_w,
            w_out,
            w_skip,
            w_delay,
            w_delay2,
            bias: [0.0; MAX_DIMS],
            prev_v: [0.0; ENSEMBLE_N],
            prev_x: [0.0; MAX_DIMS],
            prev2_x: [0.0; MAX_DIMS],
            in_dims,
            out_dims,
            lr: READOUT_LR,
            step_count: 0,
        }
    }

    /// Drive LIF bank and advance delay taps (no readout weight update).
    fn drive_reservoir(&mut self, x: &[f32], dt: f32) {
        debug_assert!(x.len() >= self.in_dims);
        let mut drives = [0.0f32; ENSEMBLE_N];
        for h in 0..ENSEMBLE_N {
            let mut drive = 0.0;
            for d in 0..self.in_dims {
                drive += self.w_in[h][d] * x[d];
            }
            let idx = &self.rec_idx[h];
            let wts = &self.rec_w[h];
            for t in 0..REC_TAPS {
                drive += wts[t] * self.prev_v[idx[t]];
            }
            drives[h] = drive;
        }

        for h in 0..ENSEMBLE_N {
            let drive = drives[h].clamp(-FEATURE_CLIP, FEATURE_CLIP);
            let _ = self.units[h].step_with_target(drive, drive, dt);
            self.prev_v[h] = self.units[h]
                .v_membrane
                .clamp(-FEATURE_CLIP, FEATURE_CLIP);
        }

        for d in 0..self.in_dims {
            self.prev2_x[d] = self.prev_x[d];
            self.prev_x[d] = x[d];
        }
    }

    fn readout_pred(&self, x: &[f32]) -> [f32; MAX_DIMS] {
        let mut pred = [0.0f32; MAX_DIMS];
        for d in 0..self.out_dims {
            let mut y = self.bias[d];
            for h in 0..ENSEMBLE_N {
                y += self.w_out[d][h] * self.prev_v[h];
            }
            for i in 0..self.in_dims {
                y += self.w_skip[d][i] * x[i];
                y += self.w_delay[d][i] * self.prev_x[i];
                y += self.w_delay2[d][i] * self.prev2_x[i];
            }
            pred[d] = y.clamp(-FEATURE_CLIP * 2.0, FEATURE_CLIP * 2.0);
        }
        pred
    }

    /// One reservoir step: drive hidden LIFs, form readout, NLMS on (ŷ − target).
    pub fn step(&mut self, x: &[f32], target: &[f32], dt: f32) -> [f32; MAX_DIMS] {
        debug_assert!(x.len() >= self.in_dims);
        debug_assert!(target.len() >= self.out_dims);

        self.drive_reservoir(x, dt);
        let pred = self.readout_pred(x);

        // Feature energy for normalized LMS.
        let mut energy = 1.0f32;
        for h in 0..ENSEMBLE_N {
            energy += self.prev_v[h] * self.prev_v[h];
        }
        for d in 0..self.in_dims {
            energy += x[d] * x[d]
                + self.prev_x[d] * self.prev_x[d]
                + self.prev2_x[d] * self.prev2_x[d];
        }

        self.step_count = self.step_count.saturating_add(1);
        let lr = self.lr / (1.0 + 0.0008 * self.step_count as f32);
        let inv_norm = 1.0 / energy.max(1e-3);

        for d in 0..self.out_dims {
            let e = (pred[d] - target[d]).clamp(-FEATURE_CLIP, FEATURE_CLIP);
            let step = lr * e * inv_norm;
            self.bias[d] = (self.bias[d] - step).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            for h in 0..ENSEMBLE_N {
                let g = step * self.prev_v[h] + READOUT_L2 * self.w_out[d][h];
                self.w_out[d][h] = (self.w_out[d][h] - g).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            }
            for i in 0..self.in_dims {
                let g_s = step * x[i] + READOUT_L2 * self.w_skip[d][i];
                self.w_skip[d][i] =
                    (self.w_skip[d][i] - g_s).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
                let g_d = step * self.prev_x[i] + READOUT_L2 * self.w_delay[d][i];
                self.w_delay[d][i] =
                    (self.w_delay[d][i] - g_d).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
                let g_d2 = step * self.prev2_x[i] + READOUT_L2 * self.w_delay2[d][i];
                self.w_delay2[d][i] =
                    (self.w_delay2[d][i] - g_d2).clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            }
        }

        pred
    }

    /// Inference step: update dynamics only (no NLMS). Used by generation / MCTS.
    pub fn step_eval(&mut self, x: &[f32], dt: f32) {
        self.drive_reservoir(x, dt);
    }

    /// Membrane features from the last reservoir step (length [`ENSEMBLE_N`]).
    pub fn hidden_state(&self) -> &[f32] {
        &self.prev_v
    }

    pub fn in_dims(&self) -> usize {
        self.in_dims
    }

    pub fn out_dims(&self) -> usize {
        self.out_dims
    }

    /// Snapshot dynamical state for MCTS branching (weights untouched).
    pub fn snapshot_dynamics(&self) -> EnsembleDynamicsSnap {
        let mut membranes = [0.0f32; ENSEMBLE_N];
        let mut refractory = [false; ENSEMBLE_N];
        let mut rng_states = [0u32; ENSEMBLE_N];
        for h in 0..ENSEMBLE_N {
            membranes[h] = self.units[h].v_membrane;
            refractory[h] = self.units[h].is_refractory;
            rng_states[h] = self.units[h].rng.lfsr;
        }
        EnsembleDynamicsSnap {
            prev_v: self.prev_v,
            prev_x: self.prev_x,
            prev2_x: self.prev2_x,
            membranes,
            refractory,
            rng_states,
        }
    }

    pub fn restore_dynamics(&mut self, snap: &EnsembleDynamicsSnap) {
        self.prev_v = snap.prev_v;
        self.prev_x = snap.prev_x;
        self.prev2_x = snap.prev2_x;
        for h in 0..ENSEMBLE_N {
            self.units[h].v_membrane = snap.membranes[h];
            self.units[h].is_refractory = snap.refractory[h];
            self.units[h].rng.lfsr = snap.rng_states[h];
        }
    }

    /// Clear dynamical state (membranes, delays) without wiping learned readout.
    pub fn reset_dynamics(&mut self) {
        self.prev_v = [0.0; ENSEMBLE_N];
        self.prev_x = [0.0; MAX_DIMS];
        self.prev2_x = [0.0; MAX_DIMS];
        for u in &mut self.units {
            u.v_membrane = u.trial_v_rest;
            u.is_refractory = false;
        }
    }
}

/// Pure dynamics of [`LifEnsemble`] (no readout weights).
#[derive(Clone, Debug)]
pub struct EnsembleDynamicsSnap {
    prev_v: [f32; ENSEMBLE_N],
    prev_x: [f32; MAX_DIMS],
    prev2_x: [f32; MAX_DIMS],
    membranes: [f32; ENSEMBLE_N],
    refractory: [bool; ENSEMBLE_N],
    rng_states: [u32; ENSEMBLE_N],
}

// Rand is a random number generator
pub struct Rand {
    pub lfsr: u32,
}

// LFSRMASK is the lfsr polynomial
const LFSRMASK: u32 = 0x80000057;

impl Rand {
    pub fn new(seed: u32) -> Rand {
        Rand { lfsr: seed }
    }

    pub fn u32(&mut self) -> u32 {
        self.lfsr = (self.lfsr >> 1) ^ ((!(self.lfsr & 1)).wrapping_add(1) & LFSRMASK);
        self.lfsr
    }

    pub fn u(&mut self) -> f32 {
        self.u32() as f32 / u32::MAX as f32
    }

    /// Uniform in approximately [-1, 1].
    pub fn signed(&mut self) -> f32 {
        2.0 * self.u() - 1.0
    }

    pub fn g(&mut self) -> (f32, f32) {
        // Box–Muller; reject u1≈0 to avoid ln(0).
        let u1 = self.u().max(1e-7);
        let u2 = self.u();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        (z0, z1)
    }
}

// ---------------------------------------------------------------------------
// Lightweight suffix array (generation-buffer dedup for MCTS)
// ---------------------------------------------------------------------------

/// Sorted suffix starts for a short byte string (generation context is small).
#[derive(Clone, Debug)]
struct LightSuffixArray {
    text: Vec<u8>,
    /// `sa[r]` = start index of the r-th suffix in lexicographic order.
    sa: Vec<usize>,
}

impl LightSuffixArray {
    fn build(text: Vec<u8>) -> Self {
        let n = text.len();
        let mut sa: Vec<usize> = (0..n).collect();
        // O(n² log n) compare — fine for prompt+sample lengths (hundreds of bytes).
        sa.sort_by(|&i, &j| text[i..].cmp(&text[j..]));
        Self { text, sa }
    }

    /// True if `pat` occurs as a substring of `text` (via SA lower-bound).
    fn contains(&self, pat: &[u8]) -> bool {
        if pat.is_empty() {
            return true;
        }
        if pat.len() > self.text.len() || self.sa.is_empty() {
            return false;
        }
        // Lower-bound binary search on suffixes, then scan the matching range.
        let mut lo = 0usize;
        let mut hi = self.sa.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            let suf = &self.text[self.sa[mid]..];
            if suf < pat {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // All suffixes that could start with `pat` are contiguous in SA order.
        let mut i = lo;
        while i < self.sa.len() {
            let suf = &self.text[self.sa[i]..];
            if !suf.starts_with(&pat[..1.min(pat.len())]) && suf > pat {
                break;
            }
            if suf.starts_with(pat) {
                return true;
            }
            // Stop once we've left the pat-prefix block.
            if pat.len() <= suf.len() && suf[..pat.len()] > *pat {
                break;
            }
            i += 1;
        }
        // Fallback: correct for any residual SA-order edge cases (n is small).
        self.text.windows(pat.len()).any(|w| w == pat)
    }

    /// Longest \(L \ge\) [`SA_DEDUP_MIN_LEN`] such that the length-\(L\) ending of
    /// `text ++ [c]` already occurs as a substring starting earlier in `text`.
    /// Returns 0 if no such repeat exists.
    fn duplicate_extension_len(&self, c: u8) -> usize {
        let n = self.text.len();
        if n < SA_DEDUP_MIN_LEN.saturating_sub(1) {
            return 0;
        }
        let max_l = (n + 1).min(SA_DEDUP_MAX_LEN);
        for l in (SA_DEDUP_MIN_LEN..=max_l).rev() {
            // Ending pattern: last (l-1) bytes of text, then c.
            let take = l - 1;
            if take > n {
                continue;
            }
            let mut pat = Vec::with_capacity(l);
            pat.extend_from_slice(&self.text[n - take..n]);
            pat.push(c);
            // Occurrence must lie entirely in the old text (start ≤ n-l).
            // Searching `pat` in `text` is sufficient: a match of length l cannot
            // use the new byte except as the terminal occurrence itself.
            if self.contains(&pat) {
                return l;
            }
        }
        0
    }

    /// Same check without a prebuilt SA (for short path extensions inside MCTS).
    fn duplicate_extension_len_raw(text: &[u8], path: &[u8], c: u8) -> usize {
        let mut s = Vec::with_capacity(text.len() + path.len() + 1);
        s.extend_from_slice(text);
        s.extend_from_slice(path);
        s.push(c);
        let n = s.len();
        let max_l = n.min(SA_DEDUP_MAX_LEN);
        for l in (SA_DEDUP_MIN_LEN..=max_l).rev() {
            let start = n - l;
            let needle = &s[start..];
            // Search needle in s[0..start]
            if start >= l && s[..start].windows(l).any(|w| w == needle) {
                return l;
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Character-level natural language model (LifEnsemble + random-forest readout)
// ---------------------------------------------------------------------------

/// Integration dt for LM ensemble steps.
const LM_DT: f32 = 10.0;
/// Dense char embedding size (must be ≤ [`MAX_DIMS`] for [`LifEnsemble`]).
const LM_EMBED_DIMS: usize = MAX_DIMS;
/// Sliding text block length (chars) over which the adjacency matrix is formed.
const LM_BLOCK_SIZE: usize = 24;
/// Project ensemble membranes to this dim before forming the adjacency Gram
/// (keeps RF feature size manageable: K(K+1)/2 + K).
const LM_ADJ_PROJ: usize = 12;
/// Default training path (Project Gutenberg Shakespeare, eBook #100).
const LM_CORPUS_PATH: &str = "100.txt.utf-8";
/// Characters used when streaming the reservoir for RF feature collection.
const LM_TRAIN_CHARS: usize = 700_000;
/// Held-out window immediately after the train prefix.
const LM_EVAL_CHARS: usize = 40_000;
/// Max labeled pairs fed to the random forest (reservoir still sees full prefix).
const RF_TRAIN_SAMPLES: usize = 56_000;
/// Number of trees in the readout forest.
const RF_N_TREES: usize = 56;
/// Max depth of each decision tree (shallower ⇒ less memorization of id noise).
const RF_MAX_DEPTH: usize = 11;
/// Minimum samples in a leaf (higher ⇒ smoother policy).
const RF_MIN_LEAF: usize = 18;
/// Candidate thresholds tried per feature at each split.
const RF_THR_CANDIDATES: usize = 10;
/// Generated sample length after training.
const LM_SAMPLE_LEN: usize = 140;
/// MCTS simulations per emitted character.
const MCTS_SIMS: usize = 96;
/// PUCT exploration constant (slightly lower ⇒ trust value more).
const MCTS_C_PUCT: f32 = 1.2;
/// Expand only the top-k policy actions at each node.
const MCTS_TOP_K: usize = 14;
/// Stochastic rollout length (chars) after expansion (short ⇒ less noise).
const MCTS_ROLLOUT: usize = 2;
/// Blend weights for shaped policy: RF + bigram + unigram (should sum to 1).
/// Bigram dominates — character LMs live or die by local co-occurrence.
const MCTS_RF_BLEND: f32 = 0.18;
const MCTS_BIGRAM_BLEND: f32 = 0.70;
const MCTS_UNIGRAM_BLEND: f32 = 0.12;
/// Floor probability mass reserved for non-top-k (keeps priors honest).
const MCTS_PRIOR_FLOOR: f32 = 1e-4;
/// Softmax temperature applied to the shaped policy before MCTS/top-k (<1 sharpens).
const MCTS_POLICY_TEMP: f32 = 0.7;
/// Weight of immediate action log-prob vs deeper path in the backup value.
const MCTS_IMMEDIATE_WEIGHT: f32 = 0.65;
/// Min repeated-suffix length that triggers SA dedup (phrase-level, not single chars).
const SA_DEDUP_MIN_LEN: usize = 4;
/// Max pattern length checked when testing a one-byte extension.
const SA_DEDUP_MAX_LEN: usize = 24;
/// Multiplicative prior scale per extra repeated byte beyond [`SA_DEDUP_MIN_LEN`].
const SA_DEDUP_PRIOR_SCALE: f32 = 0.35;
/// Extra MCTS value penalty (nats) when an action creates a long exact repeat.
const SA_DEDUP_VALUE_PENALTY: f32 = 2.0;

// ---------------------------------------------------------------------------
// Spike-emission LM: one LifNeuron per symbol; spike ⇒ emit that symbol
// ---------------------------------------------------------------------------

/// LIF time step for the spike-symbol language model.
/// With τ=5, one step gives v ≈ 0.86·I from rest → peak drive crosses thr=1.
const SPIKE_LM_DT: f32 = 10.0;
/// Membrane time constant (ms-scale abstract units).
const SPIKE_LM_TAU: f32 = 5.0;
/// Rest / reset / threshold for symbol neurons (threshold 1.0 is the fire line).
const SPIKE_LM_V_REST: f32 = 0.0;
const SPIKE_LM_V_RESET: f32 = 0.0;
const SPIKE_LM_V_THR: f32 = 1.0;
/// Keep CEM search noise tiny so symbol neurons use the deterministic LIF path.
const SPIKE_LM_STD: f32 = 0.01;
/// Peak synaptic drive for the MLE-best next symbol (supra-threshold in one step).
const SPIKE_LM_PEAK_DRIVE: f32 = 1.8;
/// Floor drive for the least-likely next under a context (stays subthreshold).
const SPIKE_LM_FLOOR_DRIVE: f32 = 0.05;
/// Softmax temperature when mapping log-probs → drive (lower ⇒ sharper WTA).
const SPIKE_LM_DRIVE_TEMP: f32 = 0.35;
/// Relative scale of second-order (prev2 → next) weights vs first-order.
const SPIKE_LM_W2_SCALE: f32 = 0.4;
/// Max sub-steps waiting for a spontaneous spike before WTA fallback.
const SPIKE_LM_MAX_TICKS: usize = 4;
/// Hebbian learning rate during the online refinement pass.
const SPIKE_LM_LR: f32 = 0.02;
/// Extra drive on the teacher-forced true next during training ticks.
const SPIKE_LM_TEACHER_BOOST: f32 = 1.2;
/// Max tokens for the Hebbian pass (full corpus still seeds n-gram weights).
const SPIKE_LM_HEBB_TOKENS: usize = 80_000;
/// Train-time subsample for reporting accuracy (every k-th pair).
const SPIKE_LM_ACC_STRIDE: usize = 4;

/// Keep printable ASCII + newline (maps curly quotes etc. away upstream).
fn is_lm_byte(b: u8) -> bool {
    b == b'\n' || (32..127).contains(&b)
}

/// Byte vocabulary built from a corpus (unknown bytes map to space).
#[derive(Clone, Debug)]
pub struct CharVocab {
    /// id → byte
    pub id_to_byte: Vec<u8>,
    /// byte → id (`u16::MAX` = missing → space id)
    byte_to_id: [u16; 256],
    space_id: usize,
}

impl CharVocab {
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut present = [false; 256];
        for &b in data {
            if is_lm_byte(b) {
                present[b as usize] = true;
            }
        }
        present[b' ' as usize] = true;
        present[b'\n' as usize] = true;

        let mut id_to_byte = Vec::with_capacity(96);
        let mut byte_to_id = [u16::MAX; 256];
        for b in 0u16..256 {
            if present[b as usize] {
                let id = id_to_byte.len() as u16;
                byte_to_id[b as usize] = id;
                id_to_byte.push(b as u8);
            }
        }
        let space_id = byte_to_id[b' ' as usize] as usize;
        Self {
            id_to_byte,
            byte_to_id,
            space_id,
        }
    }

    pub fn len(&self) -> usize {
        self.id_to_byte.len()
    }

    pub fn encode(&self, b: u8) -> usize {
        let id = self.byte_to_id[b as usize];
        if id == u16::MAX {
            self.space_id
        } else {
            id as usize
        }
    }

    pub fn decode(&self, id: usize) -> u8 {
        self.id_to_byte[id.min(self.id_to_byte.len().saturating_sub(1))]
    }
}

/// Normalize CRLF, map non-printable bytes to space, collapse space runs lightly.
fn load_corpus(path: &str) -> Result<Vec<u8>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let b = raw[i];
        let mapped = if b == b'\r' {
            if i + 1 < raw.len() && raw[i + 1] == b'\n' {
                i += 1;
            }
            b'\n'
        } else if is_lm_byte(b) {
            b
        } else if b != 0 {
            // Curly quotes / other UTF-8 → nearest ASCII-ish substitute.
            b' '
        } else {
            i += 1;
            continue;
        };
        // Avoid huge runs of spaces from stripped multi-byte UTF-8.
        if mapped == b' '
            && out.last().copied() == Some(b' ')
        {
            i += 1;
            continue;
        }
        out.push(mapped);
        i += 1;
    }
    if out.len() < 64 {
        return Err(format!("{path} too short ({} bytes)", out.len()));
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct LmTrainStats {
    pub tokens: usize,
    pub loss: f32,
    pub accuracy: f32,
    /// exp(mean NLL) under the forest class probabilities.
    pub perplexity: f32,
}

// ---------------------------------------------------------------------------
// Random-forest multi-class readout (no external deps)
// ---------------------------------------------------------------------------

/// One node in a CART-style classification tree.
#[derive(Clone, Debug)]
struct RfNode {
    /// `true` ⇒ leaf (use `hist`); `false` ⇒ split on `feature` / `threshold`.
    is_leaf: bool,
    feature: usize,
    threshold: f32,
    left: usize,
    right: usize,
    /// Class histogram at a leaf (length = n_classes). Empty on internal nodes.
    hist: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
struct DecisionTree {
    nodes: Vec<RfNode>,
}

impl DecisionTree {
    fn predict_hist(&self, x: &[f32]) -> &[u32] {
        let mut i = 0usize;
        loop {
            let n = &self.nodes[i];
            if n.is_leaf {
                return &n.hist;
            }
            i = if x[n.feature] <= n.threshold {
                n.left
            } else {
                n.right
            };
        }
    }
}

/// Bagged ensemble of multi-class decision trees.
#[derive(Clone, Debug, Default)]
pub struct RandomForest {
    trees: Vec<DecisionTree>,
    n_classes: usize,
    n_features: usize,
}

impl RandomForest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_trained(&self) -> bool {
        !self.trees.is_empty()
    }

    /// Fit `n_trees` on rows of `x` with labels `y` in `0..n_classes`.
    pub fn fit(
        &mut self,
        x: &[Vec<f32>],
        y: &[usize],
        n_classes: usize,
        n_trees: usize,
        rng: &mut Rand,
    ) {
        assert_eq!(x.len(), y.len());
        assert!(!x.is_empty());
        self.n_classes = n_classes;
        self.n_features = x[0].len();
        self.trees.clear();
        self.trees.reserve(n_trees);

        let n = x.len();
        // √d feature trials per split (at least 1, at most d).
        let mtry = ((self.n_features as f32).sqrt() as usize)
            .max(1)
            .min(self.n_features.max(1));

        // Shared scratch to cut per-split allocations across the forest.
        let mut scratch = RfScratch::new(n_classes, self.n_features, n);

        for _ in 0..n_trees {
            scratch.boot.clear();
            for _ in 0..n {
                scratch.boot.push((rng.u32() as usize) % n);
            }
            // Detach bootstrap indices so `scratch` can be mutably borrowed for splits.
            let boot = core::mem::take(&mut scratch.boot);
            let tree = build_tree(x, y, &boot, n_classes, mtry, rng, &mut scratch);
            scratch.boot = boot;
            self.trees.push(tree);
        }
    }

    /// Soft vote: average leaf class histograms → probabilities.
    pub fn predict_proba(&self, x: &[f32]) -> Vec<f32> {
        let nc = self.n_classes.max(1);
        let mut acc = vec![0.0f32; nc];
        if self.trees.is_empty() {
            let u = 1.0 / nc as f32;
            for p in &mut acc {
                *p = u;
            }
            return acc;
        }
        let inv_t = 1.0 / self.trees.len() as f32;
        for tree in &self.trees {
            let hist = tree.predict_hist(x);
            let mut tot = 0u32;
            for &c in hist {
                tot = tot.wrapping_add(c);
            }
            let inv = inv_t / tot.max(1) as f32;
            let n = hist.len().min(nc);
            for i in 0..n {
                acc[i] += hist[i] as f32 * inv;
            }
        }
        // Laplace smooth so NLL stays finite on rare classes.
        let eps = 1e-4 / nc as f32;
        let mut s = 0.0f32;
        for p in &mut acc {
            *p += eps;
            s += *p;
        }
        let inv_s = 1.0 / s.max(1e-12);
        for p in &mut acc {
            *p *= inv_s;
        }
        acc
    }

    pub fn predict(&self, x: &[f32]) -> usize {
        let p = self.predict_proba(x);
        argmax_f32(&p)
    }
}

fn argmax_f32(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best {
            best = x;
            best_i = i;
        }
    }
    best_i
}

struct RfScratch {
    boot: Vec<usize>,
    feat_order: Vec<usize>,
    left_hist: Vec<u32>,
    right_hist: Vec<u32>,
    parent_hist: Vec<u32>,
    left_idx: Vec<usize>,
    right_idx: Vec<usize>,
}

impl RfScratch {
    fn new(n_classes: usize, n_features: usize, n_samples: usize) -> Self {
        Self {
            boot: Vec::with_capacity(n_samples),
            feat_order: (0..n_features).collect(),
            left_hist: vec![0; n_classes],
            right_hist: vec![0; n_classes],
            parent_hist: vec![0; n_classes],
            left_idx: Vec::with_capacity(n_samples),
            right_idx: Vec::with_capacity(n_samples),
        }
    }
}

fn fill_hist(y: &[usize], idx: &[usize], n_classes: usize, out: &mut [u32]) -> u32 {
    out.fill(0);
    let mut total = 0u32;
    for &i in idx {
        let c = y[i];
        if c < n_classes {
            out[c] += 1;
            total += 1;
        }
    }
    total
}

fn gini(hist: &[u32], total: u32) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let inv = 1.0 / total as f32;
    let mut s = 0.0f32;
    for &c in hist {
        let p = c as f32 * inv;
        s += p * p;
    }
    1.0 - s
}

fn build_tree(
    x: &[Vec<f32>],
    y: &[usize],
    idx: &[usize],
    n_classes: usize,
    mtry: usize,
    rng: &mut Rand,
    scratch: &mut RfScratch,
) -> DecisionTree {
    let mut nodes = Vec::new();
    build_node(x, y, idx, n_classes, mtry, 0, rng, &mut nodes, scratch);
    DecisionTree { nodes }
}

fn build_node(
    x: &[Vec<f32>],
    y: &[usize],
    idx: &[usize],
    n_classes: usize,
    mtry: usize,
    depth: usize,
    rng: &mut Rand,
    nodes: &mut Vec<RfNode>,
    scratch: &mut RfScratch,
) -> usize {
    let total = fill_hist(y, idx, n_classes, &mut scratch.parent_hist);
    let pure = scratch.parent_hist.iter().filter(|&&c| c > 0).count() <= 1;
    let stop = pure
        || depth >= RF_MAX_DEPTH
        || idx.len() <= RF_MIN_LEAF
        || total as usize <= RF_MIN_LEAF;

    let me = nodes.len();
    nodes.push(RfNode {
        is_leaf: true,
        feature: 0,
        threshold: 0.0,
        left: 0,
        right: 0,
        hist: scratch.parent_hist.clone(),
    });

    if stop || x.is_empty() {
        return me;
    }

    let n_features = x[0].len();
    let m = mtry.min(n_features);
    // Partial Fisher–Yates into feat_order[0..m].
    if scratch.feat_order.len() != n_features {
        scratch.feat_order = (0..n_features).collect();
    }
    for i in 0..m {
        let j = i + (rng.u32() as usize) % (n_features - i);
        scratch.feat_order.swap(i, j);
    }

    let parent_gini = gini(&scratch.parent_hist, total);
    let mut best_gain = 0.0f32;
    let mut best_feat = 0usize;
    let mut best_thr = 0.0f32;

    for fi in 0..m {
        let f = scratch.feat_order[fi];
        for _ in 0..RF_THR_CANDIDATES {
            let s = idx[(rng.u32() as usize) % idx.len()];
            let thr = x[s][f];
            // Count-only pass (no index materialization until a split wins).
            scratch.left_hist.fill(0);
            let mut lt = 0u32;
            let mut rt = 0u32;
            for &i in idx {
                let c = y[i];
                if x[i][f] <= thr {
                    if c < n_classes {
                        scratch.left_hist[c] += 1;
                        lt += 1;
                    }
                } else if c < n_classes {
                    rt += 1;
                }
            }
            if (lt as usize) < RF_MIN_LEAF || (rt as usize) < RF_MIN_LEAF {
                continue;
            }
            // right_hist = parent - left
            for c in 0..n_classes {
                scratch.right_hist[c] = scratch.parent_hist[c].saturating_sub(scratch.left_hist[c]);
            }
            let gain = parent_gini
                - (lt as f32 / total as f32) * gini(&scratch.left_hist, lt)
                - (rt as f32 / total as f32) * gini(&scratch.right_hist, rt);
            if gain > best_gain {
                best_gain = gain;
                best_feat = f;
                best_thr = thr;
            }
        }
    }

    if best_gain <= 1e-8 {
        return me;
    }

    // Materialize best split indices once.
    scratch.left_idx.clear();
    scratch.right_idx.clear();
    for &i in idx {
        if x[i][best_feat] <= best_thr {
            scratch.left_idx.push(i);
        } else {
            scratch.right_idx.push(i);
        }
    }
    if scratch.left_idx.len() < RF_MIN_LEAF || scratch.right_idx.len() < RF_MIN_LEAF {
        return me;
    }

    // Clone index sets for recursion (scratch buffers reused deeper).
    let left_idx = scratch.left_idx.clone();
    let right_idx = scratch.right_idx.clone();
    let left_i = build_node(x, y, &left_idx, n_classes, mtry, depth + 1, rng, nodes, scratch);
    let right_i = build_node(x, y, &right_idx, n_classes, mtry, depth + 1, rng, nodes, scratch);
    nodes[me] = RfNode {
        is_leaf: false,
        feature: best_feat,
        threshold: best_thr,
        left: left_i,
        right: right_i,
        hist: Vec::new(),
    };
    me
}

/// Next-character LM: text is processed in **blocks** through [`LifEnsemble`];
/// a unit–unit **adjacency matrix** is accumulated from membrane co-activation
/// and vectorized as the embedding for the **random forest** classifier.
pub struct LifLanguageModel {
    pub vocab: CharVocab,
    /// Shared LIF bank + NLMS embedding tracker (`LM_EMBED_DIMS` I/O).
    pub ensemble: LifEnsemble,
    /// Fixed random embedding table: vocab id → dense vector ≤ [`MAX_DIMS`].
    embed: Vec<[f32; MAX_DIMS]>,
    prev_id: usize,
    prev2_id: usize,
    /// Ring buffer of the last [`LM_BLOCK_SIZE`] **projected** membrane vectors.
    v_block: Vec<[f32; LM_ADJ_PROJ]>,
    /// Running sum of outer products \(S = \sum_t z_t z_t^\top\) (row-major K×K).
    gram_sum: [f32; LM_ADJ_PROJ * LM_ADJ_PROJ],
    /// Fixed random projection ENSEMBLE_N → LM_ADJ_PROJ for adjacency features.
    adj_proj: [[f32; ENSEMBLE_N]; LM_ADJ_PROJ],
    /// Number of valid entries currently in `v_block` (≤ LM_BLOCK_SIZE).
    v_block_len: usize,
    /// Write index into the ring.
    v_block_pos: usize,
    /// Laplace-smoothed unigram P(char) from the train stream (policy prior blend).
    unigram: Vec<f32>,
    /// Laplace-smoothed bigram P(next|prev): `bigram[prev][next]`.
    bigram: Vec<Vec<f32>>,
    /// Multi-class forest over **adjacency embeddings** (+ light char meta).
    pub forest: RandomForest,
    rng: Rand,
}

impl LifLanguageModel {
    pub fn new(vocab: CharVocab, seed: u32) -> Self {
        assert!(LM_EMBED_DIMS >= 1 && LM_EMBED_DIMS <= MAX_DIMS);
        assert!(LM_BLOCK_SIZE >= 2);
        let v = vocab.len();
        let mut rng = Rand::new(seed.max(1));

        // Random unit-scale embeddings in the ensemble's input space.
        let mut embed = Vec::with_capacity(v);
        let scale = 1.0 / (LM_EMBED_DIMS as f32).sqrt();
        for _ in 0..v {
            let mut e = [0.0f32; MAX_DIMS];
            for d in 0..LM_EMBED_DIMS {
                e[d] = scale * rng.signed();
            }
            embed.push(e);
        }

        let ensemble = LifEnsemble::new(LM_EMBED_DIMS, LM_EMBED_DIMS, seed);
        let vn = v.max(1);
        let unigram = vec![1.0 / vn as f32; vn];
        let bigram = vec![vec![1.0 / vn as f32; vn]; vn];

        // Random ±1/√H projection for compact adjacency embeddings.
        let inv_h = 1.0 / (ENSEMBLE_N as f32).sqrt();
        let mut adj_proj = [[0.0f32; ENSEMBLE_N]; LM_ADJ_PROJ];
        for k in 0..LM_ADJ_PROJ {
            for h in 0..ENSEMBLE_N {
                adj_proj[k][h] = if rng.u() < 0.5 { -inv_h } else { inv_h };
            }
        }

        Self {
            vocab,
            ensemble,
            embed,
            prev_id: 0,
            prev2_id: 0,
            v_block: vec![[0.0; LM_ADJ_PROJ]; LM_BLOCK_SIZE],
            gram_sum: [0.0; LM_ADJ_PROJ * LM_ADJ_PROJ],
            adj_proj,
            v_block_len: 0,
            v_block_pos: 0,
            unigram,
            bigram,
            forest: RandomForest::new(),
            rng,
        }
    }

    #[inline]
    fn gram_add_outer(gram: &mut [f32; LM_ADJ_PROJ * LM_ADJ_PROJ], z: &[f32; LM_ADJ_PROJ], sign: f32) {
        let k = LM_ADJ_PROJ;
        for i in 0..k {
            let zi = z[i] * sign;
            let row = i * k;
            for j in 0..k {
                gram[row + j] += zi * z[j];
            }
        }
    }

    /// Project membranes → K-D, update ring + **incremental** Gram sum \(S\).
    fn record_block_state(&mut self) {
        let h = self.ensemble.hidden_state();
        let mut z = [0.0f32; LM_ADJ_PROJ];
        let n = h.len().min(ENSEMBLE_N);
        for k in 0..LM_ADJ_PROJ {
            let mut s = 0.0f32;
            let row = &self.adj_proj[k];
            for j in 0..n {
                s += row[j] * h[j];
            }
            z[k] = s;
        }
        // Sliding window: remove the vector we're about to overwrite.
        if self.v_block_len == LM_BLOCK_SIZE {
            let old = self.v_block[self.v_block_pos];
            Self::gram_add_outer(&mut self.gram_sum, &old, -1.0);
        }
        Self::gram_add_outer(&mut self.gram_sum, &z, 1.0);
        self.v_block[self.v_block_pos] = z;
        self.v_block_pos = (self.v_block_pos + 1) % LM_BLOCK_SIZE;
        if self.v_block_len < LM_BLOCK_SIZE {
            self.v_block_len += 1;
        }
    }

    /// Adjacency embedding from the running Gram \(A = S / T\).
    ///
    /// Packs upper triangle (incl. diagonal) + row-sum degrees, L2-normalized.
    /// O(K²) — does **not** rescan the block.
    fn adjacency_embedding(&self) -> Vec<f32> {
        let k = LM_ADJ_PROJ;
        let tri = k * (k + 1) / 2;
        if self.v_block_len == 0 {
            return vec![0.0; tri + k];
        }
        let inv_t = 1.0 / self.v_block_len as f32;
        let mut emb = Vec::with_capacity(tri + k);
        let mut deg = [0.0f32; LM_ADJ_PROJ];
        for i in 0..k {
            let row = i * k;
            let mut d = 0.0f32;
            for j in 0..k {
                let aij = self.gram_sum[row + j] * inv_t;
                d += aij.abs();
                if j >= i {
                    emb.push(aij);
                }
            }
            deg[i] = d;
        }
        emb.extend_from_slice(&deg);

        let mut nrm = 0.0f32;
        for &x in &emb {
            nrm += x * x;
        }
        nrm = nrm.sqrt().max(1e-6);
        let inv_n = 1.0 / nrm;
        for x in &mut emb {
            *x *= inv_n;
        }
        emb
    }

    /// Linguistic / byte-shape features (ordered better than raw vocab ids for trees).
    fn push_char_meta(f: &mut Vec<f32>, b: u8) {
        f.push(if b.is_ascii_lowercase() { 1.0 } else { 0.0 });
        f.push(if b.is_ascii_uppercase() { 1.0 } else { 0.0 });
        f.push(if b.is_ascii_alphabetic() { 1.0 } else { 0.0 });
        f.push(if b.is_ascii_digit() { 1.0 } else { 0.0 });
        f.push(if b == b' ' { 1.0 } else { 0.0 });
        f.push(if b == b'\n' { 1.0 } else { 0.0 });
        f.push(if matches!(b, b'.' | b',' | b';' | b'!' | b'?' | b':' | b'\'') {
            1.0
        } else {
            0.0
        });
        f.push(if matches!(
            b,
            b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U'
        ) {
            1.0
        } else {
            0.0
        });
        // Smooth byte code in [0,1] (letters cluster; better than sparse id index).
        f.push(b as f32 / 127.0);
    }

    fn embed_char(&self, char_id: usize) -> [f32; MAX_DIMS] {
        self.embed[char_id.min(self.embed.len().saturating_sub(1))]
    }

    /// Drive the ensemble with the char embedding; optionally track next embed (train).
    /// Always appends the post-step membrane vector to the sliding block.
    fn drive_char(&mut self, char_id: usize, target_id: Option<usize>) {
        let x = self.embed_char(char_id);
        match target_id {
            Some(id) => {
                let t = self.embed_char(id);
                let _ = self.ensemble.step(&x, &t, LM_DT);
            }
            None => {
                // Inference / MCTS: dynamics only, freeze NLMS weights.
                self.ensemble.step_eval(&x, LM_DT);
            }
        }
        self.record_block_state();
    }

    /// Feature vector: **adjacency embedding** of the current text block + light char meta.
    fn features(&self, char_id: usize) -> Vec<f32> {
        let mut f = self.adjacency_embedding();
        let cur = self.vocab.decode(char_id);
        let prev = self.vocab.decode(self.prev_id);
        let prev2 = self.vocab.decode(self.prev2_id);
        Self::push_char_meta(&mut f, cur);
        Self::push_char_meta(&mut f, prev);
        Self::push_char_meta(&mut f, prev2);
        f
    }

    /// Shape RF probabilities for search: RF + bigram + unigram, then SA dedup.
    ///
    /// When `sa` is `Some`, candidates that would extend a long exact repeat of
    /// the generation buffer are multiplicatively down-weighted.
    fn shape_policy(
        &self,
        rf: Vec<f32>,
        last_char: usize,
        sa: Option<&LightSuffixArray>,
    ) -> Vec<f32> {
        let n = rf.len();
        if n == 0 {
            return rf;
        }
        let mut p = vec![0.0f32; n];
        let prev = last_char.min(self.bigram.len().saturating_sub(1));
        for i in 0..n {
            let u = self.unigram.get(i).copied().unwrap_or(1.0 / n as f32);
            let bi = self
                .bigram
                .get(prev)
                .and_then(|row| row.get(i))
                .copied()
                .unwrap_or(u);
            let r = rf.get(i).copied().unwrap_or(u);
            p[i] = MCTS_RF_BLEND * r + MCTS_BIGRAM_BLEND * bi + MCTS_UNIGRAM_BLEND * u;
        }

        // Suffix-array phrase dedup on one-byte extensions of the buffer.
        if let Some(sa) = sa {
            for i in 0..n {
                let b = self.vocab.decode(i);
                let rep = sa.duplicate_extension_len(b);
                if rep >= SA_DEDUP_MIN_LEN {
                    let extra = (rep - SA_DEDUP_MIN_LEN + 1) as f32;
                    p[i] *= SA_DEDUP_PRIOR_SCALE.powf(extra);
                }
            }
        }

        // Temperature sharpening of the mixture prior.
        let inv_t = 1.0 / MCTS_POLICY_TEMP.max(1e-3);
        let mut maxv = f32::NEG_INFINITY;
        for &x in &p {
            let z = x.max(1e-12).ln() * inv_t;
            if z > maxv {
                maxv = z;
            }
        }
        let mut s = 0.0f32;
        for x in &mut p {
            *x = ((*x).max(1e-12).ln() * inv_t - maxv).exp();
            s += *x;
        }
        if s > 0.0 {
            for x in &mut p {
                *x /= s;
            }
        }
        p
    }

    fn reset_state(&mut self) {
        self.ensemble.reset_dynamics();
        self.prev_id = 0;
        self.prev2_id = 0;
        self.v_block_len = 0;
        self.v_block_pos = 0;
        self.gram_sum = [0.0; LM_ADJ_PROJ * LM_ADJ_PROJ];
        for row in &mut self.v_block {
            *row = [0.0; LM_ADJ_PROJ];
        }
    }

    fn snapshot_lm(&self) -> LmDynamicsSnap {
        LmDynamicsSnap {
            ensemble: self.ensemble.snapshot_dynamics(),
            prev_id: self.prev_id,
            prev2_id: self.prev2_id,
            v_block: self.v_block.clone(),
            gram_sum: self.gram_sum,
            v_block_len: self.v_block_len,
            v_block_pos: self.v_block_pos,
            rng_lfsr: self.rng.lfsr,
        }
    }

    fn restore_lm(&mut self, snap: &LmDynamicsSnap) {
        self.ensemble.restore_dynamics(&snap.ensemble);
        self.prev_id = snap.prev_id;
        self.prev2_id = snap.prev2_id;
        self.v_block.clone_from(&snap.v_block);
        self.gram_sum = snap.gram_sum;
        self.v_block_len = snap.v_block_len;
        self.v_block_pos = snap.v_block_pos;
        self.rng.lfsr = snap.rng_lfsr;
    }

    /// Raw RF (or unigram) policy — used for eval metrics.
    fn rf_policy(&self, char_id: usize) -> Vec<f32> {
        let feats = self.features(char_id);
        if self.forest.is_trained() {
            self.forest.predict_proba(&feats)
        } else {
            self.unigram.clone()
        }
    }

    /// Search policy: RF + n-gram blend, optionally SA-deduped against `sa`.
    fn policy_from_features(
        &self,
        char_id: usize,
        sa: Option<&LightSuffixArray>,
    ) -> Vec<f32> {
        self.shape_policy(self.rf_policy(char_id), char_id, sa)
    }

    /// Drive ensemble on `char_id`, return **raw** RF probs (for eval).
    pub fn observe(&mut self, char_id: usize) -> Vec<f32> {
        self.drive_char(char_id, None);
        let probs = self.rf_policy(char_id);
        self.prev2_id = self.prev_id;
        self.prev_id = char_id;
        probs
    }

    /// Like [`observe`] but returns the shaped search policy (generation / MCTS).
    /// `sa` is the suffix array of the committed generation buffer for dedup.
    fn observe_search(
        &mut self,
        char_id: usize,
        sa: Option<&LightSuffixArray>,
    ) -> Vec<f32> {
        self.drive_char(char_id, None);
        let probs = self.policy_from_features(char_id, sa);
        self.prev2_id = self.prev_id;
        self.prev_id = char_id;
        probs
    }

    fn sample_from_probs(&mut self, probs: &[f32], temperature: f32) -> usize {
        if temperature <= 1e-6 {
            return argmax_f32(probs);
        }
        let inv_t = 1.0 / temperature.max(1e-3);
        let mut logits: Vec<f32> = probs
            .iter()
            .map(|p| p.max(1e-12).ln() * inv_t)
            .collect();
        let mut max = f32::NEG_INFINITY;
        for &z in &logits {
            if z > max {
                max = z;
            }
        }
        let mut sum = 0.0f32;
        for z in &mut logits {
            *z = (*z - max).exp();
            sum += *z;
        }
        let inv = 1.0 / sum.max(1e-12);
        for z in &mut logits {
            *z *= inv;
        }
        let mut r = self.rng.u();
        let mut pick = logits.len() - 1;
        for (i, &p) in logits.iter().enumerate() {
            if r < p {
                pick = i;
                break;
            }
            r -= p;
        }
        pick
    }

    /// Top-k action indices by probability (descending).
    fn top_k_actions(probs: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        idx.truncate(k.min(probs.len()));
        idx
    }

    /// Choose next char with MCTS (PUCT + RF/n-gram prior + **suffix-array dedup**).
    ///
    /// `gen_text` is the committed generation buffer (prompt + emitted so far).
    /// Actions that would complete a repeated substring of length ≥
    /// [`SA_DEDUP_MIN_LEN`] are down-weighted in the prior and value.
    fn mcts_select_action(
        &mut self,
        last_char: usize,
        temperature: f32,
        gen_text: &[u8],
    ) -> usize {
        let root_snap = self.snapshot_lm();
        let sa = LightSuffixArray::build(gen_text.to_vec());
        let root_prior = self.policy_from_features(last_char, Some(&sa));
        let root_actions = Self::top_k_actions(&root_prior, MCTS_TOP_K);

        let mut nodes: Vec<MctsNode> = vec![MctsNode {
            action: 0,
            prior: 1.0,
            n: 0.0,
            w: 0.0,
            children: Vec::new(),
            unexpanded: root_actions
                .iter()
                .map(|&a| (a, root_prior[a].max(1e-8)))
                .collect(),
        }];

        for _ in 0..MCTS_SIMS {
            self.restore_lm(&root_snap);
            let mut path: Vec<usize> = vec![0];
            let mut node = 0usize;
            let mut path_logp = 0.0f32;
            let mut cur_char = last_char;
            let mut path_bytes: Vec<u8> = Vec::new();

            // Selection: PUCT among fully expanded interiors.
            while nodes[node].unexpanded.is_empty() && !nodes[node].children.is_empty() {
                let parent_n = nodes[node].n.max(1.0);
                let mut best_child = nodes[node].children[0];
                let mut best_score = f32::NEG_INFINITY;
                for &ch in &nodes[node].children {
                    let c = &nodes[ch];
                    let q = if c.n > 0.0 { c.w / c.n } else { 0.0 };
                    let u = MCTS_C_PUCT * c.prior * parent_n.sqrt() / (1.0 + c.n);
                    let score = q + u;
                    if score > best_score {
                        best_score = score;
                        best_child = ch;
                    }
                }
                let act = nodes[best_child].action;
                let probs = self.policy_from_features(cur_char, Some(&sa));
                let mut lp = probs.get(act).copied().unwrap_or(1e-12).max(1e-12).ln();
                let b = self.vocab.decode(act);
                let rep = LightSuffixArray::duplicate_extension_len_raw(gen_text, &path_bytes, b);
                if rep >= SA_DEDUP_MIN_LEN {
                    lp -= SA_DEDUP_VALUE_PENALTY
                        * (rep - SA_DEDUP_MIN_LEN + 1) as f32;
                }
                path_logp += lp;
                let _ = self.observe_search(act, Some(&sa));
                path_bytes.push(b);
                cur_char = act;
                path.push(best_child);
                node = best_child;
            }

            // Expansion: open one untried action (highest prior).
            let mut first_step_lp = 0.0f32;
            if !nodes[node].unexpanded.is_empty() {
                let mut pick = 0usize;
                let mut best_p = -1.0f32;
                for (i, &(_, p)) in nodes[node].unexpanded.iter().enumerate() {
                    if p > best_p {
                        best_p = p;
                        pick = i;
                    }
                }
                let (act, prior) = nodes[node].unexpanded.swap_remove(pick);
                let probs = self.policy_from_features(cur_char, Some(&sa));
                let mut lp = probs.get(act).copied().unwrap_or(1e-12).max(1e-12).ln();
                let b = self.vocab.decode(act);
                let rep = LightSuffixArray::duplicate_extension_len_raw(gen_text, &path_bytes, b);
                if rep >= SA_DEDUP_MIN_LEN {
                    lp -= SA_DEDUP_VALUE_PENALTY
                        * (rep - SA_DEDUP_MIN_LEN + 1) as f32;
                }
                if path.len() == 1 {
                    first_step_lp = lp;
                }
                path_logp += lp;
                let _ = self.observe_search(act, Some(&sa));
                path_bytes.push(b);
                cur_char = act;

                // Deeper priors: rebuild SA on gen_text+path for accurate dedup.
                let mut ext = gen_text.to_vec();
                ext.extend_from_slice(&path_bytes);
                let sa_ext = LightSuffixArray::build(ext);
                let child_prior = self.policy_from_features(cur_char, Some(&sa_ext));
                let child_actions = Self::top_k_actions(&child_prior, MCTS_TOP_K);
                let child = nodes.len();
                nodes.push(MctsNode {
                    action: act,
                    prior: prior.max(MCTS_PRIOR_FLOOR),
                    n: 0.0,
                    w: 0.0,
                    children: Vec::new(),
                    unexpanded: child_actions
                        .iter()
                        .map(|&a| (a, child_prior[a].max(MCTS_PRIOR_FLOOR)))
                        .collect(),
                });
                nodes[node].children.push(child);
                path.push(child);
            }

            // Short rollout.
            let mut rollout_logp = 0.0f32;
            let mut rc = cur_char;
            let roll_temp = temperature.clamp(0.5, 0.75);
            for _ in 0..MCTS_ROLLOUT {
                let probs = self.policy_from_features(rc, Some(&sa));
                let a = self.sample_from_probs(&probs, roll_temp);
                let mut lp = probs.get(a).copied().unwrap_or(1e-12).max(1e-12).ln();
                let b = self.vocab.decode(a);
                let rep = LightSuffixArray::duplicate_extension_len_raw(gen_text, &path_bytes, b);
                if rep >= SA_DEDUP_MIN_LEN {
                    lp -= SA_DEDUP_VALUE_PENALTY
                        * (rep - SA_DEDUP_MIN_LEN + 1) as f32;
                }
                rollout_logp += lp;
                let _ = self.observe_search(a, Some(&sa));
                path_bytes.push(b);
                rc = a;
            }

            let deep_steps = (path.len().saturating_sub(1) + MCTS_ROLLOUT).max(1) as f32;
            let deep = (path_logp + rollout_logp) / deep_steps;
            let value = if first_step_lp != 0.0 {
                MCTS_IMMEDIATE_WEIGHT * first_step_lp
                    + (1.0 - MCTS_IMMEDIATE_WEIGHT) * deep
            } else {
                deep
            };

            for &ni in path.iter().rev() {
                nodes[ni].n += 1.0;
                nodes[ni].w += value;
            }
        }

        self.restore_lm(&root_snap);

        if nodes[0].children.is_empty() {
            return argmax_f32(&root_prior);
        }
        let mut best_a = nodes[nodes[0].children[0]].action;
        let mut best_score = f32::NEG_INFINITY;
        for &ch in &nodes[0].children {
            let c = &nodes[ch];
            let q = if c.n > 0.0 { c.w / c.n } else { f32::NEG_INFINITY };
            let score = c.n + 0.35 * q + 0.5 * c.prior.ln();
            if score > best_score {
                best_score = score;
                best_a = c.action;
            }
        }
        best_a
    }

    /// Stream text in sliding **blocks** through [`LifEnsemble`], form adjacency
    /// embeddings, and fit the random-forest readout.
    ///
    /// Each step teacher-forces the next-char embedding into the ensemble; after
    /// the block ring is warm (`LM_BLOCK_SIZE` steps), features = vectorized
    /// co-activation adjacency + char meta, label = next character.
    pub fn train_bytes(&mut self, data: &[u8], _epochs: usize) -> LmTrainStats {
        if data.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }

        self.reset_state();
        let n_pairs = data.len() - 1;
        let stride = (n_pairs / RF_TRAIN_SAMPLES.max(1)).max(1);
        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(RF_TRAIN_SAMPLES);
        let mut ys: Vec<usize> = Vec::with_capacity(RF_TRAIN_SAMPLES);

        // Corpus n-grams for MCTS prior blending.
        let v = self.vocab.len().max(1);
        let mut uni = vec![1.0f32; v]; // Laplace
        let mut bi = vec![vec![1.0f32; v]; v];
        for i in 0..data.len().saturating_sub(1) {
            let a = self.vocab.encode(data[i]);
            let b = self.vocab.encode(data[i + 1]);
            if a < v {
                uni[a] += 1.0;
            }
            if a < v && b < v {
                bi[a][b] += 1.0;
            }
        }
        // last char still counts for unigram
        if let Some(&last) = data.last() {
            let id = self.vocab.encode(last);
            if id < v {
                uni[id] += 1.0;
            }
        }
        let uni_tot: f32 = uni.iter().sum::<f32>().max(1.0);
        self.unigram = uni.iter().map(|c| c / uni_tot).collect();
        self.bigram = bi
            .iter()
            .map(|row| {
                let t: f32 = row.iter().sum::<f32>().max(1.0);
                row.iter().map(|c| c / t).collect()
            })
            .collect();

        // Need a full block before adjacency embeddings are meaningful.
        let warmup = LM_BLOCK_SIZE.max((n_pairs / 25).min(15_000));
        for i in 0..n_pairs {
            let a = self.vocab.encode(data[i]);
            let b = self.vocab.encode(data[i + 1]);
            // Teacher-force next embedding; records membrane into the block ring.
            self.drive_char(a, Some(b));
            if i >= warmup
                && self.v_block_len >= LM_BLOCK_SIZE
                && i % stride == 0
                && xs.len() < RF_TRAIN_SAMPLES
            {
                xs.push(self.features(a));
                ys.push(b);
            }
            self.prev2_id = self.prev_id;
            self.prev_id = a;
        }

        let n_classes = self.vocab.len();
        self.forest
            .fit(&xs, &ys, n_classes, RF_N_TREES, &mut self.rng);

        // Cheap in-bag estimate on a stride of fit rows (full scan is O(trees·N·d)).
        let mut hits = 0u64;
        let mut nll_sum = 0.0f64;
        let mut scored = 0u64;
        let score_stride = (xs.len() / 4096).max(1);
        for (i, (x, &y)) in xs.iter().zip(ys.iter()).enumerate() {
            if i % score_stride != 0 {
                continue;
            }
            let p = self.forest.predict_proba(x);
            if argmax_f32(&p) == y {
                hits += 1;
            }
            nll_sum += -p[y].max(1e-12).ln() as f64;
            scored += 1;
        }
        let n = scored.max(1) as f64;
        let mean_nll = (nll_sum / n) as f32;
        LmTrainStats {
            tokens: xs.len(),
            loss: mean_nll,
            accuracy: hits as f32 / scored.max(1) as f32,
            perplexity: mean_nll.exp(),
        }
    }

    pub fn evaluate_bytes(&mut self, data: &[u8]) -> LmTrainStats {
        let mut hits = 0u64;
        let mut nll_sum = 0.0f64;
        let mut tokens = 0u64;
        if data.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }
        self.reset_state();
        for i in 0..data.len().saturating_sub(1) {
            let a = self.vocab.encode(data[i]);
            let b = self.vocab.encode(data[i + 1]);
            let probs = self.observe(a);
            if argmax_f32(&probs) == b {
                hits += 1;
            }
            nll_sum += -probs[b].max(1e-12).ln() as f64;
            tokens += 1;
        }
        let acc = hits as f32 / tokens.max(1) as f32;
        let mean_nll = (nll_sum / tokens.max(1) as f64) as f32;
        LmTrainStats {
            tokens: tokens as usize,
            loss: mean_nll,
            accuracy: acc,
            perplexity: mean_nll.exp(),
        }
    }

    /// Generate `n` bytes after `prompt` with **MCTS + suffix-array dedup**.
    ///
    /// For each character, runs [`MCTS_SIMS`] simulations with PUCT selection.
    /// A lightweight suffix array of the committed buffer down-weights actions
    /// that would complete a long exact repeat. `temperature` controls rollouts.
    pub fn generate(&mut self, prompt: &[u8], n: usize, temperature: f32) -> Vec<u8> {
        self.reset_state();

        let mut out = prompt.to_vec();
        if out.is_empty() {
            out.push(b' ');
        }
        // Prime dynamics on the prompt (no SA yet — buffer is the prompt itself).
        let sa_prompt = LightSuffixArray::build(out.clone());
        for &b in &out {
            let _ = self.observe_search(self.vocab.encode(b), Some(&sa_prompt));
        }
        let mut last = self.vocab.encode(*out.last().unwrap());

        for _ in 0..n {
            let next = self.mcts_select_action(last, temperature, &out);
            out.push(self.vocab.decode(next));
            let sa = LightSuffixArray::build(out.clone());
            let _ = self.observe_search(next, Some(&sa));
            last = next;
        }
        out
    }

}

/// LM dynamics snapshot for MCTS (ensemble + block adjacency buffer + RNG).
#[derive(Clone, Debug)]
struct LmDynamicsSnap {
    ensemble: EnsembleDynamicsSnap,
    prev_id: usize,
    prev2_id: usize,
    v_block: Vec<[f32; LM_ADJ_PROJ]>,
    gram_sum: [f32; LM_ADJ_PROJ * LM_ADJ_PROJ],
    v_block_len: usize,
    v_block_pos: usize,
    rng_lfsr: u32,
}

/// One node in the character-level MCTS tree.
struct MctsNode {
    /// Action (char id) from parent → this node (unused at root).
    action: usize,
    /// Policy prior P(action | parent).
    prior: f32,
    /// Visit count.
    n: f32,
    /// Total backed-up value.
    w: f32,
    /// Expanded children (arena indices).
    children: Vec<usize>,
    /// Remaining (action, prior) pairs to expand.
    unexpanded: Vec<(usize, f32)>,
}

// ---------------------------------------------------------------------------
// Spike-symbol language model (LifNeuron per char; fire ⇒ emit)
// ---------------------------------------------------------------------------

/// Character LM where each vocabulary symbol owns one [`LifNeuron`].
///
/// Context (previous / previous-previous symbols) drives every neuron through
/// learned synaptic weights. When a neuron spikes, its mapped symbol is
/// **emitted**. If several spike, a membrane WTA picks the winner; if none
/// spike within [`SPIKE_LM_MAX_TICKS`], the highest membrane wins (soft emit).
pub struct SpikeSymbolLm {
    pub vocab: CharVocab,
    /// `neurons[id]` ↔ vocabulary symbol `id`.
    pub neurons: Vec<LifNeuron>,
    /// First-order synapses: current injected into next-symbol neuron `j`
    /// after symbol `i` was emitted (`w[i][j]`).
    w: Vec<Vec<f32>>,
    /// Second-order synapses from two symbols back (`w2[prev2][next]`).
    w2: Vec<Vec<f32>>,
    /// Baseline drive per symbol (unigram-like bias current).
    bias: Vec<f32>,
    prev_id: usize,
    prev2_id: usize,
    rng: Rand,
}

impl SpikeSymbolLm {
    pub fn new(vocab: CharVocab, seed: u32) -> Self {
        let v = vocab.len().max(1);
        let mut neurons = Vec::with_capacity(v);
        for i in 0..v {
            let mut n = LifNeuron::new(
                SPIKE_LM_V_REST,
                SPIKE_LM_V_THR,
                SPIKE_LM_V_RESET,
                SPIKE_LM_TAU,
            );
            // Distinct RNG streams; keep search std low so steps stay near trial means.
            n.rng = Rand::new(seed.wrapping_mul(0x9E37).wrapping_add(i as u32).max(1));
            n.trial_v_rest = SPIKE_LM_V_REST;
            n.trial_v_threshold = SPIKE_LM_V_THR;
            n.v_membrane = SPIKE_LM_V_REST;
            neurons.push(n);
        }
        let mut model = Self {
            vocab,
            neurons,
            w: vec![vec![0.0; v]; v],
            w2: vec![vec![0.0; v]; v],
            bias: vec![0.0; v],
            prev_id: 0,
            prev2_id: 0,
            rng: Rand::new(seed.max(1)),
        };
        model.pin_search_noise();
        let space = model.vocab.encode(b' ');
        model.prev_id = space;
        model.prev2_id = space;
        model
    }

    /// Number of symbol neurons (== vocab size).
    pub fn n_symbols(&self) -> usize {
        self.neurons.len()
    }

    /// Force tiny CEM stddevs so [`LifNeuron::step`] takes the deterministic path.
    /// (A spike would otherwise reset stddevs to [`SPIKE_STD_RESET`].)
    fn pin_search_noise(&mut self) {
        for n in &mut self.neurons {
            n.v_rest_dist.set_stddev(SPIKE_LM_STD);
            n.v_threshold_dist.set_stddev(SPIKE_LM_STD);
            n.trial_v_rest = SPIKE_LM_V_REST;
            n.trial_v_threshold = SPIKE_LM_V_THR;
        }
    }

    fn reset_membranes(&mut self) {
        self.pin_search_noise();
        for n in &mut self.neurons {
            n.v_membrane = SPIKE_LM_V_REST;
            n.is_refractory = false;
        }
    }

    pub fn reset_state(&mut self) {
        self.reset_membranes();
        let space = self.vocab.encode(b' ');
        self.prev_id = space;
        self.prev2_id = space;
    }

    /// Synaptic + bias current into each symbol neuron given `(prev2, prev)`.
    fn drives(&self) -> Vec<f32> {
        let v = self.neurons.len();
        let p = self.prev_id.min(v.saturating_sub(1));
        let p2 = self.prev2_id.min(v.saturating_sub(1));
        let mut d = vec![0.0f32; v];
        for j in 0..v {
            d[j] = self.bias[j]
                + self.w[p][j]
                + SPIKE_LM_W2_SCALE * self.w2[p2][j];
        }
        d
    }

    /// One LIF update on every symbol neuron under the given drives.
    fn step_all(&mut self, drives: &[f32]) -> Vec<usize> {
        self.pin_search_noise();
        let v = self.neurons.len();
        let mut spiked = Vec::new();
        for j in 0..v {
            // Score target = drive so CEM episode cost is well-defined but unused.
            if self.neurons[j].step(drives[j], SPIKE_LM_DT) {
                spiked.push(j);
            }
        }
        // Spikes re-open search noise; pin again for the next tick.
        self.pin_search_noise();
        spiked
    }

    /// Softmax-sample an id from scores (membranes or drives).
    fn sample_from_scores(&mut self, scores: &[f32], temperature: f32) -> usize {
        let t = temperature.max(1e-3);
        let mut max_s = f32::NEG_INFINITY;
        for &s in scores {
            if s > max_s {
                max_s = s;
            }
        }
        let mut sum = 0.0f32;
        let mut weights = vec![0.0f32; scores.len()];
        for (i, &s) in scores.iter().enumerate() {
            let w = ((s - max_s) / t).exp();
            weights[i] = w;
            sum += w;
        }
        if sum <= 0.0 || !sum.is_finite() {
            return scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
        let r = self.rng.u() * sum;
        let mut acc = 0.0f32;
        for (i, &w) in weights.iter().enumerate() {
            acc += w;
            if r <= acc {
                return i;
            }
        }
        scores.len().saturating_sub(1)
    }

    /// Commit an emitted symbol: update context, reset non-winners, mark winner
    /// as just-spiked (refractory / reset).
    fn commit_emit(&mut self, emitted: usize) {
        let v = self.neurons.len();
        let e = emitted.min(v.saturating_sub(1));
        for j in 0..v {
            if j == e {
                self.neurons[j].v_membrane = self.neurons[j].v_reset;
                self.neurons[j].is_refractory = true;
            } else {
                self.neurons[j].v_membrane = self.neurons[j].trial_v_rest;
                self.neurons[j].is_refractory = false;
            }
        }
        self.prev2_id = self.prev_id;
        self.prev_id = e;
    }

    /// Integrate all symbol neurons under current context until someone spikes
    /// (or timeout). Returns the **emitted** symbol id.
    ///
    /// This is the generative core: **a spike emits that symbol**.
    pub fn step_emit(&mut self, temperature: f32) -> usize {
        let v = self.neurons.len();
        if v == 0 {
            return 0;
        }
        let drives = self.drives();
        // Fresh membranes so only the context drive decides who fires.
        self.reset_membranes();

        for _tick in 0..SPIKE_LM_MAX_TICKS {
            let spiked = self.step_all(&drives);
            if spiked.is_empty() {
                continue;
            }
            // Prefer the spiker with the strongest synaptic drive (stable WTA).
            let emitted = if temperature <= 1e-3 {
                spiked
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        drives[a]
                            .partial_cmp(&drives[b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            } else {
                let mut scores = vec![f32::NEG_INFINITY; v];
                for &j in &spiked {
                    scores[j] = drives[j];
                }
                self.sample_from_scores(&scores, temperature)
            };
            self.commit_emit(emitted);
            return emitted;
        }

        // No spike within the window: emit the strongest-driven symbol neuron.
        let emitted = if temperature <= 1e-3 {
            drives
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            self.sample_from_scores(&drives, temperature)
        };
        self.commit_emit(emitted);
        emitted
    }

    /// Teacher-force observe a true symbol (no weight update): as if that neuron
    /// just fired and emitted.
    pub fn observe(&mut self, char_id: usize) {
        self.commit_emit(char_id);
    }

    /// Map log-probabilities into a drive interval
    /// [`SPIKE_LM_FLOOR_DRIVE`, `SPIKE_LM_PEAK_DRIVE`] with a sharp softmax peak.
    fn logp_to_drive(logp: &[f32]) -> Vec<f32> {
        let mut max_lp = f32::NEG_INFINITY;
        for &lp in logp {
            if lp > max_lp {
                max_lp = lp;
            }
        }
        let t = SPIKE_LM_DRIVE_TEMP.max(1e-3);
        let mut out = vec![0.0f32; logp.len()];
        let mut z = 0.0f32;
        for &lp in logp {
            z += ((lp - max_lp) / t).exp();
        }
        let z = z.max(1e-12);
        let span = SPIKE_LM_PEAK_DRIVE - SPIKE_LM_FLOOR_DRIVE;
        // Softmax mass → drive; best next is near peak, others suppressed.
        for (j, &lp) in logp.iter().enumerate() {
            let p = ((lp - max_lp) / t).exp() / z;
            out[j] = SPIKE_LM_FLOOR_DRIVE + span * p;
        }
        // Ensure the mode is strictly supra-threshold even for flat distributions.
        if let Some((best, _)) = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            if out[best] < SPIKE_LM_V_THR + 0.15 {
                out[best] = SPIKE_LM_PEAK_DRIVE;
            }
        }
        out
    }

    /// Seed `w`, `w2`, `bias` from Laplace-smoothed n-grams so high-probability
    /// next symbols receive supra-threshold drive.
    fn fit_weights_from_counts(&mut self, data: &[u8]) {
        let v = self.neurons.len();
        if v == 0 || data.len() < 2 {
            return;
        }
        let mut uni = vec![1.0f32; v];
        let mut bi = vec![vec![1.0f32; v]; v];
        let mut tri = vec![vec![vec![1.0f32; v]; v]; v];

        let mut ids = Vec::with_capacity(data.len());
        for &b in data {
            ids.push(self.vocab.encode(b).min(v - 1));
        }
        for &id in &ids {
            uni[id] += 1.0;
        }
        for i in 0..ids.len().saturating_sub(1) {
            bi[ids[i]][ids[i + 1]] += 1.0;
        }
        for i in 0..ids.len().saturating_sub(2) {
            tri[ids[i]][ids[i + 1]][ids[i + 2]] += 1.0;
        }

        let uni_tot: f32 = uni.iter().sum::<f32>().max(1.0);
        let mut uni_lp = vec![0.0f32; v];
        for j in 0..v {
            uni_lp[j] = (uni[j] / uni_tot).ln();
        }
        self.bias = Self::logp_to_drive(&uni_lp);
        // Bias is a weak prior — scale down so transitions dominate.
        for b in &mut self.bias {
            *b *= 0.12;
        }

        for i in 0..v {
            let row_sum: f32 = bi[i].iter().sum::<f32>().max(1.0);
            let mut lp = vec![0.0f32; v];
            for j in 0..v {
                lp[j] = (bi[i][j] / row_sum).ln();
            }
            self.w[i] = Self::logp_to_drive(&lp);
        }

        // Collapse trigram counts onto (prev2 → next) for a compact second-order map.
        for p2 in 0..v {
            let mut counts = vec![1.0f32; v];
            for mid in 0..v {
                for j in 0..v {
                    counts[j] += tri[p2][mid][j];
                }
            }
            let tot: f32 = counts.iter().sum::<f32>().max(1.0);
            let mut lp = vec![0.0f32; v];
            for j in 0..v {
                lp[j] = (counts[j] / tot).ln();
            }
            self.w2[p2] = Self::logp_to_drive(&lp);
        }
    }

    /// One supervised Hebbian step: drive network with teacher boost on `true_next`,
    /// strengthen / weaken synapses based on who spiked.
    fn hebb_step(&mut self, true_next: usize) -> bool {
        let v = self.neurons.len();
        if v == 0 {
            return false;
        }
        let target = true_next.min(v - 1);
        let mut drives = self.drives();
        drives[target] += SPIKE_LM_TEACHER_BOOST;

        self.reset_membranes();
        let mut spiked = Vec::new();
        for _tick in 0..SPIKE_LM_MAX_TICKS {
            spiked = self.step_all(&drives);
            if !spiked.is_empty() {
                break;
            }
        }

        let correct = spiked.contains(&target);
        let prev = self.prev_id.min(v - 1);
        let prev2 = self.prev2_id.min(v - 1);
        let lr = SPIKE_LM_LR;

        // Reinforce true transition; suppress false spikes (clamped).
        self.w[prev][target] = (self.w[prev][target] + lr).clamp(0.0, SPIKE_LM_PEAK_DRIVE);
        self.w2[prev2][target] =
            (self.w2[prev2][target] + lr * 0.35).clamp(0.0, SPIKE_LM_PEAK_DRIVE);
        for &j in &spiked {
            if j != target {
                self.w[prev][j] =
                    (self.w[prev][j] - lr * 0.5).clamp(0.0, SPIKE_LM_PEAK_DRIVE);
            }
        }
        if !correct {
            self.w[prev][target] =
                (self.w[prev][target] + lr).clamp(0.0, SPIKE_LM_PEAK_DRIVE);
        }

        self.commit_emit(target);
        correct
    }

    /// Predict next id without mutating weights: pure spike emission (greedy).
    fn predict_next(&mut self) -> usize {
        // Snapshot dynamical state so evaluation does not desync teacher force.
        let snap_prev = self.prev_id;
        let snap_prev2 = self.prev2_id;
        let membranes: Vec<(f32, bool)> = self
            .neurons
            .iter()
            .map(|n| (n.v_membrane, n.is_refractory))
            .collect();
        let rng_lfsr = self.rng.lfsr;
        let neuron_rng: Vec<u32> = self.neurons.iter().map(|n| n.rng.lfsr).collect();

        let pred = self.step_emit(0.0);

        // Restore so the caller can teacher-force the true symbol next.
        self.prev_id = snap_prev;
        self.prev2_id = snap_prev2;
        self.rng.lfsr = rng_lfsr;
        for (n, &(v, r)) in self.neurons.iter_mut().zip(membranes.iter()) {
            n.v_membrane = v;
            n.is_refractory = r;
        }
        for (n, &lfsr) in self.neurons.iter_mut().zip(neuron_rng.iter()) {
            n.rng.lfsr = lfsr;
        }
        pred
    }

    /// Fit n-gram synaptic weights, optional Hebbian refinement, return train stats.
    pub fn train_bytes(&mut self, data: &[u8], hebb_epochs: usize) -> LmTrainStats {
        if data.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }
        self.fit_weights_from_counts(data);

        let v = self.neurons.len().max(1);
        let n_pairs = data.len() - 1;
        let hebb_n = SPIKE_LM_HEBB_TOKENS.min(n_pairs);
        for _ in 0..hebb_epochs.max(0) {
            self.reset_state();
            self.observe(self.vocab.encode(data[0]));
            for i in 0..hebb_n {
                let next = self.vocab.encode(data[i + 1]);
                let _ = self.hebb_step(next);
            }
        }

        // Accuracy / NLL under frozen weights (spike emission vs true next).
        self.reset_state();
        self.observe(self.vocab.encode(data[0]));
        let mut correct = 0u32;
        let mut tokens = 0u32;
        let mut nll = 0.0f32;
        let stride = SPIKE_LM_ACC_STRIDE.max(1);
        let mut i = 0;
        while i < n_pairs {
            let true_next = self.vocab.encode(data[i + 1]);
            if i % stride == 0 {
                let drives = self.drives();
                // Softmax NLL from drives (surrogate likelihood of the LIF policy).
                let mut max_d = f32::NEG_INFINITY;
                for &d in &drives {
                    if d > max_d {
                        max_d = d;
                    }
                }
                let mut z = 0.0f32;
                for &d in &drives {
                    z += (d - max_d).exp();
                }
                let log_p = drives[true_next.min(v - 1)] - max_d - z.max(1e-12).ln();
                nll -= log_p;
                let pred = self.predict_next();
                if pred == true_next {
                    correct += 1;
                }
                tokens += 1;
            }
            self.observe(true_next);
            i += 1;
        }
        let tokens = tokens.max(1);
        let mean_nll = nll / tokens as f32;
        LmTrainStats {
            tokens: tokens as usize,
            loss: mean_nll,
            accuracy: correct as f32 / tokens as f32,
            perplexity: mean_nll.exp(),
        }
    }

    /// Held-out accuracy / perplexity with teacher-forced context.
    pub fn evaluate_bytes(&mut self, data: &[u8]) -> LmTrainStats {
        if data.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }
        let v = self.neurons.len().max(1);
        self.reset_state();
        self.observe(self.vocab.encode(data[0]));
        let mut correct = 0u32;
        let mut tokens = 0u32;
        let mut nll = 0.0f32;
        for i in 0..data.len() - 1 {
            let true_next = self.vocab.encode(data[i + 1]);
            let drives = self.drives();
            let mut max_d = f32::NEG_INFINITY;
            for &d in &drives {
                if d > max_d {
                    max_d = d;
                }
            }
            let mut z = 0.0f32;
            for &d in &drives {
                z += (d - max_d).exp();
            }
            let log_p = drives[true_next.min(v - 1)] - max_d - z.max(1e-12).ln();
            nll -= log_p;
            let pred = self.predict_next();
            if pred == true_next {
                correct += 1;
            }
            tokens += 1;
            self.observe(true_next);
        }
        let tokens = tokens.max(1);
        let mean_nll = nll / tokens as f32;
        LmTrainStats {
            tokens: tokens as usize,
            loss: mean_nll,
            accuracy: correct as f32 / tokens as f32,
            perplexity: mean_nll.exp(),
        }
    }

    /// Generate `n` bytes after `prompt` by repeated spike emission.
    pub fn generate(&mut self, prompt: &[u8], n: usize, temperature: f32) -> Vec<u8> {
        self.reset_state();
        let mut out = prompt.to_vec();
        if out.is_empty() {
            out.push(b' ');
        }
        for &b in &out {
            self.observe(self.vocab.encode(b));
        }
        for _ in 0..n {
            let id = self.step_emit(temperature);
            out.push(self.vocab.decode(id));
        }
        out
    }
}

/// Train + evaluate + sample from `100.txt.utf-8` (or a provided path).
fn run_language_model(path: &str) -> Result<(), String> {
    println!();
    println!("=== LifEnsemble block-adjacency LM (RF readout) ===");
    println!("corpus: {path}");

    let corpus = load_corpus(path)?;
    let train_end = LM_TRAIN_CHARS.min(corpus.len().saturating_sub(LM_EVAL_CHARS + 2));
    let eval_end = (train_end + LM_EVAL_CHARS).min(corpus.len());
    if train_end < 1024 {
        return Err(format!(
            "corpus too short for train/eval split ({} bytes)",
            corpus.len()
        ));
    }
    let train = &corpus[..train_end];
    let eval = &corpus[train_end..eval_end];

    let vocab = CharVocab::from_bytes(train);
    println!(
        "bytes: corpus={} train={} eval={} vocab={} ensemble={} embed={} block={} trees={} rf_samples={}",
        corpus.len(),
        train.len(),
        eval.len(),
        vocab.len(),
        ENSEMBLE_N,
        LM_EMBED_DIMS,
        LM_BLOCK_SIZE,
        RF_N_TREES,
        RF_TRAIN_SAMPLES
    );

    let mut model = LifLanguageModel::new(vocab, 0xC0FFEE);
    // Block-wise LifEnsemble adjacency embeddings → random forest.
    let train_stats = model.train_bytes(train, 1);
    println!(
        "train (block-adj+RF): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        train_stats.tokens, train_stats.accuracy, train_stats.loss, train_stats.perplexity
    );

    let eval_stats = model.evaluate_bytes(eval);
    println!(
        "eval (block-adj+RF): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        eval_stats.tokens, eval_stats.accuracy, eval_stats.loss, eval_stats.perplexity
    );

    let prompt = b"To be, or not to be";
    println!();
    println!(
        "MCTS generate: sims={MCTS_SIMS} top_k={MCTS_TOP_K} rollout={MCTS_ROLLOUT} c_puct={MCTS_C_PUCT}"
    );
    let sample_mcts = model.generate(prompt, LM_SAMPLE_LEN, 0.7);
    println!("sample MCTS (rollout_temp=0.7, prompt+{LM_SAMPLE_LEN} bytes):");
    println!("----");
    println!("{}", String::from_utf8_lossy(&sample_mcts));
    println!("----");
    println!(
        "{{lm: {{\"train_acc\": {:.6}, \"eval_acc\": {:.6}, \"eval_ppl\": {:.4}, \"vocab\": {}}}}}",
        train_stats.accuracy,
        eval_stats.accuracy,
        eval_stats.perplexity,
        model.vocab.len()
    );
    Ok(())
}

/// Spike-emission LM: one [`LifNeuron`] per symbol; a spike emits that symbol.
fn run_spike_symbol_lm(path: &str) -> Result<(), String> {
    println!();
    println!("=== Spike-symbol LM (LifNeuron per char; fire ⇒ emit) ===");
    println!("corpus: {path}");

    let corpus = load_corpus(path)?;
    let train_end = LM_TRAIN_CHARS.min(corpus.len().saturating_sub(LM_EVAL_CHARS + 2));
    let eval_end = (train_end + LM_EVAL_CHARS).min(corpus.len());
    if train_end < 1024 {
        return Err(format!(
            "corpus too short for train/eval split ({} bytes)",
            corpus.len()
        ));
    }
    let train = &corpus[..train_end];
    let eval = &corpus[train_end..eval_end];

    let vocab = CharVocab::from_bytes(train);
    println!(
        "bytes: corpus={} train={} eval={} vocab={} neurons={} hebb_cap={} max_ticks={}",
        corpus.len(),
        train.len(),
        eval.len(),
        vocab.len(),
        vocab.len(),
        SPIKE_LM_HEBB_TOKENS,
        SPIKE_LM_MAX_TICKS
    );

    let mut model = SpikeSymbolLm::new(vocab, 0x51A6E);
    // Seed synapses from n-grams, then a light Hebbian pass on a token cap.
    let train_stats = model.train_bytes(train, 1);
    println!(
        "train (n-gram synapses + Hebb≤{SPIKE_LM_HEBB_TOKENS}): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        train_stats.tokens, train_stats.accuracy, train_stats.loss, train_stats.perplexity
    );

    let eval_stats = model.evaluate_bytes(eval);
    println!(
        "eval (spike emit): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        eval_stats.tokens, eval_stats.accuracy, eval_stats.loss, eval_stats.perplexity
    );

    let prompt = b"To be, or not to be";
    println!();
    println!("spike-generate: max_ticks={SPIKE_LM_MAX_TICKS} temp=0.6");
    let sample = model.generate(prompt, LM_SAMPLE_LEN, 0.6);
    println!("sample spike-emit (temp=0.6, prompt+{LM_SAMPLE_LEN} bytes):");
    println!("----");
    println!("{}", String::from_utf8_lossy(&sample));
    println!("----");
    println!(
        "{{spike_lm: {{\"train_acc\": {:.6}, \"eval_acc\": {:.6}, \"eval_ppl\": {:.4}, \"neurons\": {}}}}}",
        train_stats.accuracy,
        eval_stats.accuracy,
        eval_stats.perplexity,
        model.n_symbols()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_chronological() {
        let mut rb = RingBuffer::<i32, 4>::new();
        assert!(rb.is_empty());

        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.get(0), Some(&10));
        assert_eq!(rb.get(2), Some(&30));
        assert_eq!(rb.latest(), Some(&30));
        assert_eq!(rb.get_newest(0), Some(&30));
        assert_eq!(rb.get_newest(1), Some(&20));
        assert_eq!(rb.get_newest(2), Some(&10));

        rb.push(40);
        rb.push(50);
        assert!(rb.is_full());
        assert_eq!(rb.len(), 4);
        assert_eq!(rb.get(0), Some(&20));
        assert_eq!(rb.get(3), Some(&50));

        assert_eq!(rb.pop(), Some(20));
        assert_eq!(rb.get(0), Some(&30));
        assert_eq!(rb.latest(), Some(&50));
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut rb = RingBuffer::<f32, 4>::new();
        rb.push(1.0);
        rb.push(2.0);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        rb.push(3.0);
        assert_eq!(rb.latest(), Some(&3.0));
    }

    #[test]
    fn test_elite_rank_weights_prefer_best() {
        let mut w = [0.0; 4];
        elite_rank_weights(4, &mut w);
        assert!(w[0] > w[1]);
        assert!(w[1] > w[2]);
        assert!(w[2] > w[3]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_update_moves_toward_elites() {
        let mut p = GaussianParam::new(0.0, 0.5);
        let elites = [2.0f32, 2.2, 1.8];
        let mut weights = [0.0; 3];
        elite_rank_weights(3, &mut weights);
        p.update_from_elites(&elites, &weights);
        assert!(p.mean > 0.5); // pulled toward ~2
        assert!(p.stddev >= STD_EPS);
        assert!(p.stddev <= STD_MAX);
    }

    #[test]
    fn test_spike_resets_search_stddevs() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist.set_stddev(0.01);
        neuron.v_threshold_dist.set_stddev(0.01);
        neuron.v_membrane = 10.0;
        neuron.pending_antithetic = None;
        neuron.episode_step = 0;

        assert!(neuron.step(0.0, 1.0), "expected spike from high membrane");
        assert!(
            (neuron.v_rest_dist.stddev - SPIKE_STD_RESET).abs() < 1e-5,
            "v_rest stddev should reset on spike, got {}",
            neuron.v_rest_dist.stddev
        );
        assert!(
            (neuron.v_threshold_dist.stddev - SPIKE_STD_RESET).abs() < 1e-5,
            "v_threshold stddev should reset on spike, got {}",
            neuron.v_threshold_dist.stddev
        );
    }

    #[test]
    fn test_refractory_still_scores_episode() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_membrane = 10.0;
        neuron.pending_antithetic = None;

        let spiked = neuron.step(0.0, 1.0);
        assert!(spiked);
        assert_eq!(neuron.input.len(), 1);
        assert_eq!(neuron.output.len(), 1);
        assert_eq!(neuron.episode_step, 1);

        // Refractory step still records I/O and advances the episode.
        let spiked = neuron.step(1.0, 1.0);
        assert!(!spiked);
        assert_eq!(neuron.input.len(), 2);
        assert_eq!(neuron.output.len(), 2);
        assert_eq!(neuron.v_membrane, neuron.v_reset);
    }

    #[test]
    fn test_firing_is_probabilistic_near_threshold() {
        // V near thr mean with non-zero σ ⇒ Monte Carlo majority is stochastic.
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 1.0e9); // huge τ → V ≈ fixed
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.5);
        neuron.v_membrane = 1.0;
        neuron.pending_antithetic = None;
        neuron.episode_step = 0;

        let mut spikes = 0u32;
        let trials = 200u32;
        for _ in 0..trials {
            neuron.v_membrane = 1.0;
            neuron.is_refractory = false;
            if neuron.step(0.0, 1e-6) {
                spikes += 1;
            }
            neuron.episode_step = 0;
            neuron.episode_error_sum = 0.0;
            neuron.episode_spike_count = 0;
        }
        // Per-sample fire prob ≈ ½; majority of MC_SAMPLES still mixes.
        assert!(
            spikes > 20 && spikes < 180,
            "expected mixed MC majority spikes near threshold, got {spikes}/{trials}"
        );
    }

    #[test]
    fn test_firing_certain_when_far_above_threshold() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.3);
        neuron.pending_antithetic = None;

        for _ in 0..50 {
            neuron.v_membrane = 5.0;
            neuron.is_refractory = false;
            neuron.episode_step = 0;
            assert!(
                neuron.step(0.0, 1.0),
                "V far above thr should almost surely spike under MC majority"
            );
        }
    }

    #[test]
    fn test_monte_carlo_majority_fire_decision() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 0.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_membrane = 1.0;
        neuron.pending_antithetic = None;
        neuron.episode_step = 0;
        // Tiny σ, V well above thr mean ⇒ every MC micro-step fires ⇒ majority fire.
        let (v_hi, fire_hi) = neuron.monte_carlo_step(0.0, 1.0);
        assert!(fire_hi, "expected majority fire when V >> thr");
        assert!(v_hi.is_finite());

        neuron.v_membrane = -2.0;
        let (v_lo, fire_lo) = neuron.monte_carlo_step(0.0, 1.0);
        assert!(!fire_lo, "expected no fire when V << thr");
        assert!(v_lo.is_finite());
    }

    #[test]
    fn test_cem_prefers_low_error_params() {
        let mut neuron = LifNeuron::new(5.0, 5.0, 0.0, 10.0);
        let rest_before = neuron.v_rest();
        let thr_before = neuron.v_threshold();
        // Good episodes: rest near 0, threshold moderate.
        for _ in 0..4 {
            neuron.episode_v_rest.push(0.0);
            neuron.episode_v_threshold.push(2.0);
            neuron.episode_fitness.push(0.1);
        }
        // Bad episodes.
        for _ in 0..4 {
            neuron.episode_v_rest.push(8.0);
            neuron.episode_v_threshold.push(8.0);
            neuron.episode_fitness.push(5.0);
        }
        neuron.cem_update();
        // Soft update moves partway toward elite means (~0 rest, ~2 thr).
        assert!(
            neuron.v_rest() < rest_before,
            "mean rest should decrease toward 0, {} -> {}",
            rest_before,
            neuron.v_rest()
        );
        assert!(
            neuron.v_threshold() < thr_before,
            "mean threshold should decrease toward 2, {} -> {}",
            thr_before,
            neuron.v_threshold()
        );
        assert_eq!(neuron.generation, 1);
        assert!(neuron.episode_fitness.is_empty());
    }

    #[test]
    fn test_learning_reduces_tracking_error() {
        let mut neuron = LifNeuron::new(3.0, 4.0, 0.0, 10.0);
        let rest_before = neuron.v_rest().abs();
        let dt = 10.0;
        let mut injected = 1.0f32;
        let total = 512usize;

        for _ in 0..total {
            let _ = neuron.step(injected, dt);
            injected = if injected == 1.0 { 0.0 } else { 1.0 };
        }

        assert!(
            neuron.generation >= 2,
            "expected multiple CEM generations, got {}",
            neuron.generation
        );
        // Steady-state V ≈ v_rest + I tracks I best when v_rest → 0.
        assert!(
            neuron.v_rest().abs() < rest_before,
            "v_rest should move toward 0, before={rest_before} after={}",
            neuron.v_rest()
        );
        assert!(
            neuron.last_gen_fitness.is_finite() && neuron.last_gen_fitness >= 0.0,
            "last_gen_fitness should be a valid cost"
        );
    }

    #[test]
    fn test_lfsr() {
        let mut lfsr = Rand::new(1);
        let mut count: u64 = 1;
        loop {
            let s = lfsr.u32();
            if s == 1 {
                break;
            }
            count += 1;
        }
        assert_eq!(count, u32::MAX as u64);
    }

    #[test]
    fn test_g() {
        const N: usize = 8 * 1024;
        let mut lfsr = Rand::new(1);
        let mut za: [f32; N] = [0.0; N];
        let mut zb: [f32; N] = [0.0; N];
        for step in 0..N {
            let (z0, z1) = lfsr.g();
            za[step] = z0;
            zb[step] = z1;
        }
        for series in [&za[..], &zb[..]] {
            let mut avg = 0.0;
            for &value in series {
                avg += value;
            }
            avg /= N as f32;
            let mut stddev = 0.0;
            for &value in series {
                let diff = value - avg;
                stddev += diff * diff;
            }
            stddev /= N as f32;
            stddev = stddev.sqrt();
            assert_eq!((10.0 * avg).round() / 10.0, 0.0);
            assert_eq!(stddev.round(), 1.0);
        }
    }

    #[test]
    fn test_signal_generators_shapes() {
        let sw = gen_square_wave(8);
        assert_eq!(sw.len(), 8);
        assert_eq!(sw[0][0], 1.0);
        assert_eq!(sw[1][0], 0.0);

        let oh = gen_one_hot_cycle(8, 4);
        assert_eq!(oh[0][0], 1.0);
        assert_eq!(oh[1][1], 1.0);
        assert_eq!(oh[4][0], 1.0);
        assert!((oh[0].iter().sum::<f32>() - 1.0).abs() < 1e-5);

        let uc = gen_unit_circle(4, core::f32::consts::FRAC_PI_2);
        assert!((uc[0][0] - 1.0).abs() < 1e-5);
        assert!(uc[0][1].abs() < 1e-5);
        assert!(uc[1][0].abs() < 1e-5);
        assert!((uc[1][1] - 1.0).abs() < 1e-5);

        let ms = gen_multi_sine(16, 0.1);
        assert!(ms.iter().all(|v| v[0].is_finite()));

        let lz = gen_lorenz(64, 0.01, 20);
        assert_eq!(lz.len(), 64);
        assert!(lz.iter().all(|v| v[0].is_finite() && v[1].is_finite() && v[2].is_finite()));
        // Consecutive emitted samples should differ (non-trivial next-step).
        let step_jump = (lz[0][0] - lz[1][0]).abs()
            + (lz[0][1] - lz[1][1]).abs()
            + (lz[0][2] - lz[1][2]).abs();
        assert!(
            step_jump > 0.01,
            "Lorenz stride should produce a real jump, got {step_jump}"
        );
    }

    #[test]
    fn test_benchmark_suite_errors_are_finite() {
        // Short horizon keeps the test fast while still exercising CEM + all tasks.
        let results = run_benchmark_suite(64, BENCH_DT);
        assert_eq!(results.len(), 5);
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert_eq!(
            names,
            [
                "square_wave",
                "one_hot_cycle",
                "unit_circle",
                "multi_sine",
                "lorenz"
            ]
        );
        for r in &results {
            assert!(r.mae.is_finite() && r.mae >= 0.0, "{} mae", r.name);
            assert!(r.rmse.is_finite() && r.rmse >= 0.0, "{} rmse", r.name);
            assert!(r.mean_l2.is_finite() && r.mean_l2 >= 0.0, "{} l2", r.name);
            assert!(r.early_mae.is_finite(), "{} early", r.name);
            assert!(r.late_mae.is_finite(), "{} late", r.name);
            assert!(r.dims >= 1);
            assert!(r.steps > 0);
        }
        // Same-step square-wave tracking should beat next-step one-hot (harder).
        let sq = results.iter().find(|r| r.name == "square_wave").unwrap();
        let oh = results.iter().find(|r| r.name == "one_hot_cycle").unwrap();
        assert_eq!(sq.mode, ScoreMode::Track);
        assert_eq!(oh.mode, ScoreMode::PredictNext);
        assert_eq!(oh.dims, 4);
        // Ensemble + multi-pass should keep absolute errors in a useful range.
        assert!(sq.mae < 0.25, "square_wave MAE too high: {}", sq.mae);
        assert!(oh.mae < 0.45, "one_hot_cycle MAE too high: {}", oh.mae);
    }

    #[test]
    fn test_ensemble_learns_unit_circle_next_step() {
        let series = gen_unit_circle(128, 0.15);
        let r = evaluate_series("unit_circle", 2, &series, BENCH_DT, ScoreMode::PredictNext);
        assert!(r.mae.is_finite());
        assert!(
            r.late_mae < 0.2,
            "expected low late error on unit circle, late_mae={}",
            r.late_mae
        );
    }

    #[test]
    fn test_char_vocab_roundtrip() {
        let data = b"Hello, Shakespeare!\nTo be, or not to be.";
        let v = CharVocab::from_bytes(data);
        assert!(v.len() >= 10);
        for &b in data {
            let id = v.encode(b);
            assert_eq!(v.decode(id), b);
        }
        // Unknown byte maps to a valid id.
        let unk = v.encode(0x01);
        assert!(unk < v.len());
    }

    #[test]
    fn test_lif_language_model_learns_tiny_corpus() {
        // Repeating phrase should be learnable as a next-byte task with RF readout.
        let text = b"to be or not to be or not to be or not to be or not to be or not ";
        let vocab = CharVocab::from_bytes(text);
        let mut model = LifLanguageModel::new(vocab, 42);
        let stats = model.train_bytes(text, 1);
        assert!(stats.tokens > 0);
        assert!(
            stats.accuracy > 0.25,
            "expected RF readout to learn the tiny loop, acc={}",
            stats.accuracy
        );
        assert!(stats.perplexity.is_finite() && stats.perplexity >= 1.0);
        assert!(model.forest.is_trained());

        // Short MCTS sample (few sims still exercise the path).
        let sample = model.generate(b"to be", 8, 0.5);
        assert!(sample.len() > 5);
        assert!(std::str::from_utf8(&sample).is_ok());
    }

    #[test]
    fn test_spike_symbol_lm_maps_neuron_to_emit() {
        // Tight loop: n-gram drives + spike emission should recover the cycle.
        let text = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcab";
        let vocab = CharVocab::from_bytes(text);
        let mut model = SpikeSymbolLm::new(vocab, 99);
        assert_eq!(model.n_symbols(), model.vocab.len());

        let stats = model.train_bytes(text, 1);
        assert!(stats.tokens > 0);
        assert!(
            stats.accuracy > 0.5,
            "spike-symbol LM should learn abc loop, acc={}",
            stats.accuracy
        );
        assert!(stats.perplexity.is_finite() && stats.perplexity >= 1.0);

        // After observing 'a', the 'b' neuron should be the one that fires.
        model.reset_state();
        model.observe(model.vocab.encode(b'a'));
        let emitted = model.step_emit(0.0);
        assert_eq!(
            model.vocab.decode(emitted),
            b'b',
            "after 'a', expected 'b' neuron to spike/emit"
        );

        let sample = model.generate(b"ab", 12, 0.0);
        assert!(sample.len() >= 14);
        assert!(std::str::from_utf8(&sample).is_ok());
        // Greedy free-run from "ab" should continue the abc cycle.
        let tail = &sample[2..];
        let ok_cycle = tail.windows(3).filter(|w| *w == b"abc" || *w == b"bca" || *w == b"cab").count()
            >= 2
            || tail.iter().filter(|&&c| c == b'a' || c == b'b' || c == b'c').count()
                >= tail.len().saturating_sub(2);
        assert!(
            ok_cycle,
            "expected abc-like free-run, got {:?}",
            String::from_utf8_lossy(&sample)
        );
    }

    #[test]
    fn test_spike_symbol_lm_shakespeare_prefix() {
        let corpus = load_corpus(LM_CORPUS_PATH).expect("100.txt.utf-8 should exist");
        let train = &corpus[..50_000.min(corpus.len())];
        let eval = &corpus[50_000..55_000.min(corpus.len())];
        let vocab = CharVocab::from_bytes(train);
        let mut model = SpikeSymbolLm::new(vocab, 7);
        // n-gram synaptic init + one short Hebbian pass (capped tokens).
        let train_stats = model.train_bytes(train, 1);
        assert!(train_stats.accuracy > 0.15, "train acc={}", train_stats.accuracy);
        let eval_stats = model.evaluate_bytes(eval);
        assert!(
            eval_stats.accuracy > 0.12,
            "expected bigram-driven spike LM above chance, eval_acc={}",
            eval_stats.accuracy
        );
        let sample = model.generate(b"To be", 40, 0.4);
        assert!(sample.len() > 40);
        // Should not collapse to a single repeated character for 40 steps.
        let uniq: std::collections::BTreeSet<u8> = sample.iter().copied().collect();
        assert!(
            uniq.len() >= 5,
            "degenerate sample: {:?}",
            String::from_utf8_lossy(&sample)
        );
    }

    #[test]
    fn test_random_forest_basic_fit_predict() {
        // Two Gaussian blobs → RF should separate them.
        let mut rng = Rand::new(7);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..80 {
            let (z0, z1) = rng.g();
            xs.push(vec![z0 - 2.0, z1]);
            ys.push(0);
            let (z0, z1) = rng.g();
            xs.push(vec![z0 + 2.0, z1]);
            ys.push(1);
        }
        let mut rf = RandomForest::new();
        rf.fit(&xs, &ys, 2, 16, &mut rng);
        let mut correct = 0;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            if rf.predict(x) == y {
                correct += 1;
            }
        }
        assert!(
            correct as f32 / xs.len() as f32 > 0.85,
            "RF should separate simple blobs, acc={}",
            correct as f32 / xs.len() as f32
        );
    }

    #[test]
    fn test_suffix_array_detects_duplicate_extension() {
        // "hello hell" + 'o' completes "hello", already present at the start.
        let sa = LightSuffixArray::build(b"hello hell".to_vec());
        assert!(sa.contains(b"hello"));
        assert!(sa.contains(b"hell"));
        let rep = sa.duplicate_extension_len(b'o');
        assert!(
            rep >= SA_DEDUP_MIN_LEN,
            "expected long duplicate extension for 'o', got {rep}"
        );
        // Novel continuation should not flag a long repeat.
        let rep_z = sa.duplicate_extension_len(b'z');
        assert_eq!(rep_z, 0, "novel byte should not form a long prior substring");

        // text="hello ", path="hel", c='l' → "... hell" with "hell" already in text.
        let raw = LightSuffixArray::duplicate_extension_len_raw(b"hello ", b"hel", b'l');
        assert!(
            raw >= SA_DEDUP_MIN_LEN,
            "path-aware raw check should see 'hell' repeating, got {raw}"
        );
    }

    #[test]
    fn test_load_shakespeare_corpus_prefix() {
        let corpus = load_corpus(LM_CORPUS_PATH).expect("100.txt.utf-8 should exist");
        assert!(corpus.len() > 100_000);
        assert!(!corpus.contains(&b'\r'));
        let vocab = CharVocab::from_bytes(&corpus[..50_000]);
        assert!(vocab.len() > 40 && vocab.len() < 200);
    }
}

// ---------------------------------------------------------------------------
// Benchmark signals + evaluation harness
// ---------------------------------------------------------------------------

/// Integration / wall-clock dt used when driving the LIF.
const BENCH_DT: f32 = 10.0;
/// Default horizon (prediction tasks need one extra generator step).
const BENCH_STEPS: usize = 256;

/// How the target is formed relative to the injected input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScoreMode {
    /// Same-step tracking: inject x[t], score V against x[t].
    Track,
    /// Next-step prediction: inject x[t], score V against x[t+1].
    PredictNext,
}

#[derive(Clone, Debug)]
struct BenchResult {
    name: &'static str,
    mode: ScoreMode,
    dims: usize,
    steps: usize,
    /// Mean absolute error over all dims × scored steps.
    mae: f32,
    /// Root mean squared error over all dims × scored steps.
    rmse: f32,
    /// Mean Euclidean residual ‖e‖₂ per time step (multi-D sensitive).
    mean_l2: f32,
    /// MAE on the last 25% of steps (post-learning window).
    late_mae: f32,
    /// MAE on the first 25% of steps (pre/early learning).
    early_mae: f32,
}

/// Square wave on {0, 1} — baseline 1-D tracking task.
fn gen_square_wave(steps: usize) -> Vec<[f32; MAX_DIMS]> {
    (0..steps)
        .map(|t| {
            let mut v = [0.0; MAX_DIMS];
            v[0] = if t % 2 == 0 { 1.0 } else { 0.0 };
            v
        })
        .collect()
}

/// One-hot cycle over `k` symbols: e_0 → e_1 → … → e_{k-1} → e_0.
/// Discrete next-symbol prediction when scored with [`ScoreMode::PredictNext`].
fn gen_one_hot_cycle(steps: usize, k: usize) -> Vec<[f32; MAX_DIMS]> {
    assert!(k >= 2 && k <= MAX_DIMS);
    (0..steps)
        .map(|t| {
            let mut v = [0.0; MAX_DIMS];
            v[t % k] = 1.0;
            v
        })
        .collect()
}

/// Unit circle: (cos θ_t, sin θ_t) with constant angular step.
/// Smooth 2-D next-position prediction under [`ScoreMode::PredictNext`].
fn gen_unit_circle(steps: usize, dtheta: f32) -> Vec<[f32; MAX_DIMS]> {
    (0..steps)
        .map(|t| {
            let theta = t as f32 * dtheta;
            let mut v = [0.0; MAX_DIMS];
            v[0] = theta.cos();
            v[1] = theta.sin();
            v
        })
        .collect()
}

/// Superposition of incommensurate sines (1-D temporal structure).
fn gen_multi_sine(steps: usize, time_dt: f32) -> Vec<[f32; MAX_DIMS]> {
    (0..steps)
        .map(|t| {
            let time = t as f32 * time_dt;
            let mut v = [0.0; MAX_DIMS];
            // Amplitudes sum to ~1.75 peak; scale into a neuron-friendly range.
            let raw = time.sin() + 0.5 * (2.3 * time).sin() + 0.25 * (0.7 * time).sin();
            v[0] = raw * 0.5;
            v
        })
        .collect()
}

/// Lorenz attractor (σ=10, ρ=28, β=8/3).
///
/// Integrates with a small stable Euler step `h`, but only **emits** a sample
/// every `stride` micro-steps so consecutive targets are a meaningful jump on
/// the attractor (true multi-step-ahead chaos, not a near-identity map).
fn gen_lorenz(steps: usize, h: f32, stride: usize) -> Vec<[f32; MAX_DIMS]> {
    const SIGMA: f32 = 10.0;
    const RHO: f32 = 28.0;
    const BETA: f32 = 8.0 / 3.0;

    let stride = stride.max(1);
    let mut x = 1.0f32;
    let mut y = 1.0f32;
    let mut z = 1.0f32;
    // Warm up onto the attractor before scoring.
    for _ in 0..(500 * stride) {
        let dx = SIGMA * (y - x);
        let dy = x * (RHO - z) - y;
        let dz = x * y - BETA * z;
        x += h * dx;
        y += h * dy;
        z += h * dz;
    }

    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        // Rough normalization into O(1) so membrane tracking is comparable.
        let mut v = [0.0; MAX_DIMS];
        v[0] = x / 20.0;
        v[1] = y / 20.0;
        v[2] = (z - 25.0) / 25.0;
        out.push(v);

        for _ in 0..stride {
            let dx = SIGMA * (y - x);
            let dy = x * (RHO - z) - y;
            let dz = x * y - BETA * z;
            x += h * dx;
            y += h * dy;
            z += h * dz;
        }
    }
    out
}

/// Drive an [`LifEnsemble`] (reservoir + SGD readout); score predictions vs target.
fn evaluate_series(
    name: &'static str,
    dims: usize,
    series: &[[f32; MAX_DIMS]],
    dt: f32,
    mode: ScoreMode,
) -> BenchResult {
    assert!(dims >= 1 && dims <= MAX_DIMS);
    assert!(series.len() >= 2);

    let scored_steps = match mode {
        ScoreMode::Track => series.len(),
        ScoreMode::PredictNext => series.len() - 1,
    };

    // Stable seed from name so runs are reproducible per benchmark.
    let seed = name.bytes().fold(1u32, |a, b| a.wrapping_mul(16777619) ^ b as u32);
    let mut net = LifEnsemble::new(dims, dims, seed);

    let mut sum_abs = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut sum_l2 = 0.0f32;
    let mut n_elem = 0.0f32;

    let early_end = (scored_steps / 4).max(1);
    let late_start = scored_steps.saturating_sub(early_end);
    let mut early_abs = 0.0f32;
    let mut early_n = 0.0f32;
    let mut late_abs = 0.0f32;
    let mut late_n = 0.0f32;

    // Several online passes; only the final pass contributes to reported error
    // so early CEM / readout transients do not dominate the headline MAE.
    for pass in 0..TRAIN_PASSES {
        let score_pass = pass + 1 == TRAIN_PASSES;
        for t in 0..scored_steps {
            let x = &series[t];
            let target = match mode {
                ScoreMode::Track => x,
                ScoreMode::PredictNext => &series[t + 1],
            };

            let pred = net.step(x, target, dt);
            if !score_pass {
                continue;
            }

            let mut err_sq_vec = 0.0f32;
            for d in 0..dims {
                let e = pred[d] - target[d];
                let ae = e.abs();
                sum_abs += ae;
                sum_sq += e * e;
                err_sq_vec += e * e;
                n_elem += 1.0;

                if t < early_end {
                    early_abs += ae;
                    early_n += 1.0;
                }
                if t >= late_start {
                    late_abs += ae;
                    late_n += 1.0;
                }
            }
            sum_l2 += err_sq_vec.sqrt();
        }
    }

    BenchResult {
        name,
        mode,
        dims,
        steps: scored_steps,
        mae: sum_abs / n_elem.max(1.0),
        rmse: (sum_sq / n_elem.max(1.0)).sqrt(),
        mean_l2: sum_l2 / (scored_steps as f32).max(1.0),
        late_mae: late_abs / late_n.max(1.0),
        early_mae: early_abs / early_n.max(1.0),
    }
}

fn mode_tag(mode: ScoreMode) -> &'static str {
    match mode {
        ScoreMode::Track => "track",
        ScoreMode::PredictNext => "next",
    }
}

fn print_bench_table(results: &[BenchResult]) {
    println!(
        "{:<16} {:>5} {:>4} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "benchmark", "mode", "dim", "steps", "MAE", "RMSE", "mean_L2", "early", "late"
    );
    println!("{}", "-".repeat(78));
    for r in results {
        println!(
            "{:<16} {:>5} {:>4} {:>5} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            r.name,
            mode_tag(r.mode),
            r.dims,
            r.steps,
            r.mae,
            r.rmse,
            r.mean_l2,
            r.early_mae,
            r.late_mae
        );
    }
}

/// Run the full suite used by `main` and the integration test.
///
/// `steps` is the number of scored LIF updates per benchmark. Predict-next
/// tasks generate one extra sample so x[t+1] exists at the last step.
fn run_benchmark_suite(steps: usize, dt: f32) -> Vec<BenchResult> {
    let n_next = steps + 1;

    let square = gen_square_wave(steps);
    let one_hot = gen_one_hot_cycle(n_next, 4);
    let circle = gen_unit_circle(n_next, 0.15);
    let multi_sine = gen_multi_sine(n_next, 0.12);
    // Stable micro-step h=0.01; stride=20 ⇒ Δt=0.2 between samples (chaotic).
    let lorenz = gen_lorenz(steps.max(512) + 1, 0.01, 20);

    vec![
        evaluate_series("square_wave", 1, &square, dt, ScoreMode::Track),
        evaluate_series("one_hot_cycle", 4, &one_hot, dt, ScoreMode::PredictNext),
        evaluate_series("unit_circle", 2, &circle, dt, ScoreMode::PredictNext),
        evaluate_series("multi_sine", 1, &multi_sine, dt, ScoreMode::PredictNext),
        evaluate_series("lorenz", 3, &lorenz, dt, ScoreMode::PredictNext),
    ]
}

fn main() {
    println!(
        "CEM-LIF ensemble: units={ENSEMBLE_N} pop={POP_SIZE} episode={EPISODE_LEN} \
         elite={ELITE_COUNT} readout_lr={READOUT_LR} passes={TRAIN_PASSES} \
         mc_samples={MC_SAMPLES}"
    );
    println!(
        "score modes: square_wave = same-step track; others = next-step prediction \
         ({ENSEMBLE_N} LIF reservoir + NLMS readout with input/delay skips)"
    );
    println!();

    let results = run_benchmark_suite(BENCH_STEPS, BENCH_DT);
    print_bench_table(&results);

    println!();
    println!("error summary (MAE):");
    for r in &results {
        println!(
            "  {:>14}: MAE={:.4}  RMSE={:.4}  late_MAE={:.4}  (early={:.4})",
            r.name, r.mae, r.rmse, r.late_mae, r.early_mae
        );
    }

    // Compact machine-readable line for scripts / CI.
    let total_mae: f32 = results.iter().map(|r| r.mae).sum();
    println!();
    println!(
        "{{scores: {{{}}}}}",
        results
            .iter()
            .map(|r| format!("\"{}\": {:.6}", r.name, r.mae))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("{{total_mae: {:.6}}}", total_mae);

    // Character LM on Shakespeare (eBook #100): ensemble+RF block model, then
    // the spike-emission model (one LifNeuron per symbol).
    if let Err(e) = run_language_model(LM_CORPUS_PATH) {
        eprintln!("language model error: {e}");
        std::process::exit(1);
    }
    if let Err(e) = run_spike_symbol_lm(LM_CORPUS_PATH) {
        eprintln!("spike-symbol language model error: {e}");
        std::process::exit(1);
    }
}
