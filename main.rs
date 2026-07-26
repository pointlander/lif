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

// ---------------------------------------------------------------------------
// Simple leaky integrate-and-fire neuron (deterministic, no CEM)
// ---------------------------------------------------------------------------

/// Minimal LIF cell: exact subthreshold Euler-free step, one-step refractory.
///
/// Dynamics: \(\tau \dot V = -(V - v_{\mathrm{rest}}) + I\). Spike when
/// \(V \ge v_{\mathrm{threshold}}\), then hold at \(v_{\mathrm{reset}}\) for one step.
/// Used directly by [`SpikeWordLm`]; CEM/MC adaptation lives in [`CemLifNeuron`].
#[derive(Clone, Debug)]
pub struct LifNeuron {
    pub v_membrane: f32,
    pub v_rest: f32,
    pub v_threshold: f32,
    pub v_reset: f32,
    pub tau_m: f32,
    pub is_refractory: bool,
}

impl LifNeuron {
    pub fn new(v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32) -> Self {
        Self {
            v_membrane: v_rest,
            v_rest,
            v_threshold,
            v_reset,
            tau_m,
            is_refractory: false,
        }
    }

    /// Place membrane at rest and clear refractory.
    pub fn reset(&mut self) {
        self.v_membrane = self.v_rest;
        self.is_refractory = false;
    }

    /// One deterministic LIF update. Returns `true` if the neuron spiked.
    pub fn step(&mut self, drive: f32, dt: f32) -> bool {
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            return false;
        }
        let decay = (-dt / self.tau_m.max(1e-3)).exp();
        let thr = self.v_threshold.max(self.v_reset + 0.05);
        let v_inf = self.v_rest + drive;
        let v_new = v_inf + (self.v_membrane - v_inf) * decay;
        self.v_membrane = v_new;
        if v_new >= thr {
            self.is_refractory = true;
            true
        } else {
            false
        }
    }

    /// Exact subthreshold step with explicit rest/threshold (used by CEM trials).
    #[inline]
    pub fn integrate(v0: f32, drive: f32, v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32, dt: f32) -> (f32, bool) {
        let decay = (-dt / tau_m.max(1e-3)).exp();
        let thr = v_threshold.max(v_reset + 0.05);
        let v_inf = v_rest + drive;
        let v_new = v_inf + (v0 - v_inf) * decay;
        (v_new, v_new >= thr)
    }
}

// ---------------------------------------------------------------------------
// CEM + Monte Carlo adaptive LIF (ensemble / benchmarks)
// ---------------------------------------------------------------------------

/// Adaptive LIF: wraps a [`LifNeuron`] with CEM search over `(v_rest, v_threshold)`
/// and optional Monte Carlo majority-vote firing.
pub struct CemLifNeuron {
    pub cell: LifNeuron,
    /// Search distribution over resting potential.
    pub v_rest_dist: GaussianParam,
    /// Search distribution over spike threshold.
    pub v_threshold_dist: GaussianParam,
    /// Parameters active for the current episode.
    pub trial_v_rest: f32,
    pub trial_v_threshold: f32,
    pending_antithetic: Option<(f32, f32)>,
    pub input: RingBuffer<f32, POP_SIZE>,
    pub output: RingBuffer<f32, POP_SIZE>,
    pub episode_fitness: RingBuffer<f32, POP_SIZE>,
    pub episode_v_rest: RingBuffer<f32, POP_SIZE>,
    pub episode_v_threshold: RingBuffer<f32, POP_SIZE>,
    episode_error_sum: f32,
    episode_spike_count: u32,
    episode_step: u32,
    pub generation: u64,
    pub last_gen_fitness: f32,
    pub rng: Rand,
}

impl CemLifNeuron {
    pub fn new(v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32) -> Self {
        let mut neuron = Self {
            cell: LifNeuron::new(v_rest, v_threshold, v_reset, tau_m),
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

    pub fn v_membrane(&self) -> f32 {
        self.cell.v_membrane
    }

    pub fn is_refractory(&self) -> bool {
        self.cell.is_refractory
    }

    pub fn v_reset(&self) -> f32 {
        self.cell.v_reset
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

    pub fn resample_trial(&mut self) {
        if let Some((rest, thr)) = self.pending_antithetic.take() {
            self.trial_v_rest = rest;
            self.trial_v_threshold = thr;
            return;
        }
        let (z0, z1) = self.rng.g();
        self.trial_v_rest = self.v_rest_dist.sample(z0);
        self.trial_v_threshold = self
            .v_threshold_dist
            .sample(z1)
            .max(self.cell.v_reset + 0.05);
        self.pending_antithetic = Some((
            self.v_rest_dist.sample(-z0),
            self.v_threshold_dist
                .sample(-z1)
                .max(self.cell.v_reset + 0.05),
        ));
    }

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

    pub fn cem_update(&mut self) {
        let n = self.episode_fitness.len();
        if n == 0 {
            return;
        }
        let mut order = [0usize; POP_SIZE];
        for i in 0..n {
            order[i] = i;
        }
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
        if self.v_threshold_dist.mean < self.cell.v_reset + 0.1 {
            self.v_threshold_dist.mean = self.cell.v_reset + 0.1;
        }
        self.last_gen_fitness = self.episode_fitness.sum() / n as f32;
        self.generation += 1;
        self.episode_fitness.clear();
        self.episode_v_rest.clear();
        self.episode_v_threshold.clear();
        self.pending_antithetic = None;
    }

    fn record_step(&mut self, drive: f32, score_target: f32) {
        let err = score_target - self.cell.v_membrane;
        self.episode_error_sum += err * err;
        self.episode_step += 1;
        self.input.push(drive);
        self.output.push(self.cell.v_membrane);
        if self.episode_step >= EPISODE_LEN as u32 {
            self.finish_episode();
        }
    }

    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
        self.step_with_target(i_input, i_input, dt)
    }

    /// LIF step with CEM trial params; MC majority vote when search noise is high.
    /// On spike, search stddevs reset to [`SPIKE_STD_RESET`].
    pub fn step_with_target(&mut self, drive: f32, score_target: f32, dt: f32) -> bool {
        if self.cell.is_refractory {
            self.cell.is_refractory = false;
            self.cell.v_membrane = self.cell.v_reset;
            self.record_step(drive, score_target);
            return false;
        }

        let (v_mean, spiked) = if self.v_rest_dist.stddev <= MC_DET_THRESH
            && self.v_threshold_dist.stddev <= MC_DET_THRESH
        {
            LifNeuron::integrate(
                self.cell.v_membrane,
                drive,
                self.trial_v_rest,
                self.trial_v_threshold,
                self.cell.v_reset,
                self.cell.tau_m,
                dt,
            )
        } else {
            self.monte_carlo_step(drive, dt)
        };
        self.cell.v_membrane = v_mean;
        if spiked {
            self.cell.is_refractory = true;
            self.episode_spike_count += 1;
            self.v_rest_dist.set_stddev(SPIKE_STD_RESET);
            self.v_threshold_dist.set_stddev(SPIKE_STD_RESET);
        }
        self.record_step(drive, score_target);
        spiked
    }

    pub fn monte_carlo_step(&mut self, drive: f32, dt: f32) -> (f32, bool) {
        let v0 = self.cell.v_membrane;
        let thr_floor = self.cell.v_reset + 0.05;
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
            let (v_new, fired) = LifNeuron::integrate(
                v0,
                drive,
                v_rest,
                v_thr,
                self.cell.v_reset,
                self.cell.tau_m,
                dt,
            );
            v_sum += v_new;
            if fired {
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
    pub units: Vec<CemLifNeuron>,
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
            let mut n = CemLifNeuron::new(v_rest0, thr0, 0.0, tau);
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
            n.trial_v_threshold = n.v_threshold_dist.mean.max(n.cell.v_reset + 0.1);
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
                .cell
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
            membranes[h] = self.units[h].cell.v_membrane;
            refractory[h] = self.units[h].cell.is_refractory;
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
            self.units[h].cell.v_membrane = snap.membranes[h];
            self.units[h].cell.is_refractory = snap.refractory[h];
            self.units[h].rng.lfsr = snap.rng_states[h];
        }
    }

    /// Clear dynamical state (membranes, delays) without wiping learned readout.
    pub fn reset_dynamics(&mut self) {
        self.prev_v = [0.0; ENSEMBLE_N];
        self.prev_x = [0.0; MAX_DIMS];
        self.prev2_x = [0.0; MAX_DIMS];
        for u in &mut self.units {
            u.cell.v_membrane = u.trial_v_rest;
            u.cell.is_refractory = false;
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
// Spike-emission LM: one LifNeuron per word; spike ⇒ emit that word
// ---------------------------------------------------------------------------

/// LIF time step for the spike-word language model.
/// With τ=5, one step gives v ≈ 0.86·I from rest → peak drive crosses thr=1.
const SPIKE_LM_DT: f32 = 10.0;
/// Membrane time constant (ms-scale abstract units).
const SPIKE_LM_TAU: f32 = 5.0;
/// Rest / reset / threshold for word neurons (threshold 1.0 is the fire line).
const SPIKE_LM_V_REST: f32 = 0.0;
const SPIKE_LM_V_RESET: f32 = 0.0;
const SPIKE_LM_V_THR: f32 = 1.0;
/// Peak synaptic drive for the MLE-best next word (supra-threshold in one step).
const SPIKE_LM_PEAK_DRIVE: f32 = 1.8;
/// Floor drive for the least-likely next under a context (stays subthreshold).
const SPIKE_LM_FLOOR_DRIVE: f32 = 0.05;
/// Softmax temperature when mapping log-probs → drive (lower ⇒ sharper WTA).
const SPIKE_LM_DRIVE_TEMP: f32 = 0.35;
/// Shared context-pool size (leaky LifNeurons that hold state across word steps).
const SPIKE_LM_CTX_POOL: usize = 48;
/// Membrane time constant for context units (slower leak ⇒ longer memory).
const SPIKE_LM_CTX_TAU: f32 = 24.0;
/// High threshold so context units act as subthreshold integrators (rarely spike).
const SPIKE_LM_CTX_THR: f32 = 50.0;
/// Pulse gain when the last emitted word is injected into the context pool.
const SPIKE_LM_CTX_PULSE: f32 = 3.2;
/// Extra zero-drive pool ticks after a word pulse (leak + mix).
const SPIKE_LM_CTX_EXTRA_TICKS: usize = 1;
/// Max sub-steps in the word-layer multi-tick race under shared inhibition.
const SPIKE_LM_MAX_TICKS: usize = 6;
/// Lateral inhibition gain: each word sees `I_j − gain · mean_{k≠j} relu(V_k)`.
const SPIKE_LM_INH_GAIN: f32 = 0.55;
/// Learning rate for pool→word readout (LMS / Hebbian).
const SPIKE_LM_LR: f32 = 0.12;
/// LMS epochs over the train stream when seeding the readout.
const SPIKE_LM_LMS_EPOCHS: usize = 2;
/// Extra drive on the teacher-forced true next during training ticks.
const SPIKE_LM_TEACHER_BOOST: f32 = 1.2;
/// Max word tokens for the Hebbian / LMS pass.
const SPIKE_LM_HEBB_TOKENS: usize = 60_000;
/// Train-time subsample for reporting accuracy (every k-th pair).
const SPIKE_LM_ACC_STRIDE: usize = 2;
/// Clamp on learned pool→word weights.
const SPIKE_LM_WOUT_CLIP: f32 = 2.5;
/// Word tokens used for training (from the start of the tokenized corpus).
/// Vocab size = number of unique words in this train slice (+ `<unk>`).
const SPIKE_LM_TRAIN_WORDS: usize = 100_000;
/// Held-out word window after the train prefix.
const SPIKE_LM_EVAL_WORDS: usize = 6_000;
/// Words to generate after the prompt.
const SPIKE_LM_SAMPLE_WORDS: usize = 40;
/// MCTS simulations per emitted word (more sims ⇒ deeper tree via expansion).
const SPIKE_MCTS_SIMS: usize = 192;
/// Expand only the top-k drive-policy actions at each MCTS node.
const SPIKE_MCTS_TOP_K: usize = 14;
/// PUCT exploration constant for spike-word MCTS.
const SPIKE_MCTS_C_PUCT: f32 = 1.25;
/// Stochastic rollout length (words) after expansion — main search depth signal.
const SPIKE_MCTS_ROLLOUT: usize = 8;
/// Softmax temperature on drive→prior for MCTS (<1 sharpens).
const SPIKE_MCTS_POLICY_TEMP: f32 = 0.65;
/// Log-prob penalty for immediately repeating a word.
const SPIKE_MCTS_REPEAT_PENALTY: f32 = 0.85;

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

/// Word vocabulary for the spike-emission LM (unknown tokens → `<unk>`).
#[derive(Clone, Debug)]
pub struct WordVocab {
    /// id → word string (`0` is always `<unk>`).
    pub id_to_word: Vec<String>,
    /// word → id (missing → unk).
    word_to_id: std::collections::HashMap<String, usize>,
    unk_id: usize,
}

impl WordVocab {
    /// Build a vocab from already-tokenized words: **one entry per unique word**
    /// in `tokens`, ordered by descending frequency (ties broken lexicographically).
    ///
    /// Id `0` is always `<unk>` (for OOV at eval time). Vocab size is
    /// `1 + n_unique` (or `1` if `tokens` is empty).
    pub fn from_tokens(tokens: &[String]) -> Self {
        let mut counts: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for t in tokens {
            *counts.entry(t.clone()).or_insert(0) += 1;
        }
        let mut ranked: Vec<(String, u32)> = counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut id_to_word = Vec::with_capacity(ranked.len() + 1);
        let mut word_to_id = std::collections::HashMap::with_capacity(ranked.len() + 1);
        id_to_word.push("<unk>".to_string());
        word_to_id.insert("<unk>".to_string(), 0);
        for (w, _) in ranked {
            if w == "<unk>" {
                continue;
            }
            let id = id_to_word.len();
            word_to_id.insert(w.clone(), id);
            id_to_word.push(w);
        }
        Self {
            id_to_word,
            word_to_id,
            unk_id: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    pub fn encode(&self, word: &str) -> usize {
        self.word_to_id.get(word).copied().unwrap_or(self.unk_id)
    }

    pub fn decode(&self, id: usize) -> &str {
        self.id_to_word
            .get(id)
            .map(|s| s.as_str())
            .unwrap_or("<unk>")
    }

    /// Encode a token stream to ids.
    pub fn encode_tokens(&self, tokens: &[String]) -> Vec<usize> {
        tokens.iter().map(|t| self.encode(t)).collect()
    }
}

/// Split corpus bytes into word tokens (alphanumeric + apostrophe).
///
/// **Case is preserved**: `"The"` and `"the"` are distinct tokens (and thus
/// distinct vocab ids / LifNeurons when both appear).
///
/// Punctuation (`. , ! ? ; :`) is emitted as its own one-character token
/// (so `be.` → `["be", "."]`).
pub fn tokenize_words(data: &[u8]) -> Vec<String> {
    let s = String::from_utf8_lossy(data);
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() || ch == '\'' {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            match ch {
                '.' | ',' | '!' | '?' | ';' | ':' => words.push(ch.to_string()),
                _ => {}
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
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
// Spike-word language model (LifNeuron per word; fire ⇒ emit)
// ---------------------------------------------------------------------------

/// Word-level LM where each vocabulary word owns one [`LifNeuron`].
///
/// **Context** is a shared pool of [`SPIKE_LM_CTX_POOL`] LifNeurons that:
/// 1. receive a pulse when a word is emitted (fixed random word→pool weights),
/// 2. hold leaky membrane state across word steps (not reset on emit),
/// 3. are read by every word neuron via learned pool→word weights.
///
/// Drive into candidate word \(j\):
///
/// ```text
/// I_j = bias[j] + Σ_c  w_out[j][c] · V_ctx[c]
/// ```
///
/// When a word neuron spikes, that word is **emitted** and injected into the pool.
pub struct SpikeWordLm {
    pub vocab: WordVocab,
    /// `neurons[id]` ↔ vocabulary word `id` (emit on spike).
    pub neurons: Vec<LifNeuron>,
    /// Shared context pool (subthreshold leaky integrators).
    pub ctx: Vec<LifNeuron>,
    /// Fixed word→pool synapses: `w_in[c][word]` scales the inject pulse into unit `c`.
    w_in: Vec<Vec<f32>>,
    /// Learned pool→word readout: `w_out[word][c]`.
    w_out: Vec<Vec<f32>>,
    /// Baseline drive per word (unigram-like).
    bias: Vec<f32>,
    /// Last committed word id (for diagnostics).
    last_word: usize,
    rng: Rand,
}

impl SpikeWordLm {
    pub fn new(vocab: WordVocab, seed: u32) -> Self {
        let v = vocab.len().max(1);
        let c = SPIKE_LM_CTX_POOL.max(1);
        let mut rng = Rand::new(seed.max(1));

        let mut neurons = Vec::with_capacity(v);
        for _ in 0..v {
            neurons.push(LifNeuron::new(
                SPIKE_LM_V_REST,
                SPIKE_LM_V_THR,
                SPIKE_LM_V_RESET,
                SPIKE_LM_TAU,
            ));
        }

        // Context pool: slow leak, high threshold ⇒ continuous memory, rare spikes.
        let mut ctx = Vec::with_capacity(c);
        for _ in 0..c {
            ctx.push(LifNeuron::new(
                SPIKE_LM_V_REST,
                SPIKE_LM_CTX_THR,
                SPIKE_LM_V_RESET,
                SPIKE_LM_CTX_TAU,
            ));
        }

        // Fixed random word→pool projection (±1/√C).
        let inv = 1.0 / (c as f32).sqrt();
        let mut w_in = vec![vec![0.0f32; v]; c];
        for ci in 0..c {
            for wi in 0..v {
                w_in[ci][wi] = if rng.u() < 0.5 { -inv } else { inv };
            }
        }

        // Learned readout starts near zero (filled by LMS on the train stream).
        let w_out = vec![vec![0.0f32; c]; v];
        let unk = vocab.unk_id();

        Self {
            vocab,
            neurons,
            ctx,
            w_in,
            w_out,
            bias: vec![0.0; v],
            last_word: unk,
            rng,
        }
    }

    /// Number of word neurons (== vocab size).
    pub fn n_words(&self) -> usize {
        self.neurons.len()
    }

    /// Size of the shared context pool.
    pub fn ctx_pool_size(&self) -> usize {
        self.ctx.len()
    }

    /// Reset **word** membranes only (context pool is left intact).
    fn reset_word_membranes(&mut self) {
        for n in &mut self.neurons {
            n.reset();
        }
    }

    /// Reset word membranes **and** clear the context pool.
    pub fn reset_state(&mut self) {
        self.reset_word_membranes();
        for n in &mut self.ctx {
            n.reset();
        }
        self.last_word = self.vocab.unk_id();
    }

    /// Inject the emitted word into the shared pool and integrate (state persists).
    fn inject_word_to_pool(&mut self, word_id: usize) {
        let v = self.neurons.len().max(1);
        let w = word_id.min(v - 1);
        let c = self.ctx.len();

        // One strong pulse encoding the word via fixed w_in.
        for ci in 0..c {
            self.ctx[ci].is_refractory = false;
            let drive = SPIKE_LM_CTX_PULSE * self.w_in[ci][w];
            let _ = self.ctx[ci].step(drive, SPIKE_LM_DT);
        }
        // Free evolution: leak mixes past pulses into a multi-word trace.
        for _ in 0..SPIKE_LM_CTX_EXTRA_TICKS {
            for ci in 0..c {
                self.ctx[ci].is_refractory = false;
                let _ = self.ctx[ci].step(0.0, SPIKE_LM_DT);
            }
        }
        self.last_word = w;
    }

    /// Bounded context features (membranes can grow; tanh keeps readout stable).
    fn ctx_features(&self) -> Vec<f32> {
        self.ctx.iter().map(|n| n.v_membrane.tanh()).collect()
    }

    /// Drive into word neurons from bias + linear readout of the context pool.
    fn drives(&self) -> Vec<f32> {
        let v = self.neurons.len();
        let feats = self.ctx_features();
        let c = feats.len();
        let mut d = vec![0.0f32; v];
        for j in 0..v {
            let mut s = self.bias[j];
            let row = &self.w_out[j];
            for ci in 0..c {
                s += row[ci] * feats[ci];
            }
            d[j] = s;
        }
        d
    }

    /// One race tick: **lateral inhibition** from other word membranes, then LIF step.
    ///
    /// Shared inhibition for unit \(j\):
    /// \(I^{\mathrm{eff}}_j = I_j - g \cdot \mathrm{mean}_{k \ne j}\operatorname{relu}(V_k)\).
    /// Returns ids that spiked this tick (usually 0 or 1 after competition).
    fn race_tick(&mut self, base_drives: &[f32]) -> Vec<usize> {
        let v = self.neurons.len();
        if v == 0 {
            return Vec::new();
        }
        let mut act = vec![0.0f32; v];
        let mut total = 0.0f32;
        for j in 0..v {
            let a = self.neurons[j].v_membrane.max(0.0);
            act[j] = a;
            total += a;
        }
        let inv_others = 1.0 / (v.saturating_sub(1).max(1) as f32);
        let g = SPIKE_LM_INH_GAIN;

        let mut spiked = Vec::new();
        for j in 0..v {
            let others_mean = (total - act[j]) * inv_others;
            let i_eff = base_drives[j] - g * others_mean;
            if self.neurons[j].step(i_eff, SPIKE_LM_DT) {
                spiked.push(j);
            }
        }
        spiked
    }

    /// Pick a winner among simultaneous spikers (membrane WTA, optional temp sample).
    fn pick_spiker(&mut self, spiked: &[usize], drives: &[f32], temperature: f32) -> usize {
        if spiked.is_empty() {
            return 0;
        }
        if spiked.len() == 1 {
            return spiked[0];
        }
        let v = self.neurons.len();
        if temperature <= 1e-3 {
            return spiked
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    self.neurons[a]
                        .v_membrane
                        .partial_cmp(&self.neurons[b].v_membrane)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            drives[a]
                                .partial_cmp(&drives[b])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                })
                .unwrap_or(0);
        }
        let mut scores = vec![f32::NEG_INFINITY; v];
        for &j in spiked {
            // Prefer membrane (who actually won the race), blend base drive.
            scores[j] = self.neurons[j].v_membrane + 0.25 * drives[j];
        }
        self.sample_from_scores(&scores, temperature)
    }

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

    /// LMS update of pool→word weights toward target drives given frozen context features.
    fn lms_update_readout(&mut self, true_next: usize, drives: &[f32], ctx_feat: &[f32]) {
        let v = self.neurons.len();
        let c = ctx_feat.len();
        if v == 0 || c == 0 {
            return;
        }
        let target = true_next.min(v - 1);
        let lr = SPIKE_LM_LR;
        let clip = SPIKE_LM_WOUT_CLIP;

        // Pull true next toward PEAK drive.
        let err_t = (SPIKE_LM_PEAK_DRIVE - drives[target]).clamp(-3.0, 3.0);
        for ci in 0..c {
            let g = lr * err_t * ctx_feat[ci];
            self.w_out[target][ci] = (self.w_out[target][ci] + g).clamp(-clip, clip);
        }

        // Push the top few competitors toward FLOOR (hard-negative LMS).
        let mut neg: [(usize, f32); 3] = [(0, f32::NEG_INFINITY); 3];
        for (j, &d) in drives.iter().enumerate() {
            if j == target {
                continue;
            }
            if d > neg[2].1 {
                neg[2] = (j, d);
                // insertion-sort the three slots
                if neg[2].1 > neg[1].1 {
                    neg.swap(1, 2);
                }
                if neg[1].1 > neg[0].1 {
                    neg.swap(0, 1);
                }
            }
        }
        for &(j, d) in &neg {
            if d == f32::NEG_INFINITY {
                continue;
            }
            let err_b = (SPIKE_LM_FLOOR_DRIVE - d).clamp(-3.0, 3.0);
            for ci in 0..c {
                let g = lr * 0.35 * err_b * ctx_feat[ci];
                self.w_out[j][ci] = (self.w_out[j][ci] + g).clamp(-clip, clip);
            }
        }
    }

    /// Commit an emitted word: reset word bank, inject into context pool.
    fn commit_emit(&mut self, emitted: usize) {
        let v = self.neurons.len();
        let e = emitted.min(v.saturating_sub(1));
        for j in 0..v {
            if j == e {
                self.neurons[j].v_membrane = self.neurons[j].v_reset;
                self.neurons[j].is_refractory = true;
            } else {
                self.neurons[j].reset();
            }
        }
        // Context pool keeps prior state and receives the new word pulse.
        self.inject_word_to_pool(e);
    }

    /// Multi-tick **race** under shared lateral inhibition until a word spikes.
    ///
    /// 1. Reset word membranes; freeze pool-derived base drives \(I_j\).
    /// 2. For up to [`SPIKE_LM_MAX_TICKS`] ticks, each unit receives
    ///    \(I_j - g\cdot\mathrm{mean}_{k\ne j}\operatorname{relu}(V_k)\) and integrates.
    /// 3. First tick with ≥1 spike: emit the winner (membrane WTA if several).
    /// 4. Timeout: soft-emit argmax membrane / drive.
    ///
    /// Core rule: **a spike emits that word**.
    pub fn step_emit(&mut self, temperature: f32) -> usize {
        let v = self.neurons.len();
        if v == 0 {
            return 0;
        }
        let drives = self.drives();
        self.reset_word_membranes();

        for _tick in 0..SPIKE_LM_MAX_TICKS {
            let spiked = self.race_tick(&drives);
            if spiked.is_empty() {
                continue;
            }
            let emitted = self.pick_spiker(&spiked, &drives, temperature);
            self.commit_emit(emitted);
            return emitted;
        }

        // No spike in the race window: soft-emit from membrane + base drive.
        let mut scores = vec![0.0f32; v];
        for j in 0..v {
            scores[j] = self.neurons[j].v_membrane + 0.35 * drives[j];
        }
        let emitted = if temperature <= 1e-3 {
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            self.sample_from_scores(&scores, temperature)
        };
        self.commit_emit(emitted);
        emitted
    }

    /// Teacher-force observe a true word (no weight update): inject into pool.
    pub fn observe(&mut self, word_id: usize) {
        self.commit_emit(word_id);
    }

    /// Map log-probabilities into a drive interval with a sharp softmax peak.
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
        for (j, &lp) in logp.iter().enumerate() {
            let p = ((lp - max_lp) / t).exp() / z;
            out[j] = SPIKE_LM_FLOOR_DRIVE + span * p;
        }
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

    /// Seed unigram bias, then LMS-fit pool→word readout on the train stream.
    fn fit_weights_from_ids(&mut self, ids: &[usize]) {
        let v = self.neurons.len();
        if v == 0 || ids.len() < 2 {
            return;
        }
        let mut uni = vec![1.0f32; v];
        for &id in ids {
            uni[id.min(v - 1)] += 1.0;
        }
        let uni_tot: f32 = uni.iter().sum::<f32>().max(1.0);
        let mut uni_lp = vec![0.0f32; v];
        for j in 0..v {
            uni_lp[j] = (uni[j] / uni_tot).ln();
        }
        self.bias = Self::logp_to_drive(&uni_lp);
        for b in &mut self.bias {
            *b *= 0.08; // weak prior; pool readout should dominate
        }

        // Teacher-forced LMS: pool holds multi-word trace; readout learns next word.
        let n_fit = ids.len() - 1;
        for _epoch in 0..SPIKE_LM_LMS_EPOCHS {
            self.reset_state();
            self.observe(ids[0].min(v - 1));
            for i in 0..n_fit {
                let true_next = ids[i + 1].min(v - 1);
                let feats = self.ctx_features();
                let drives = self.drives();
                self.lms_update_readout(true_next, &drives, &feats);
                self.observe(true_next);
            }
        }
    }

    fn hebb_step(&mut self, true_next: usize) -> bool {
        let v = self.neurons.len();
        if v == 0 {
            return false;
        }
        let target = true_next.min(v - 1);
        let feats = self.ctx_features();
        let drives_raw = self.drives();
        let mut drives = drives_raw.clone();
        drives[target] += SPIKE_LM_TEACHER_BOOST;

        self.reset_word_membranes();
        let mut spiked = Vec::new();
        for _tick in 0..SPIKE_LM_MAX_TICKS {
            spiked = self.race_tick(&drives);
            if !spiked.is_empty() {
                break;
            }
        }

        let correct = spiked.contains(&target);
        self.lms_update_readout(target, &drives_raw, &feats);
        if !correct {
            let err = (SPIKE_LM_PEAK_DRIVE - drives_raw[target]).clamp(-3.0, 3.0);
            for ci in 0..feats.len() {
                let g = SPIKE_LM_LR * err * feats[ci];
                self.w_out[target][ci] =
                    (self.w_out[target][ci] + g).clamp(-SPIKE_LM_WOUT_CLIP, SPIKE_LM_WOUT_CLIP);
            }
        }

        self.commit_emit(target);
        correct
    }

    fn predict_next(&mut self) -> usize {
        // Snapshot word + context dynamics so eval does not advance the pool.
        let snap_last = self.last_word;
        let word_mem: Vec<LifNeuron> = self.neurons.clone();
        let ctx_mem: Vec<LifNeuron> = self.ctx.clone();
        let rng_lfsr = self.rng.lfsr;

        let pred = self.step_emit(0.0);

        self.last_word = snap_last;
        self.rng.lfsr = rng_lfsr;
        self.neurons = word_mem;
        self.ctx = ctx_mem;
        pred
    }

    fn score_stream(&mut self, ids: &[usize], stride: usize) -> LmTrainStats {
        if ids.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }
        let v = self.neurons.len().max(1);
        let stride = stride.max(1);
        self.reset_state();
        self.observe(ids[0].min(v - 1));
        let mut correct = 0u32;
        let mut tokens = 0u32;
        let mut nll = 0.0f32;
        let n_pairs = ids.len() - 1;
        let mut i = 0;
        while i < n_pairs {
            let true_next = ids[i + 1].min(v - 1);
            if i % stride == 0 {
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
                let log_p = drives[true_next] - max_d - z.max(1e-12).ln();
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

    /// Fit n-gram synaptic weights, optional Hebbian refinement on word ids.
    pub fn train_ids(&mut self, ids: &[usize], hebb_epochs: usize) -> LmTrainStats {
        if ids.len() < 2 {
            return LmTrainStats {
                tokens: 0,
                loss: 0.0,
                accuracy: 0.0,
                perplexity: 1.0,
            };
        }
        self.fit_weights_from_ids(ids);

        let v = self.neurons.len().max(1);
        let n_pairs = ids.len() - 1;
        let hebb_n = SPIKE_LM_HEBB_TOKENS.min(n_pairs);
        for _ in 0..hebb_epochs.max(0) {
            self.reset_state();
            self.observe(ids[0].min(v - 1));
            for i in 0..hebb_n {
                let next = ids[i + 1].min(v - 1);
                let _ = self.hebb_step(next);
            }
        }

        self.score_stream(ids, SPIKE_LM_ACC_STRIDE)
    }

    /// Held-out accuracy / perplexity with teacher-forced word context.
    pub fn evaluate_ids(&mut self, ids: &[usize]) -> LmTrainStats {
        self.score_stream(ids, 1)
    }

    /// Snapshot pool + word membranes for MCTS branching.
    fn snapshot_spike(&self) -> SpikeWordSnap {
        SpikeWordSnap {
            neurons: self.neurons.clone(),
            ctx: self.ctx.clone(),
            last_word: self.last_word,
            rng_lfsr: self.rng.lfsr,
        }
    }

    fn restore_spike(&mut self, snap: &SpikeWordSnap) {
        self.neurons.clone_from(&snap.neurons);
        self.ctx.clone_from(&snap.ctx);
        self.last_word = snap.last_word;
        self.rng.lfsr = snap.rng_lfsr;
    }

    /// Softmax policy over generation drives from the context pool.
    fn gen_policy(&self) -> Vec<f32> {
        let drives = self.drives();
        let t = SPIKE_MCTS_POLICY_TEMP.max(1e-3);
        let mut max_d = f32::NEG_INFINITY;
        for &d in &drives {
            if d > max_d {
                max_d = d;
            }
        }
        let mut p = vec![0.0f32; drives.len()];
        let mut z = 0.0f32;
        for (i, &d) in drives.iter().enumerate() {
            let e = ((d - max_d) / t).exp();
            p[i] = e;
            z += e;
        }
        let inv = 1.0 / z.max(1e-12);
        for x in &mut p {
            *x *= inv;
        }
        p
    }

    fn top_k_actions(probs: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..probs.len()).collect();
        idx.sort_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k.min(probs.len()).max(1));
        idx
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
        let mut pick = logits.len().saturating_sub(1);
        for (i, &p) in logits.iter().enumerate() {
            if r < p {
                pick = i;
                break;
            }
            r -= p;
        }
        pick
    }

    /// Log-prob of taking `act` under `probs`, with a mild anti-stutter penalty.
    fn action_value_lp(&self, probs: &[f32], act: usize, last_word: usize) -> f32 {
        let mut lp = probs.get(act).copied().unwrap_or(1e-12).max(1e-12).ln();
        if act == last_word {
            lp -= SPIKE_MCTS_REPEAT_PENALTY;
        }
        lp
    }

    /// Choose next word with **MCTS + PUCT** under the spike-drive policy.
    ///
    /// Prior at each state = softmax of generation drives (period-boosted).
    /// State is the shared context pool + last word; branching snapshots and
    /// restores pool membranes so real generation only commits the final pick.
    fn mcts_select_word(&mut self, temperature: f32) -> usize {
        let root_snap = self.snapshot_spike();
        let root_prior = self.gen_policy();
        let root_actions = Self::top_k_actions(&root_prior, SPIKE_MCTS_TOP_K);

        let mut nodes: Vec<MctsNode> = vec![MctsNode {
            action: 0,
            prior: 1.0,
            n: 0.0,
            w: 0.0,
            children: Vec::new(),
            unexpanded: root_actions
                .iter()
                .map(|&a| (a, root_prior[a].max(MCTS_PRIOR_FLOOR)))
                .collect(),
        }];

        for _ in 0..SPIKE_MCTS_SIMS {
            self.restore_spike(&root_snap);
            let mut path: Vec<usize> = vec![0];
            let mut node = 0usize;
            let mut path_logp = 0.0f32;
            let mut cur_last = self.last_word;

            // Selection: PUCT among expanded children.
            while nodes[node].unexpanded.is_empty() && !nodes[node].children.is_empty() {
                let parent_n = nodes[node].n.max(1.0);
                let mut best_child = nodes[node].children[0];
                let mut best_score = f32::NEG_INFINITY;
                for &ch in &nodes[node].children {
                    let c = &nodes[ch];
                    let q = if c.n > 0.0 { c.w / c.n } else { 0.0 };
                    let u = SPIKE_MCTS_C_PUCT * c.prior * parent_n.sqrt() / (1.0 + c.n);
                    let score = q + u;
                    if score > best_score {
                        best_score = score;
                        best_child = ch;
                    }
                }
                let act = nodes[best_child].action;
                let probs = self.gen_policy();
                path_logp += self.action_value_lp(&probs, act, cur_last);
                self.observe(act);
                cur_last = act;
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
                let probs = self.gen_policy();
                let lp = self.action_value_lp(&probs, act, cur_last);
                if path.len() == 1 {
                    first_step_lp = lp;
                }
                path_logp += lp;
                self.observe(act);
                cur_last = act;

                let child_prior = self.gen_policy();
                let child_actions = Self::top_k_actions(&child_prior, SPIKE_MCTS_TOP_K);
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

            // Short stochastic rollout under drive policy.
            let mut rollout_logp = 0.0f32;
            let roll_temp = temperature.clamp(0.45, 0.85);
            for _ in 0..SPIKE_MCTS_ROLLOUT {
                let probs = self.gen_policy();
                let a = self.sample_from_probs(&probs, roll_temp);
                rollout_logp += self.action_value_lp(&probs, a, cur_last);
                self.observe(a);
                cur_last = a;
            }

            let deep_steps = (path.len().saturating_sub(1) + SPIKE_MCTS_ROLLOUT).max(1) as f32;
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

        self.restore_spike(&root_snap);

        if nodes[0].children.is_empty() {
            return argmax_f32(&root_prior);
        }
        // Prefer most-visited child; break ties with Q and prior.
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

    /// One-step greedy/sampled emission (no tree search).
    pub fn generate_direct(&mut self, prompt: &str, n: usize, temperature: f32) -> String {
        self.reset_state();
        let prompt_tokens = tokenize_words(prompt.as_bytes());
        let mut out_words: Vec<String> = if prompt_tokens.is_empty() {
            vec!["the".to_string()]
        } else {
            prompt_tokens
        };
        for w in &out_words {
            self.observe(self.vocab.encode(w));
        }
        for _ in 0..n {
            let id = self.step_emit(temperature);
            out_words.push(self.vocab.decode(id).to_string());
        }
        join_word_tokens(&out_words)
    }

    /// Generate `n` words after `prompt` with **MCTS + PUCT** on spike drives.
    ///
    /// Each step runs [`SPIKE_MCTS_SIMS`] simulations. Prompt tokens are
    /// teacher-forced into the context pool.
    pub fn generate(&mut self, prompt: &str, n: usize, temperature: f32) -> String {
        self.reset_state();
        let prompt_tokens = tokenize_words(prompt.as_bytes());
        let mut out_words: Vec<String> = if prompt_tokens.is_empty() {
            vec!["the".to_string()]
        } else {
            prompt_tokens
        };
        for w in &out_words {
            self.observe(self.vocab.encode(w));
        }
        for _ in 0..n {
            let id = self.mcts_select_word(temperature);
            self.observe(id);
            out_words.push(self.vocab.decode(id).to_string());
        }
        join_word_tokens(&out_words)
    }
}

/// Dynamics snapshot for spike-word MCTS (word bank + context pool + RNG).
#[derive(Clone, Debug)]
struct SpikeWordSnap {
    neurons: Vec<LifNeuron>,
    ctx: Vec<LifNeuron>,
    last_word: usize,
    rng_lfsr: u32,
}

/// Join word tokens with spaces, attaching single-char punctuation without a
/// preceding space (`word` + `.` → `word.`).
fn join_word_tokens(words: &[String]) -> String {
    let mut s = String::new();
    for w in words {
        let is_punct = w.len() == 1
            && matches!(w.as_bytes()[0], b'.' | b',' | b'!' | b'?' | b';' | b':');
        if s.is_empty() {
            s.push_str(w);
        } else if is_punct {
            s.push_str(w);
        } else {
            s.push(' ');
            s.push_str(w);
        }
    }
    s
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

/// Spike-emission word LM: one [`LifNeuron`] per word; a spike emits that word.
fn run_spike_word_lm(path: &str) -> Result<(), String> {
    println!();
    println!("=== Spike-word LM (LifNeuron per word; fire ⇒ emit) ===");
    println!("corpus: {path}");

    let corpus = load_corpus(path)?;
    let all_tokens = tokenize_words(&corpus);
    if all_tokens.len() < 256 {
        return Err(format!(
            "too few word tokens after tokenize ({})",
            all_tokens.len()
        ));
    }
    let train_end = SPIKE_LM_TRAIN_WORDS.min(all_tokens.len().saturating_sub(SPIKE_LM_EVAL_WORDS + 2));
    let eval_end = (train_end + SPIKE_LM_EVAL_WORDS).min(all_tokens.len());
    if train_end < 128 {
        return Err(format!(
            "word stream too short for train/eval split ({} tokens)",
            all_tokens.len()
        ));
    }
    let train_toks = &all_tokens[..train_end];
    let eval_toks = &all_tokens[train_end..eval_end];

    // Vocab = every unique train word (+ `<unk>`); each word owns one neuron.
    let vocab = WordVocab::from_tokens(train_toks);
    let train_ids = vocab.encode_tokens(train_toks);
    let eval_ids = vocab.encode_tokens(eval_toks);
    let n_unique = vocab.len().saturating_sub(1); // exclude <unk>

    println!(
        "words: corpus_tokens={} train={} eval={} unique={} vocab={} word_neurons={} ctx_pool={} hebb_cap={} max_ticks={}",
        all_tokens.len(),
        train_ids.len(),
        eval_ids.len(),
        n_unique,
        vocab.len(),
        vocab.len(),
        SPIKE_LM_CTX_POOL,
        SPIKE_LM_HEBB_TOKENS,
        SPIKE_LM_MAX_TICKS
    );

    let mut model = SpikeWordLm::new(vocab, 0x51A6E);
    let train_stats = model.train_ids(&train_ids, 1);
    println!(
        "train (word n-gram synapses + Hebb≤{SPIKE_LM_HEBB_TOKENS}): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        train_stats.tokens, train_stats.accuracy, train_stats.loss, train_stats.perplexity
    );

    let eval_stats = model.evaluate_ids(&eval_ids);
    println!(
        "eval (spike emit word): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        eval_stats.tokens, eval_stats.accuracy, eval_stats.loss, eval_stats.perplexity
    );

    let prompt = "To be, or not to be";
    println!();
    println!(
        "spike-MCTS generate: sims={SPIKE_MCTS_SIMS} top_k={SPIKE_MCTS_TOP_K} rollout={SPIKE_MCTS_ROLLOUT} \
         c_puct={SPIKE_MCTS_C_PUCT} n={SPIKE_LM_SAMPLE_WORDS}"
    );
    let sample = model.generate(prompt, SPIKE_LM_SAMPLE_WORDS, 0.7);
    println!("sample spike-MCTS (temp=0.7, prompt+{SPIKE_LM_SAMPLE_WORDS} words):");
    println!("----");
    println!("{sample}");
    println!("----");
    println!(
        "{{spike_word_lm: {{\"train_acc\": {:.6}, \"eval_acc\": {:.6}, \"eval_ppl\": {:.4}, \"neurons\": {}, \"mcts_sims\": {}}}}}",
        train_stats.accuracy,
        eval_stats.accuracy,
        eval_stats.perplexity,
        model.n_words(),
        SPIKE_MCTS_SIMS
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
    fn test_simple_lif_steps_and_spikes() {
        let mut n = LifNeuron::new(0.0, 1.0, 0.0, 5.0);
        // Strong drive should cross thr=1 in one step (v ≈ 0.86·I with dt=10,τ=5).
        assert!(n.step(1.8, 10.0), "expected spike from supra-threshold drive");
        assert!(n.is_refractory);
        assert!(!n.step(1.8, 10.0), "refractory step should not spike");
        assert_eq!(n.v_membrane, n.v_reset);
        n.reset();
        assert!(!n.is_refractory);
        assert_eq!(n.v_membrane, n.v_rest);
    }

    #[test]
    fn test_spike_resets_search_stddevs() {
        let mut neuron = CemLifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist.set_stddev(0.01);
        neuron.v_threshold_dist.set_stddev(0.01);
        neuron.cell.v_membrane = 10.0;
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
        let mut neuron = CemLifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.cell.v_membrane = 10.0;
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
        assert_eq!(neuron.cell.v_membrane, neuron.cell.v_reset);
    }

    #[test]
    fn test_firing_is_probabilistic_near_threshold() {
        // V near thr mean with non-zero σ ⇒ Monte Carlo majority is stochastic.
        let mut neuron = CemLifNeuron::new(0.0, 1.0, 0.0, 1.0e9); // huge τ → V ≈ fixed
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.5);
        neuron.cell.v_membrane = 1.0;
        neuron.pending_antithetic = None;
        neuron.episode_step = 0;

        let mut spikes = 0u32;
        let trials = 200u32;
        for _ in 0..trials {
            neuron.cell.v_membrane = 1.0;
            neuron.cell.is_refractory = false;
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
        let mut neuron = CemLifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.3);
        neuron.pending_antithetic = None;

        for _ in 0..50 {
            neuron.cell.v_membrane = 5.0;
            neuron.cell.is_refractory = false;
            neuron.episode_step = 0;
            assert!(
                neuron.step(0.0, 1.0),
                "V far above thr should almost surely spike under MC majority"
            );
        }
    }

    #[test]
    fn test_monte_carlo_majority_fire_decision() {
        let mut neuron = CemLifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 0.0;
        neuron.v_rest_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.v_threshold_dist = GaussianParam::new(0.0, STD_MIN);
        neuron.cell.v_membrane = 1.0;
        neuron.pending_antithetic = None;
        neuron.episode_step = 0;
        // Tiny σ, V well above thr mean ⇒ every MC micro-step fires ⇒ majority fire.
        let (v_hi, fire_hi) = neuron.monte_carlo_step(0.0, 1.0);
        assert!(fire_hi, "expected majority fire when V >> thr");
        assert!(v_hi.is_finite());

        neuron.cell.v_membrane = -2.0;
        let (v_lo, fire_lo) = neuron.monte_carlo_step(0.0, 1.0);
        assert!(!fire_lo, "expected no fire when V << thr");
        assert!(v_lo.is_finite());
    }

    #[test]
    fn test_cem_prefers_low_error_params() {
        let mut neuron = CemLifNeuron::new(5.0, 5.0, 0.0, 10.0);
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
        let mut neuron = CemLifNeuron::new(3.0, 4.0, 0.0, 10.0);
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
    fn test_tokenize_words_and_vocab() {
        let toks = tokenize_words(b"To be, or not to be.");
        assert_eq!(
            toks,
            vec!["To", "be", ",", "or", "not", "to", "be", "."]
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>()
        );
        // "." is its own token, not glued onto "be".
        assert!(toks.iter().any(|t| t == "."));
        assert!(!toks.iter().any(|t| t != "." && t.contains('.')));

        let vocab = WordVocab::from_tokens(&toks);
        assert_eq!(vocab.decode(0), "<unk>");
        // unique: To, be, ,, or, not, to, . → 7 + <unk> (case-sensitive)
        assert_eq!(vocab.len(), 8);
        assert_ne!(vocab.encode("To"), vocab.encode("to"));
        assert_eq!(vocab.encode("be"), vocab.encode("be"));
        assert_eq!(vocab.decode(vocab.encode(".")), ".");
        assert_eq!(vocab.decode(vocab.encode("xyzzy")), "<unk>");
        assert_eq!(join_word_tokens(&toks), "To be, or not to be.");
    }

    #[test]
    fn test_spike_word_lm_maps_neuron_to_emit() {
        // Tight word loop: n-gram drives + spike emission should recover the cycle.
        let text = "cat dog fish cat dog fish cat dog fish cat dog fish cat dog fish cat dog fish ";
        let tokens = tokenize_words(text.as_bytes());
        let vocab = WordVocab::from_tokens(&tokens);
        // <unk> + cat, dog, fish
        assert_eq!(vocab.len(), 4);
        let ids = vocab.encode_tokens(&tokens);
        let mut model = SpikeWordLm::new(vocab, 99);
        assert_eq!(model.n_words(), model.vocab.len());
        assert_eq!(model.ctx_pool_size(), SPIKE_LM_CTX_POOL);

        let stats = model.train_ids(&ids, 1);
        assert!(stats.tokens > 0);
        assert!(
            stats.accuracy > 0.5,
            "spike-word LM should learn cat/dog/fish loop, acc={}",
            stats.accuracy
        );
        assert!(stats.perplexity.is_finite() && stats.perplexity >= 1.0);

        // After observing "cat", the "dog" neuron should fire.
        model.reset_state();
        model.observe(model.vocab.encode("cat"));
        let emitted = model.step_emit(0.0);
        assert_eq!(
            model.vocab.decode(emitted),
            "dog",
            "after 'cat', expected 'dog' neuron to spike/emit"
        );

        // Direct emit (no tree) for a cheap cycle check.
        let sample = model.generate_direct("cat dog", 9, 0.0);
        assert!(sample.contains("fish") || sample.contains("cat") || sample.contains("dog"));
        let sample_toks = tokenize_words(sample.as_bytes());
        assert!(sample_toks.len() >= 5);

        // Short MCTS sample exercises search + pool snapshots.
        let mcts_sample = model.generate("cat", 4, 0.5);
        assert!(mcts_sample.split_whitespace().count() >= 3);
        assert!(std::str::from_utf8(mcts_sample.as_bytes()).is_ok());
    }

    #[test]
    fn test_spike_word_lm_shakespeare_prefix() {
        let corpus = load_corpus(LM_CORPUS_PATH).expect("100.txt.utf-8 should exist");
        // Modest prefix keeps the pool LMS test fast under debug builds.
        let prefix = &corpus[..40_000.min(corpus.len())];
        let tokens = tokenize_words(prefix);
        let train_end = (tokens.len() * 9 / 10).max(64);
        let train_toks = &tokens[..train_end];
        let eval_toks = &tokens[train_end..];
        let vocab = WordVocab::from_tokens(train_toks);
        // Vocab covers every unique train word (+ <unk>).
        assert!(vocab.len() > 50);
        let train_ids = vocab.encode_tokens(train_toks);
        let eval_ids = vocab.encode_tokens(eval_toks);
        let mut model = SpikeWordLm::new(vocab, 7);
        assert_eq!(model.ctx_pool_size(), SPIKE_LM_CTX_POOL);
        // LMS-only fit (0 Hebb epochs) — exercises the context pool path quickly.
        let train_stats = model.train_ids(&train_ids, 0);
        assert!(
            train_stats.accuracy > 0.05,
            "train acc={}",
            train_stats.accuracy
        );
        let eval_stats = model.evaluate_ids(&eval_ids);
        assert!(
            eval_stats.accuracy > 0.03,
            "expected pool-context spike LM above chance, eval_acc={}",
            eval_stats.accuracy
        );
        let sample = model.generate_direct("to be or not", 20, 0.5);
        assert!(!sample.is_empty());
        let uniq: std::collections::BTreeSet<&str> = sample.split_whitespace().collect();
        assert!(
            uniq.len() >= 3,
            "degenerate sample: {sample:?}"
        );
        // MCTS path (few words) on a warm pool after direct gen.
        let mcts = model.generate("to be", 3, 0.6);
        assert!(mcts.split_whitespace().count() >= 3);
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

/// Which demo stages to run. With no CLI flags, all stages run.
#[derive(Clone, Copy, Debug)]
struct RunFlags {
    /// CEM-LIF ensemble benchmark suite.
    ensemble: bool,
    /// Character LifEnsemble + RF block-adjacency LM.
    language_model: bool,
    /// Spike-emission word LM (+ MCTS).
    spike_word: bool,
}

impl RunFlags {
    /// Parse `args` (excluding program name). No stage flags ⇒ run everything.
    fn from_args(args: &[String]) -> Result<Self, String> {
        let mut ensemble = false;
        let mut language_model = false;
        let mut spike_word = false;
        let mut any_stage = false;

        for a in args {
            match a.as_str() {
                "-h" | "--help" => {
                    print_cli_help();
                    std::process::exit(0);
                }
                "--ensemble" | "--cem" | "--bench" => {
                    ensemble = true;
                    any_stage = true;
                }
                "--lm" | "--language-model" => {
                    language_model = true;
                    any_stage = true;
                }
                "--spike" | "--spike-word" | "--spike-lm" => {
                    spike_word = true;
                    any_stage = true;
                }
                other => {
                    return Err(format!(
                        "unknown argument: {other}\nRun with --help for usage."
                    ));
                }
            }
        }

        if !any_stage {
            // Default: full pipeline.
            ensemble = true;
            language_model = true;
            spike_word = true;
        }
        Ok(Self {
            ensemble,
            language_model,
            spike_word,
        })
    }
}

fn print_cli_help() {
    println!(
        "\
lif — CEM-LIF ensemble, character LM, spike-word LM

USAGE:
    lif [OPTIONS]

OPTIONS:
    --ensemble, --cem, --bench
            Run CEM-LIF ensemble benchmark suite only (or with other flags).

    --lm, --language-model
            Run character LifEnsemble + RF block-adjacency language model.

    --spike, --spike-word, --spike-lm
            Run spike-emission word language model (context pool + MCTS).

    -h, --help
            Show this help.

With no stage flags, all three stages run in order.
Combine flags to run a subset, e.g.  lif --spike --lm
"
    );
}

/// CEM-LIF ensemble tracking / next-step benchmarks.
fn run_cem_ensemble_benchmarks() {
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
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flags = match RunFlags::from_args(&args) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    if flags.ensemble {
        run_cem_ensemble_benchmarks();
    }

    if flags.language_model {
        if let Err(e) = run_language_model(LM_CORPUS_PATH) {
            eprintln!("language model error: {e}");
            std::process::exit(1);
        }
    }

    if flags.spike_word {
        if let Err(e) = run_spike_word_lm(LM_CORPUS_PATH) {
            eprintln!("spike-word language model error: {e}");
            std::process::exit(1);
        }
    }
}
