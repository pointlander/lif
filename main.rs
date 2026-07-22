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
/// Keep exploration alive; clamp explosion.
const STD_MIN: f32 = 0.05;
const STD_MAX: f32 = 2.0;
/// Extra cost when a spike resets the membrane during tracking.
const SPIKE_PENALTY: f32 = 0.35;
/// Mild preference for small |v_rest| (steady-state V ≈ v_rest + I tracks I best at 0).
const V_REST_L2: f32 = 0.01;
/// Monte Carlo draws of (v_rest, v_threshold) per step (odd ⇒ no ties).
const MC_SAMPLES: usize = 5;

/// Hidden LIF units in the reservoir ensemble.
const ENSEMBLE_N: usize = 32;
/// Normalized-LMS step size for the linear readout (decays mildly over a run).
const READOUT_LR: f32 = 0.4;
/// L2 weight decay on readout weights.
const READOUT_L2: f32 = 1e-4;
/// Spectral radius target for sparse recurrent weights.
const REC_RADIUS: f32 = 0.4;
/// Clamp membrane features / drives so a spike storm cannot blow up SGD.
const FEATURE_CLIP: f32 = 4.0;
/// Clamp absolute readout weights.
const WEIGHT_CLIP: f32 = 4.0;
/// Online passes over each series (last pass is scored).
const TRAIN_PASSES: usize = 3;
/// Max state dimension across benchmarks / ensemble I/O (one-hot, Lorenz, …).
const MAX_DIMS: usize = 4;

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
            stddev: stddev.clamp(STD_MIN, STD_MAX),
        }
    }

    pub fn sample(&self, z: f32) -> f32 {
        self.mean + z * self.stddev
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
        // Keep exploration alive while the mean is still traveling; pure elite
        // variance collapses when the whole population sits in a bad cluster.
        let travel = (self.mean - old_mean).abs();
        let target_std = elite_std.max(travel).max(STD_MIN);
        self.stddev =
            ((1.0 - LR_STD) * self.stddev + LR_STD * target_std).clamp(STD_MIN, STD_MAX);
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
            v_rest_dist: GaussianParam::new(v_rest, 0.8),
            v_threshold_dist: GaussianParam::new(v_threshold, 0.8),
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

    /// Monte Carlo LIF step under drive `drive`, score membrane vs `score_target`.
    ///
    /// Draws `MC_SAMPLES` independent parameter pairs from the search laws
    /// centered on the CEM trial particle:
    ///
    /// ```text
    /// v_rest      ~ N(trial_v_rest,      v_rest_dist.stddev)
    /// v_threshold ~ N(trial_v_threshold, v_threshold_dist.stddev)
    /// ```
    ///
    /// For each draw, integrates one exact subthreshold step from the current
    /// membrane, then:
    /// - sets `v_membrane` to the mean of the sampled post-step voltages
    /// - spikes only if a **strict majority** of those sampled steps would fire
    pub fn step_with_target(&mut self, drive: f32, score_target: f32, dt: f32) -> bool {
        // Refractory: hold at reset, still score so spikes that wreck the
        // trajectory are charged to the active trial parameters.
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            self.record_step(drive, score_target);
            return false;
        }

        let (v_mean, spiked) = self.monte_carlo_step(drive, dt);
        self.v_membrane = v_mean;
        if spiked {
            self.is_refractory = true;
            self.episode_spike_count += 1;
        }

        self.record_step(drive, score_target);
        spiked
    }

    /// Run `MC_SAMPLES` LIF micro-steps with params drawn from
    /// `v_rest_dist` / `v_threshold_dist` (trial mean, learned stddev).
    /// Returns `(mean_post_voltage, majority_fired)`.
    fn monte_carlo_step(&mut self, drive: f32, dt: f32) -> (f32, bool) {
        let v0 = self.v_membrane;
        let decay = (-dt / self.tau_m.max(1e-3)).exp();
        let thr_floor = self.v_reset + 0.05;

        // Episode law: CEM particle as mean, search-dist stddev as MC noise.
        let rest_law =
            GaussianParam::new(self.trial_v_rest, self.v_rest_dist.stddev);
        let thr_law =
            GaussianParam::new(self.trial_v_threshold, self.v_threshold_dist.stddev);

        let mut fire_votes = 0usize;
        let mut v_sum = 0.0f32;

        for _ in 0..MC_SAMPLES {
            let (z_rest, z_thr) = self.rng.g();
            let v_rest = rest_law.sample(z_rest);
            let v_thr = thr_law.sample(z_thr).max(thr_floor);

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
    /// Sparse recurrent mix of previous membrane voltages.
    w_rec: Vec<[f32; ENSEMBLE_N]>,
    /// Hidden readout: contribution of membrane V_h to output dim d.
    w_out: Vec<[f32; ENSEMBLE_N]>,
    /// Input skip: contribution of x_i to output dim d (critical for next-step).
    w_skip: Vec<[f32; MAX_DIMS]>,
    /// One-step delay skip: contribution of x_i[t-1] (temporal context).
    w_delay: Vec<[f32; MAX_DIMS]>,
    bias: [f32; MAX_DIMS],
    prev_v: [f32; ENSEMBLE_N],
    prev_x: [f32; MAX_DIMS],
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
        let mut w_rec = Vec::with_capacity(ENSEMBLE_N);

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

            let mut row_rec = [0.0f32; ENSEMBLE_N];
            // Sparse recurrence (~4 taps per unit).
            for _ in 0..4 {
                let j = (rng.u32() as usize) % ENSEMBLE_N;
                row_rec[j] += 0.4 * rng.signed();
            }
            w_rec.push(row_rec);
        }

        // Rescale recurrence toward REC_RADIUS (row L1 proxy).
        let mut max_abs = 1e-6f32;
        for row in &w_rec {
            let s: f32 = row.iter().map(|w| w.abs()).sum();
            if s > max_abs {
                max_abs = s;
            }
        }
        let rec_scale = REC_RADIUS / max_abs;
        for row in &mut w_rec {
            for w in row.iter_mut() {
                *w *= rec_scale;
            }
        }

        let mut w_out = Vec::with_capacity(out_dims);
        let mut w_skip = Vec::with_capacity(out_dims);
        let mut w_delay = Vec::with_capacity(out_dims);
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
                skip[d] = 0.5;
            }
            w_skip.push(skip);
            w_delay.push([0.0; MAX_DIMS]);
        }

        Self {
            units,
            w_in,
            w_rec,
            w_out,
            w_skip,
            w_delay,
            bias: [0.0; MAX_DIMS],
            prev_v: [0.0; ENSEMBLE_N],
            prev_x: [0.0; MAX_DIMS],
            in_dims,
            out_dims,
            lr: READOUT_LR,
            step_count: 0,
        }
    }

    /// One reservoir step: drive hidden LIFs, form readout, NLMS on (ŷ − target).
    pub fn step(&mut self, x: &[f32], target: &[f32], dt: f32) -> [f32; MAX_DIMS] {
        debug_assert!(x.len() >= self.in_dims);
        debug_assert!(target.len() >= self.out_dims);

        // --- hidden drives ---
        let mut drives = [0.0f32; ENSEMBLE_N];
        for h in 0..ENSEMBLE_N {
            let mut drive = 0.0;
            for d in 0..self.in_dims {
                drive += self.w_in[h][d] * x[d];
            }
            for j in 0..ENSEMBLE_N {
                drive += self.w_rec[h][j] * self.prev_v[j];
            }
            drives[h] = drive;
        }

        // Units CEM-track their drive (keeps features ≈ projections of x + state).
        for h in 0..ENSEMBLE_N {
            let drive = drives[h].clamp(-FEATURE_CLIP, FEATURE_CLIP);
            let _ = self.units[h].step_with_target(drive, drive, dt);
            self.prev_v[h] = self.units[h]
                .v_membrane
                .clamp(-FEATURE_CLIP, FEATURE_CLIP);
        }

        // Feature energy for normalized LMS.
        let mut energy = 1.0f32; // bias feature ≡ 1
        for h in 0..ENSEMBLE_N {
            energy += self.prev_v[h] * self.prev_v[h];
        }
        for d in 0..self.in_dims {
            energy += x[d] * x[d] + self.prev_x[d] * self.prev_x[d];
        }

        // --- linear readout: y = b + W_h V + W_x x + W_Δ x_prev ---
        let mut pred = [0.0f32; MAX_DIMS];
        for d in 0..self.out_dims {
            let mut y = self.bias[d];
            for h in 0..ENSEMBLE_N {
                y += self.w_out[d][h] * self.prev_v[h];
            }
            for i in 0..self.in_dims {
                y += self.w_skip[d][i] * x[i];
                y += self.w_delay[d][i] * self.prev_x[i];
            }
            pred[d] = y.clamp(-FEATURE_CLIP * 2.0, FEATURE_CLIP * 2.0);
        }

        self.step_count = self.step_count.saturating_add(1);
        let lr = self.lr / (1.0 + 0.001 * self.step_count as f32);
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
            }
        }

        for d in 0..self.in_dims {
            self.prev_x[d] = x[d];
        }

        pred
    }
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
// Character-level natural language model (LIF reservoir + NLMS readout)
// ---------------------------------------------------------------------------

/// Hidden LIF units for the language model (larger than the toy ensemble).
const LM_HIDDEN: usize = 128;
/// Sparse recurrent taps per hidden unit.
const LM_REC_TAPS: usize = 8;
/// Integration dt for LM reservoir steps.
const LM_DT: f32 = 10.0;
/// Default training path (Project Gutenberg Shakespeare, eBook #100).
const LM_CORPUS_PATH: &str = "100.txt.utf-8";
/// Characters used for one online training pass (streaming prefix of corpus).
const LM_TRAIN_CHARS: usize = 400_000;
/// Held-out window immediately after the train prefix.
const LM_EVAL_CHARS: usize = 40_000;
/// Online passes over the training prefix.
const LM_EPOCHS: usize = 3;
/// Learning rate for softmax cross-entropy readout updates.
const LM_READOUT_LR: f32 = 0.06;
/// Generated sample length after training.
const LM_SAMPLE_LEN: usize = 400;

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
    /// exp(mean NLL) under a softmax interpretation of the linear scores.
    pub perplexity: f32,
}

/// Next-character LM: LIF reservoir encodes history; linear readout scores vocab.
pub struct LifLanguageModel {
    pub vocab: CharVocab,
    units: Vec<LifNeuron>,
    /// w_in[char_id][h] — column layout for one-hot input.
    w_in: Vec<Vec<f32>>,
    rec_idx: Vec<[usize; LM_REC_TAPS]>,
    rec_w: Vec<[f32; LM_REC_TAPS]>,
    /// w_out[char_id][h]
    w_out: Vec<Vec<f32>>,
    /// Direct bigram-ish skip: w_skip[next][prev]
    w_skip: Vec<Vec<f32>>,
    bias: Vec<f32>,
    prev_v: Vec<f32>,
    prev_id: usize,
    hidden: usize,
    lr: f32,
    step_count: u32,
    rng: Rand,
}

impl LifLanguageModel {
    pub fn new(vocab: CharVocab, seed: u32) -> Self {
        let v = vocab.len();
        let h = LM_HIDDEN;
        let mut rng = Rand::new(seed.max(1));
        let mut units = Vec::with_capacity(h);
        let taus = [4.0f32, 6.0, 8.0, 10.0, 14.0, 18.0, 24.0, 32.0];
        let in_scale = 1.15 / (v as f32).sqrt();

        for i in 0..h {
            let tau = taus[i % taus.len()];
            let mut n = LifNeuron::new(0.2 * rng.signed(), 2.2 + rng.u(), 0.0, tau);
            n.rng = Rand::new(seed.wrapping_mul(2246822519).wrapping_add(i as u32 + 3));
            n.resample_trial();
            units.push(n);
        }

        // Random input columns per character.
        let mut w_in = Vec::with_capacity(v);
        for _ in 0..v {
            let mut col = vec![0.0f32; h];
            for hh in 0..h {
                col[hh] = in_scale * rng.signed();
            }
            w_in.push(col);
        }

        // Sparse recurrence.
        let mut rec_idx = Vec::with_capacity(h);
        let mut rec_w = Vec::with_capacity(h);
        for _ in 0..h {
            let mut idx = [0usize; LM_REC_TAPS];
            let mut wts = [0.0f32; LM_REC_TAPS];
            for t in 0..LM_REC_TAPS {
                idx[t] = (rng.u32() as usize) % h;
                wts[t] = 0.45 * rng.signed();
            }
            let s: f32 = wts.iter().map(|w| w.abs()).sum::<f32>().max(1e-3);
            let scale = REC_RADIUS / s;
            for t in 0..LM_REC_TAPS {
                wts[t] *= scale;
            }
            rec_idx.push(idx);
            rec_w.push(wts);
        }

        let mut w_out = Vec::with_capacity(v);
        let mut w_skip = Vec::with_capacity(v);
        let mut bias = vec![0.0f32; v];
        for c in 0..v {
            w_out.push(vec![0.0f32; h]);
            w_skip.push(vec![0.0f32; v]);
            // Mild unigram prior: slightly prefer common-looking bytes later via data.
            bias[c] = 0.01 * rng.signed();
        }

        Self {
            vocab,
            units,
            w_in,
            rec_idx,
            rec_w,
            w_out,
            w_skip,
            bias,
            prev_v: vec![0.0; h],
            prev_id: 0,
            hidden: h,
            lr: LM_READOUT_LR,
            step_count: 0,
            rng,
        }
    }

    fn drive_hidden(&mut self, char_id: usize) {
        let col = &self.w_in[char_id];
        for h in 0..self.hidden {
            let mut drive = col[h];
            for t in 0..LM_REC_TAPS {
                let j = self.rec_idx[h][t];
                drive += self.rec_w[h][t] * self.prev_v[j];
            }
            let drive = drive.clamp(-FEATURE_CLIP, FEATURE_CLIP);
            let _ = self.units[h].step_with_target(drive, drive, LM_DT);
            self.prev_v[h] = self.units[h].v_membrane.clamp(-FEATURE_CLIP, FEATURE_CLIP);
        }
    }

    /// Linear scores over the vocabulary (not normalized).
    fn logits(&self, char_id: usize) -> Vec<f32> {
        let v = self.vocab.len();
        let mut scores = vec![0.0f32; v];
        // Reservoir residual is down-weighted so it refines the bigram prior
        // instead of drowning it early in training.
        const RES_SCALE: f32 = 0.35;
        for c in 0..v {
            let mut y = self.bias[c] + self.w_skip[c][char_id];
            let row = &self.w_out[c];
            for h in 0..self.hidden {
                y += RES_SCALE * row[h] * self.prev_v[h];
            }
            scores[c] = y;
        }
        scores
    }

    /// Softmax probabilities from logits (stable).
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let mut max = f32::NEG_INFINITY;
        for &z in logits {
            if z > max {
                max = z;
            }
        }
        let mut out = vec![0.0f32; logits.len()];
        let mut sum = 0.0f32;
        for (i, &z) in logits.iter().enumerate() {
            let e = (z - max).exp();
            out[i] = e;
            sum += e;
        }
        let inv = 1.0 / sum.max(1e-12);
        for p in &mut out {
            *p *= inv;
        }
        out
    }

    fn argmax(scores: &[f32]) -> usize {
        let mut best_i = 0;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &s) in scores.iter().enumerate() {
            if s > best_v {
                best_v = s;
                best_i = i;
            }
        }
        best_i
    }

    /// One supervised step: observe `char_id`, predict next, train toward `target_id`.
    /// Softmax cross-entropy on linear scores; returns (top1_hit, nll).
    pub fn learn_step(&mut self, char_id: usize, target_id: usize) -> (bool, f32) {
        self.drive_hidden(char_id);
        let scores = self.logits(char_id);
        let probs = Self::softmax(&scores);
        let hit = Self::argmax(&scores) == target_id;
        let nll = -probs[target_id].max(1e-12).ln();

        // Feature energy for step-size normalization (bias + skip + hidden).
        let mut energy = 2.0f32;
        for h in 0..self.hidden {
            energy += self.prev_v[h] * self.prev_v[h];
        }
        self.step_count = self.step_count.saturating_add(1);
        let lr = self.lr / (1.0 + 0.00005 * self.step_count as f32);
        let inv = 1.0 / energy.max(1e-3);

        // ∇_z CE = p − y  (y one-hot). Update readout with normalized CE gradient.
        // Match forward residual scale so reservoir weights get the correct ∂L/∂W.
        const RES_SCALE: f32 = 0.35;
        const LM_WCLIP: f32 = 8.0;
        let v = self.vocab.len();
        for c in 0..v {
            let grad = probs[c] - if c == target_id { 1.0 } else { 0.0 };
            let step = lr * grad * inv;
            self.bias[c] = (self.bias[c] - step).clamp(-LM_WCLIP, LM_WCLIP);
            self.w_skip[c][char_id] =
                (self.w_skip[c][char_id] - step).clamp(-LM_WCLIP, LM_WCLIP);
            for h in 0..self.hidden {
                let g = step * RES_SCALE * self.prev_v[h] + READOUT_L2 * self.w_out[c][h];
                self.w_out[c][h] = (self.w_out[c][h] - g).clamp(-LM_WCLIP, LM_WCLIP);
            }
        }

        self.prev_id = char_id;
        (hit, nll)
    }

    /// Forward without weight updates (eval / generation).
    pub fn observe(&mut self, char_id: usize) -> Vec<f32> {
        self.drive_hidden(char_id);
        let scores = self.logits(char_id);
        self.prev_id = char_id;
        scores
    }

    /// Seed `w_skip` from empirical bigram log-probs; bias gets unigram log-probs
    /// only as a fallback for rare contexts (scaled down so we do not double-count).
    pub fn warm_start_ngrams(&mut self, data: &[u8]) {
        let v = self.vocab.len();
        let mut uni = vec![1.0f32; v]; // Laplace
        let mut bi = vec![vec![1.0f32; v]; v];
        if data.len() < 2 {
            return;
        }
        for i in 0..data.len().saturating_sub(1) {
            let a = self.vocab.encode(data[i]);
            let b = self.vocab.encode(data[i + 1]);
            uni[b] += 1.0;
            bi[a][b] += 1.0;
        }
        let uni_tot: f32 = uni.iter().sum::<f32>().max(1.0);
        for c in 0..v {
            // Mild unigram prior (not a second full log-prob term).
            self.bias[c] = 0.15 * (uni[c] / uni_tot).ln();
            let row_tot: f32 = bi[c].iter().sum::<f32>().max(1.0);
            for n in 0..v {
                // w_skip[next][prev] = log P(next | prev)
                self.w_skip[n][c] = (bi[c][n] / row_tot).ln();
            }
        }
    }

    pub fn train_bytes(&mut self, data: &[u8], epochs: usize) -> LmTrainStats {
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
        self.warm_start_ngrams(data);
        for _ in 0..epochs.max(1) {
            for i in 0..data.len().saturating_sub(1) {
                let a = self.vocab.encode(data[i]);
                let b = self.vocab.encode(data[i + 1]);
                let (hit, nll) = self.learn_step(a, b);
                if hit {
                    hits += 1;
                }
                nll_sum += nll as f64;
                tokens += 1;
            }
        }
        let acc = if tokens == 0 {
            0.0
        } else {
            hits as f32 / tokens as f32
        };
        let mean_nll = if tokens == 0 {
            0.0
        } else {
            (nll_sum / tokens as f64) as f32
        };
        LmTrainStats {
            tokens: tokens as usize,
            loss: mean_nll,
            accuracy: acc,
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
        for i in 0..data.len().saturating_sub(1) {
            let a = self.vocab.encode(data[i]);
            let b = self.vocab.encode(data[i + 1]);
            let scores = self.observe(a);
            let probs = Self::softmax(&scores);
            if Self::argmax(&scores) == b {
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

    /// Sample `n` bytes continuing from `prompt` (greedy if temperature ≤ 0).
    pub fn generate(&mut self, prompt: &[u8], n: usize, temperature: f32) -> Vec<u8> {
        // Reset soft state but keep weights.
        for v in &mut self.prev_v {
            *v = 0.0;
        }
        for u in &mut self.units {
            u.v_membrane = u.trial_v_rest;
            u.is_refractory = false;
        }

        let mut out = prompt.to_vec();
        if out.is_empty() {
            out.push(b' ');
        }
        // Prime reservoir; scores after the last prompt byte predict the next char.
        let mut scores = Vec::new();
        for &b in &out {
            scores = self.observe(self.vocab.encode(b));
        }

        for _ in 0..n {
            let next = if temperature <= 1e-6 {
                Self::argmax(&scores)
            } else {
                // Temperature softmax sampling.
                let inv_t = 1.0 / temperature.max(1e-3);
                let mut scaled = scores.clone();
                for z in &mut scaled {
                    *z *= inv_t;
                }
                let probs = Self::softmax(&scaled);
                let mut r = self.rng.u();
                let mut pick = probs.len() - 1;
                for (i, &p) in probs.iter().enumerate() {
                    if r < p {
                        pick = i;
                        break;
                    }
                    r -= p;
                }
                pick
            };
            out.push(self.vocab.decode(next));
            scores = self.observe(next);
        }
        out
    }
}

/// Train + evaluate + sample from `100.txt.utf-8` (or a provided path).
fn run_language_model(path: &str) -> Result<(), String> {
    println!();
    println!("=== LIF character language model ===");
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
        "bytes: corpus={} train={} eval={} vocab={} hidden={} epochs={}",
        corpus.len(),
        train.len(),
        eval.len(),
        vocab.len(),
        LM_HIDDEN,
        LM_EPOCHS
    );

    let mut model = LifLanguageModel::new(vocab, 0xC0FFEE);
    // Bigram prior alone (reservoir weights still zero).
    model.warm_start_ngrams(train);
    let bigram_eval = model.evaluate_bytes(eval);
    println!(
        "bigram prior eval: acc={:.3}  nll={:.3}  ppl={:.2}",
        bigram_eval.accuracy, bigram_eval.loss, bigram_eval.perplexity
    );

    // Full train (re-warms ngrams, then CE over reservoir + readout).
    let train_stats = model.train_bytes(train, LM_EPOCHS);
    println!(
        "train: tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        train_stats.tokens, train_stats.accuracy, train_stats.loss, train_stats.perplexity
    );

    let eval_stats = model.evaluate_bytes(eval);
    println!(
        "eval (LIF+readout): tokens={}  acc={:.3}  nll={:.3}  ppl={:.2}",
        eval_stats.tokens, eval_stats.accuracy, eval_stats.loss, eval_stats.perplexity
    );

    let prompt = b"To be, or not to be";
    let sample_hot = model.generate(prompt, LM_SAMPLE_LEN, 0.7);
    let sample_greedy = {
        // Fresh continuation for greedy (re-prime inside generate).
        model.generate(prompt, LM_SAMPLE_LEN / 2, 0.0)
    };
    println!();
    println!("sample greedy (temp=0):");
    println!("----");
    println!("{}", String::from_utf8_lossy(&sample_greedy));
    println!("----");
    println!("sample (temp=0.7, prompt+{LM_SAMPLE_LEN} bytes):");
    println!("----");
    println!("{}", String::from_utf8_lossy(&sample_hot));
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
        assert!(p.stddev >= STD_MIN);
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
        // Repeating phrase should be learnable as a next-byte task.
        let text = b"to be or not to be or not to be or not to be or not ";
        let vocab = CharVocab::from_bytes(text);
        let mut model = LifLanguageModel::new(vocab, 42);
        let stats = model.train_bytes(text, 8);
        assert!(stats.tokens > 0);
        assert!(
            stats.accuracy > 0.35,
            "expected to beat unigram noise on tiny loop, acc={}",
            stats.accuracy
        );
        assert!(stats.perplexity.is_finite() && stats.perplexity > 1.0);

        let sample = model.generate(b"to be", 32, 0.5);
        assert!(sample.len() > 5);
        // Should stay within vocab bytes (valid UTF-8 for this ASCII corpus).
        assert!(std::str::from_utf8(&sample).is_ok());
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

    // Character LM on Shakespeare (eBook #100).
    if let Err(e) = run_language_model(LM_CORPUS_PATH) {
        eprintln!("language model error: {e}");
        std::process::exit(1);
    }
}
