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
const POP_SIZE: usize = 8;
/// Wall-clock steps per episode with fixed sampled parameters
/// (includes refractory steps so generation rate stays predictable).
const EPISODE_LEN: usize = 4;
/// Number of top episodes used for the distribution update.
const ELITE_COUNT: usize = 4;

/// Soft-update rates toward elite statistics (0 = freeze, 1 = hard replace).
const LR_MEAN: f32 = 0.65;
const LR_STD: f32 = 0.45;
/// Keep exploration alive; clamp explosion.
const STD_MIN: f32 = 0.05;
const STD_MAX: f32 = 2.0;
/// Extra cost when a spike resets the membrane during tracking.
const SPIKE_PENALTY: f32 = 1.5;
/// Mild preference for small |v_rest| (steady-state V ≈ v_rest + I tracks I best at 0).
const V_REST_L2: f32 = 0.01;

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

    /// Search distribution over resting potential.
    pub v_rest_dist: GaussianParam,
    /// Search distribution over spike threshold. Also supplies the scale for
    /// per-step stochastic (escape-noise) thresholding: each step draws
    /// thr ~ N(trial_v_threshold, v_threshold_dist.stddev) and spikes if V ≥ thr.
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

    fn record_step(&mut self, i_input: f32) {
        let err = i_input - self.v_membrane;
        self.episode_error_sum += err * err;
        self.episode_step += 1;
        self.input.push(i_input);
        self.output.push(self.v_membrane);

        if self.episode_step >= EPISODE_LEN as u32 {
            self.finish_episode();
        }
    }

    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
        // Refractory: hold at reset, still score tracking error so spikes that
        // wreck the trajectory are charged to the active trial parameters.
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            self.record_step(i_input);
            return false;
        }

        let v_rest = self.trial_v_rest;

        // Euler: dv = (-(v - v_rest) + I) * (dt / tau_m)
        let dv = (-(self.v_membrane - v_rest) + i_input) * (dt / self.tau_m);
        self.v_membrane += dv;

        // Probabilistic firing via escape noise: sample threshold from the
        // episode law N(trial_v_threshold, v_threshold_dist.stddev). Equivalent
        // to spiking with P = Φ((V − μ) / σ) under that Gaussian. CEM still
        // attributes the episode to trial_v_threshold (μ); σ is shared and
        // shrinks as the search distribution concentrates.
        let thr = self.sample_threshold();
        let spiked = self.v_membrane >= thr;
        if spiked {
            self.is_refractory = true;
            self.episode_spike_count += 1;
        }

        self.record_step(i_input);
        spiked
    }

    /// Draw one threshold sample from N(trial_v_threshold, v_threshold_dist.stddev).
    fn sample_threshold(&mut self) -> f32 {
        let (z, _) = self.rng.g();
        let thr = self.trial_v_threshold + z * self.v_threshold_dist.stddev;
        thr.max(self.v_reset + 0.05)
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
        // With V sitting on the trial mean and non-zero σ, some steps spike and
        // some do not (escape-noise threshold).
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 1.0e9); // huge τ → V ≈ fixed
        neuron.trial_v_rest = 1.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.5);
        neuron.v_membrane = 1.0;
        neuron.pending_antithetic = None;
        // Avoid finishing episodes / resampling mid-test.
        neuron.episode_step = 0;

        let mut spikes = 0u32;
        let trials = 200u32;
        for _ in 0..trials {
            // Hold membrane at the threshold mean; leak is negligible (large τ).
            neuron.v_membrane = 1.0;
            neuron.is_refractory = false;
            if neuron.step(0.0, 1e-6) {
                spikes += 1;
            }
            // Don't let episode/CEM machinery reshape params.
            neuron.episode_step = 0;
            neuron.episode_error_sum = 0.0;
            neuron.episode_spike_count = 0;
        }
        // Φ(0) = 0.5 under a symmetric Gaussian threshold; allow Monte Carlo slack.
        assert!(
            spikes > 30 && spikes < 170,
            "expected mixed spikes near threshold, got {spikes}/{trials}"
        );
    }

    #[test]
    fn test_firing_certain_when_far_above_threshold() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        neuron.trial_v_rest = 0.0;
        neuron.trial_v_threshold = 1.0;
        neuron.v_threshold_dist = GaussianParam::new(1.0, 0.3);
        neuron.pending_antithetic = None;

        for _ in 0..50 {
            neuron.v_membrane = 5.0;
            neuron.is_refractory = false;
            neuron.episode_step = 0;
            assert!(
                neuron.step(0.0, 1.0),
                "V far above μ should almost surely spike"
            );
        }
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
}

fn main() {
    // rest=2 (bad init so learning is visible), threshold=2.5 (above early V so
    // non-spiking policies are reachable), reset=0, tau=10ms
    let mut neuron = LifNeuron::new(2.0, 2.5, 0.0, 10.0);

    let dt = 10.0;
    let total_steps = 256;
    let mut injected_current = 1.0;

    println!(
        "CEM-LIF: pop={POP_SIZE} episode={EPISODE_LEN} elite={ELITE_COUNT} lr_mean={LR_MEAN}"
    );
    println!(
        "init: v_rest={:.3}±{:.3}  v_th={:.3}±{:.3}",
        neuron.v_rest(),
        neuron.v_rest_stddev(),
        neuron.v_threshold(),
        neuron.v_threshold_stddev()
    );
    println!("Time | V_m     | params (rest/th)     | Action");
    println!("----------------------------------------------------");

    let mut score = 0.0f32;
    for step in 1..=total_steps {
        let spiked = neuron.step(injected_current, dt);
        let visual_bar = "*".repeat((neuron.v_membrane.max(0.0) * 12.0) as usize);

        if spiked {
            println!(
                "{:>-4} | {:>-7.2} | r={:>5.2} th={:>5.2} | SPIKE",
                step,
                neuron.v_membrane,
                neuron.trial_v_rest,
                neuron.trial_v_threshold
            );
        } else {
            println!(
                "{:>-4} | {:>-7.2} | r={:>5.2} th={:>5.2} | {}",
                step,
                neuron.v_membrane,
                neuron.trial_v_rest,
                neuron.trial_v_threshold,
                visual_bar
            );
        }

        injected_current = if injected_current == 1.0 { 0.0 } else { 1.0 };
        if let (Some(&i), Some(&o)) = (neuron.input.latest(), neuron.output.latest()) {
            score += (i - o).abs();
        }
    }

    println!("----------------------------------------------------");
    println!(
        "final: v_rest={:.3}±{:.3}  v_th={:.3}±{:.3}  generations={}  last_gen_fit={:.4}",
        neuron.v_rest(),
        neuron.v_rest_stddev(),
        neuron.v_threshold(),
        neuron.v_threshold_stddev(),
        neuron.generation,
        neuron.last_gen_fitness
    );
    println!("{{score: {:?}}}", score);
}
