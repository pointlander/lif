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

const LENGTH: usize = 8;

// Define the Leaky Integrate-and-Fire neuron
pub struct LifNeuron {
    pub v_membrane: f32, // Current membrane potential (mV)
    pub v_rest: f32,     // Resting membrane potential (mV)
    pub v_rest_stddev: f32,
    pub v_rest_buffer: RingBuffer<f32, LENGTH>,
    pub v_threshold: f32, // Spike generation threshold (mV)
    pub v_threshold_stddev: f32,
    pub v_threshold_buffer: RingBuffer<f32, LENGTH>,
    pub v_reset: f32, // Potential after a spike occurs (mV)
    pub tau_m: f32,   // Membrane time constant (ms)
    pub is_refractory: bool, // Track if the neuron is in a refractory state
    pub input: RingBuffer<f32, LENGTH>,
    pub output: RingBuffer<f32, LENGTH>,
    pub fitness: RingBuffer<f32, LENGTH>,
    pub iteration: u64,
    pub rng: Rand,
}

impl LifNeuron {
    pub fn new(v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32) -> Self {
        Self {
            v_membrane: v_rest,
            v_rest,
            v_rest_stddev: 0.1,
            v_rest_buffer: RingBuffer::<f32, LENGTH>::new(),
            v_threshold,
            v_threshold_stddev: 0.1,
            v_threshold_buffer: RingBuffer::<f32, LENGTH>::new(),
            v_reset,
            tau_m,
            is_refractory: false,
            input: RingBuffer::<f32, LENGTH>::new(),
            output: RingBuffer::<f32, LENGTH>::new(),
            fitness: RingBuffer::<f32, LENGTH>::new(),
            iteration: 0,
            rng: Rand::new(1),
        }
    }

    /// Fitness = absolute tracking error between input and membrane voltage.
    /// Uses same-timestep samples so parallel buffers stay aligned.
    fn tracking_error(input: f32, output: f32) -> f32 {
        (input - output).abs()
    }

    /// Select the best half of recent trials (lowest error) and update
    /// `v_rest` / `v_threshold` mean + stddev. Leaves ring order intact.
    fn adapt_from_elites(&mut self) {
        let n = self.fitness.len();
        if n == 0 {
            return;
        }

        // Indices into chronological order (0 = oldest).
        let mut order = [0usize; LENGTH];
        for i in 0..n {
            order[i] = i;
        }

        // Bubble sort ascending by fitness (lower error is better).
        let mut swapped = true;
        while swapped {
            swapped = false;
            for i in 0..n.saturating_sub(1) {
                let fa = *self.fitness.get(order[i]).unwrap_or(&f32::INFINITY);
                let fb = *self.fitness.get(order[i + 1]).unwrap_or(&f32::INFINITY);
                if fa > fb {
                    order.swap(i, i + 1);
                    swapped = true;
                }
            }
        }

        let elite_n = (n / 2).max(1);
        let elite = &order[..elite_n];
        let (rest_mean, rest_std) = mean_stddev_from_elites(&self.v_rest_buffer, elite);
        let (th_mean, th_std) = mean_stddev_from_elites(&self.v_threshold_buffer, elite);
        self.v_rest = rest_mean;
        self.v_rest_stddev = rest_std.max(0.01);
        self.v_threshold = th_mean;
        self.v_threshold_stddev = th_std.max(0.01);
    }

    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
        // Refractory: reset membrane only. Do not touch history buffers so
        // input / output / fitness / param rings stay the same length.
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            return false;
        }

        let (z0, _) = self.rng.g();
        let v_rest = z0 * self.v_rest_stddev + self.v_rest;

        // Euler method integration: dv = (-(v - v_rest) + I) * (dt / tau_m)
        let dv = (-(self.v_membrane - v_rest) + i_input) * (dt / self.tau_m);
        self.v_membrane += dv;

        let (z0, _) = self.rng.g();
        let v_threshold = z0 * self.v_threshold_stddev + self.v_threshold;

        // Record one aligned trial: params used this step + tracking error.
        self.input.push(i_input);
        self.output.push(self.v_membrane);
        self.v_rest_buffer.push(v_rest);
        self.v_threshold_buffer.push(v_threshold);
        self.fitness
            .push(Self::tracking_error(i_input, self.v_membrane));

        self.iteration += 1;
        if self.iteration == LENGTH as u64 {
            self.iteration = 0;
            self.adapt_from_elites();
        }

        if self.v_membrane >= v_threshold {
            self.is_refractory = true;
            true // Spike emitted!
        } else {
            false
        }
    }
}

fn mean_stddev_from_elites(
    buffer: &RingBuffer<f32, LENGTH>,
    elite_chrono_indices: &[usize],
) -> (f32, f32) {
    let n = elite_chrono_indices.len();
    if n == 0 {
        return (0.0, 0.01);
    }

    let mut avg = 0.0;
    for &i in elite_chrono_indices {
        avg += *buffer.get(i).unwrap_or(&0.0);
    }
    avg /= n as f32;

    let mut var = 0.0;
    for &i in elite_chrono_indices {
        let diff = *buffer.get(i).unwrap_or(&0.0) - avg;
        var += diff * diff;
    }
    var /= n as f32;
    (avg, var.sqrt())
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
        let u1 = self.u();
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
        assert_eq!(rb.get(0), Some(&10)); // oldest
        assert_eq!(rb.get(2), Some(&30)); // newest
        assert_eq!(rb.latest(), Some(&30));
        assert_eq!(rb.get_newest(0), Some(&30));
        assert_eq!(rb.get_newest(1), Some(&20));
        assert_eq!(rb.get_newest(2), Some(&10));

        rb.push(40);
        rb.push(50); // overwrites 10
        assert!(rb.is_full());
        assert_eq!(rb.len(), 4);
        assert_eq!(rb.get(0), Some(&20)); // oldest now
        assert_eq!(rb.get(1), Some(&30));
        assert_eq!(rb.get(2), Some(&40));
        assert_eq!(rb.get(3), Some(&50)); // newest
        assert_eq!(rb.latest(), Some(&50));

        assert_eq!(rb.pop(), Some(20));
        assert_eq!(rb.get(0), Some(&30));
        assert_eq!(rb.latest(), Some(&50));
    }

    #[test]
    fn test_ring_buffer_copy_chronological() {
        let mut rb = RingBuffer::<f32, 4>::new();
        for v in [1.0, 2.0, 3.0] {
            rb.push(v);
        }
        let mut dst = [0.0; 4];
        let n = rb.copy_chronological(&mut dst);
        assert_eq!(n, 3);
        assert_eq!(&dst[..3], &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parallel_buffers_stay_aligned() {
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        // Force a spike then a refractory step.
        neuron.v_membrane = 10.0;
        neuron.v_threshold = 1.0;
        neuron.v_threshold_stddev = 0.0;
        neuron.v_rest_stddev = 0.0;

        let spiked = neuron.step(0.0, 1.0);
        assert!(spiked);
        assert_eq!(neuron.input.len(), 1);
        assert_eq!(neuron.output.len(), 1);
        assert_eq!(neuron.fitness.len(), 1);
        assert_eq!(neuron.v_rest_buffer.len(), 1);
        assert_eq!(neuron.v_threshold_buffer.len(), 1);

        // Refractory step must not desync histories.
        let spiked = neuron.step(1.0, 1.0);
        assert!(!spiked);
        assert_eq!(neuron.input.len(), 1);
        assert_eq!(neuron.output.len(), 1);
        assert_eq!(neuron.fitness.len(), 1);
    }

    #[test]
    fn test_fitness_is_abs_tracking_error() {
        let mut neuron = LifNeuron::new(0.0, 100.0, 0.0, 10.0);
        neuron.v_rest_stddev = 0.0;
        neuron.v_threshold_stddev = 0.0;

        neuron.step(1.0, 10.0); // dt/tau = 1 → v becomes ~1.0 from rest 0 with I=1
        let err = *neuron.fitness.latest().unwrap();
        let input = *neuron.input.latest().unwrap();
        let output = *neuron.output.latest().unwrap();
        assert!((err - (input - output).abs()).abs() < 1e-5);
    }

    #[test]
    fn test_adapt_prefers_lower_error() {
        // Manually fill elite selection inputs.
        let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
        // Four trials: two good (rest near 0), two bad (rest far).
        let rests = [0.0, 0.1, 5.0, 5.1];
        let thresholds = [1.0, 1.1, 9.0, 9.1];
        let fitness = [0.1, 0.2, 4.0, 5.0];
        for i in 0..4 {
            neuron.v_rest_buffer.push(rests[i]);
            neuron.v_threshold_buffer.push(thresholds[i]);
            neuron.fitness.push(fitness[i]);
        }
        neuron.adapt_from_elites();
        // Elite half is the two lowest errors → rests 0.0 and 0.1
        assert!((neuron.v_rest - 0.05).abs() < 1e-5);
        assert!((neuron.v_threshold - 1.05).abs() < 1e-5);
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
            assert_eq!(avg.round(), 0.0);
            assert_eq!(stddev.round(), 1.0);
        }
    }
}

fn main() {
    // 1. Initialize neuron: rest=0mV, threshold=1.0mV, reset=0mV, tau_m=10.0ms
    let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);

    // Simulation parameters
    let dt = 10.0; // 1 millisecond per timestep
    let total_steps = 128; // Simulate for 30 milliseconds
    let mut injected_current = 1.0; // Steady current injected every step

    println!(
        "Simulating 30ms with constant current injection of {} mA:",
        injected_current
    );
    println!("Time(ms) | Voltage(mV) | Action");
    println!("---------------------------------");

    // 2. Loop through time steps
    let mut score: f32 = 0.0;
    for step in 1..=total_steps {
        let spiked = neuron.step(injected_current, dt);

        let visual_bar = "*".repeat((neuron.v_membrane.max(0.0) * 15.0) as usize);

        if spiked {
            println!("{:>-8} | {:>-11.2} | SPIKE! ⚡", step, neuron.v_membrane);
        } else {
            println!("{:>-8} | {:>-11.2} | {}", step, neuron.v_membrane, visual_bar);
        }
        if injected_current == 1.0 {
            injected_current = 0.0;
        } else {
            injected_current = 1.0;
        }
        // Sum current fitness window once per step (not raw slots).
        score += neuron.fitness.sum();
    }
    println!("{{score: {:?}}}", score)
}
