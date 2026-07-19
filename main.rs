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
        // Initialize an empty array using const expressions
        Self {
            buffer: [const { None }; N],
            write_idx: 0,
            read_idx: 0,
            size: 0,
        }
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
}

const LENGTH:usize = 8;

// Define the Leaky Integrate-and-Fire neuron
pub struct LifNeuron {
    pub v_membrane: f32,    // Current membrane potential (mV)
    pub v_rest: f32,        // Resting membrane potential (mV)
    pub v_rest_stddev: f32,
    pub v_rest_buffer: RingBuffer::<f32, LENGTH>,
    pub v_threshold: f32,   // Spike generation threshold (mV)
    pub v_threshold_stddev: f32,
    pub v_threshold_buffer: RingBuffer::<f32, LENGTH>,
    pub v_reset: f32,       // Potential after a spike occurs (mV)
    pub tau_m: f32,         // Membrane time constant (ms)
    pub is_refractory: bool,// Track if the neuron is in a refractory state
    pub input: RingBuffer::<f32, LENGTH>,
    pub output: RingBuffer::<f32, LENGTH>,
    pub fitness: RingBuffer::<f32, LENGTH>,
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

    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
    	self.input.push(i_input);
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            return false;
        }

		let (z0, _) = self.rng.g();
		let v_rest = z0*self.v_rest_stddev + self.v_rest;
		self.v_rest_buffer.push(v_rest);

        // Euler method integration: dv = (-(v - v_rest) + I) * (dt / tau_m)
        let dv = (-(self.v_membrane - v_rest) + i_input) * (dt / self.tau_m);
        self.v_membrane += dv;

		self.output.push(self.v_membrane);
		let mut diff = self.input.buffer[(LENGTH-2+LENGTH)%LENGTH].unwrap_or(0.0) - self.output.buffer[(LENGTH-1+LENGTH)%LENGTH].unwrap_or(0.0);
		if diff < 0.0 {
			diff = -diff;
		}
		self.fitness.push(diff);

		let (z0, _) = self.rng.g();
		let v_threshold = z0*self.v_threshold_stddev + self.v_threshold;
		self.v_threshold_buffer.push(v_threshold);

		self.iteration += 1;
		if self.iteration == LENGTH as u64 {
			self.iteration = 0;
			let mut swapped = true;    
			while swapped {
				swapped = false;
				// Loop through the slice, stopping at the second-to-last element
				for i in 0..self.fitness.buffer.len().saturating_sub(1) {
					if self.fitness.buffer[i] < self.fitness.buffer[i + 1] {
						self.fitness.buffer.swap(i, i + 1);
						self.v_rest_buffer.buffer.swap(i, i + 1);
						self.v_threshold_buffer.buffer.swap(i, i+1);
						swapped = true; // Set to true to trigger another pass
					}
				}
			}
			{
				let length = self.v_rest_buffer.buffer.len()/2;
				let mut avg = 0.0;
				for i in 0..length {
					avg += self.v_rest_buffer.buffer[i].unwrap_or(0.0);
				}
				avg /= length as f32;
				let mut stddev = 0.0;
				for i in 0..length {
					let diff = self.v_rest_buffer.buffer[i].unwrap_or(0.0) - avg;
					stddev += diff*diff;
				}
				stddev /= length as f32;
				stddev = stddev.sqrt();
				self.v_rest = avg;
				if stddev < 0.01 {
					stddev = 0.01;
				}
				self.v_rest_stddev = stddev;
			}
			{
				let length = self.v_threshold_buffer.buffer.len()/2;
				let mut avg = 0.0;
				for i in 0..length {
					avg += self.v_threshold_buffer.buffer[i].unwrap_or(0.0);
				}
				avg /= length as f32;
				let mut stddev = 0.0;
				for i in 0..length {
					let diff = self.v_threshold_buffer.buffer[i].unwrap_or(0.0) - avg;
					stddev += diff*diff;
				}
				stddev /= length as f32;
				stddev = stddev.sqrt();
				self.v_threshold = avg;
				if stddev < 0.01 {
					stddev = 0.01;
				}
				self.v_threshold_stddev = stddev;
			}
		}

        if self.v_membrane >= v_threshold {
            self.is_refractory = true;
            true // Spike emitted!
        } else {
            false
        }
    }
}

// Rand is a random number generator
pub struct Rand {
	pub lfsr: u32,
}

// LFSRMASK is the lfsr polynomial
const LFSRMASK:u32 = 0x80000057;

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
		let theta = 2.0 * 3.1459 * u2;
		let z0 = r * theta.cos();
		let z1 = r * theta.sin();
		(z0, z1)
	}
}

#[cfg(test)]
mod tests {
    use super::*; // Brings the outer functions into scope

    #[test]
    fn test_lfsr() {
    	let mut lfsr = Rand::new(1);
		let mut count:u64 = 1;
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
		const LENGTH:usize = 8*1024;
		let mut lfsr = Rand::new(1);
		let mut za:[f32; LENGTH] = [0.0; LENGTH];
		let mut zb:[f32; LENGTH] = [0.0; LENGTH];
		for step in 0..LENGTH {
			let (z0, z1) = lfsr.g();
			za[step] = z0;
			zb[step] = z1;
		}
		{
			let mut avg = 0.0;
			for value in za {
				avg += value;
			}
			avg /= LENGTH as f32;
			let mut stddev = 0.0;
			for value in za {
				let diff = value - avg;
				stddev += diff*diff;
			}
			stddev /= LENGTH as f32;
			stddev = stddev.sqrt();
			assert_eq!(avg.round(), 0.0);
			assert_eq!(stddev.round(), 1.0);
		}
		{
			let mut avg = 0.0;
			for value in zb {
				avg += value;
			}
			avg /= LENGTH as f32;
			let mut stddev = 0.0;
			for value in zb {
				let diff = value - avg;
				stddev += diff*diff;
			}
			stddev /= LENGTH as f32;
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
    let dt = 10.0;            // 1 millisecond per timestep
    let total_steps = 128;    // Simulate for 30 milliseconds
    let mut injected_current = 1.0; // Steady current injected every step

    println!("Simulating 30ms with constant current injection of {} mA:", injected_current);
    println!("Time(ms) | Voltage(mV) | Action");
    println!("---------------------------------");

    // 2. Loop through time steps
    let mut score:f32 = 0.0;
    for step in 1..=total_steps {
        // Feed current into the neuron step function
        let spiked = neuron.step(injected_current, dt);
        
        // Format a small text-based horizontal bar to visualize voltage
        let visual_bar = "*".repeat((neuron.v_membrane.max(0.0) * 15.0) as usize);

        if spiked {
            println!("{:>-8} | {:>-11.2} | SPIKE! ⚡", step, neuron.v_threshold_stddev);
        } else {
            println!("{:>-8} | {:>-11.2} | {}", step, neuron.v_threshold_stddev, visual_bar);
        }
        if injected_current == 1.0 {
        	injected_current = 0.0;
        } else {
        	injected_current = 1.0;
        }
        for i in 0..LENGTH {
        	score += neuron.fitness.buffer[i].unwrap_or(0.0);
        }
        //println!("{:?}", neuron.input.buffer);
        //println!("{:?}", neuron.output.buffer);
    }
    println!("{{score: {:?}}}", score)
}
