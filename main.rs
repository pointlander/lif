pub struct RingBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    write_idx: usize,
    read_idx: usize,
    size: usize,
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

// Define the Leaky Integrate-and-Fire neuron
pub struct LifNeuron {
    pub v_membrane: f32,    // Current membrane potential (mV)
    pub v_rest: f32,        // Resting membrane potential (mV)
    pub v_rest_stddev: f32,
    pub v_threshold: f32,   // Spike generation threshold (mV)
    pub v_threshold_stddev: f32,
    pub v_reset: f32,       // Potential after a spike occurs (mV)
    pub tau_m: f32,         // Membrane time constant (ms)
    pub is_refractory: bool,// Track if the neuron is in a refractory state
}

impl LifNeuron {
    pub fn new(v_rest: f32, v_threshold: f32, v_reset: f32, tau_m: f32) -> Self {
        Self {
            v_membrane: v_rest,
            v_rest,
            v_rest_stddev: 0.1,
            v_threshold,
            v_threshold_stddev: 0.1,
            v_reset,
            tau_m,
            is_refractory: false,
        }
    }

    pub fn step(&mut self, i_input: f32, dt: f32) -> bool {
        if self.is_refractory {
            self.is_refractory = false;
            self.v_membrane = self.v_reset;
            return false;
        }

        // Euler method integration: dv = (-(v - v_rest) + I) * (dt / tau_m)
        let dv = (-(self.v_membrane - self.v_rest) + i_input) * (dt / self.tau_m);
        self.v_membrane += dv;

        if self.v_membrane >= self.v_threshold {
            self.is_refractory = true;
            true // Spike emitted!
        } else {
            false
        }
    }
}

fn main() {
    // 1. Initialize neuron: rest=0mV, threshold=1.0mV, reset=0mV, tau_m=10.0ms
    let mut neuron = LifNeuron::new(0.0, 1.0, 0.0, 10.0);
    const LENGTH:usize = 8;
    let mut input = RingBuffer::<f32, LENGTH>::new();
    let mut output = RingBuffer::<f32, LENGTH>::new();
    let mut cost = RingBuffer::<f32, LENGTH>::new();
    
    // Simulation parameters
    let dt = 1.0;            // 1 millisecond per timestep
    let total_steps = 30;    // Simulate for 30 milliseconds
    let injected_current = 0.35; // Steady current injected every step

    println!("Simulating 30ms with constant current injection of {} mA:", injected_current);
    println!("Time(ms) | Voltage(mV) | Action");
    println!("---------------------------------");

    // 2. Loop through time steps
    for step in 1..=total_steps {
    	input.push(injected_current);
        // Feed current into the neuron step function
        let spiked = neuron.step(injected_current, dt);
        
        // Format a small text-based horizontal bar to visualize voltage
        let visual_bar = "*".repeat((neuron.v_membrane.max(0.0) * 15.0) as usize);

        if spiked {
            println!("{:>-8} | {:>-11.2} | SPIKE! ⚡", step, neuron.v_membrane);
        } else {
            println!("{:>-8} | {:>-11.2} | {}", step, neuron.v_membrane, visual_bar);
        }
        output.push(neuron.v_membrane);
        let mut diff = input.buffer[(LENGTH-2+LENGTH)%LENGTH].unwrap_or(0.0) - input.buffer[(LENGTH-1+LENGTH)%LENGTH].unwrap_or(0.0);
        if diff < 0.0 {
        	diff = -diff;
        }
        cost.push(diff);
        println!("{:?}", input.buffer);
        println!("{:?}", output.buffer);
    }

    const LFSRMASK:u32 = 0x80000057;
    let mut lfsr:u32 = 1;
    let mut count:u64 = 0;
    loop {
    	lfsr = (lfsr >> 1) ^ ((!(lfsr & 1)).wrapping_add(1) & LFSRMASK);
    	if lfsr == 1 {
    		break;
    	}
    	count += 1;
    }
    println!("{:?} {:?}", count, u32::MAX);

	lfsr = 1;
	let mut z:[f64; 256] = [0.0; 256];
	for step in 0..128{
    	lfsr = (lfsr >> 1) ^ ((!(lfsr & 1)).wrapping_add(1) & LFSRMASK);
    	println!("{:?}", lfsr);
    	let u1 = lfsr as f64 / 4294967295.0;
    	lfsr = (lfsr >> 1) ^ ((!(lfsr & 1)).wrapping_add(1) & LFSRMASK);
    	println!("{:?}", lfsr);
    	let u2 = lfsr as f64 / 4294967295.0;
    	println!("{:.10} {:.10}", u1, u2);

    	let r = (-2.0 * u1.ln()).sqrt();
    	let theta = 2.0 * 3.1459 * u2;
    	let z0 = r * theta.cos();
    	let z1 = r * theta.sin();
    	println!("{:.10} {:.10}", z0, z1);
    	z[step*2] = z0;
    	z[step*2+1] = z1;
    }

    let mut avg = 0.0;
    for value in z {
    	avg += value;
    }
    avg = avg/256.0;
    println!("{:.10}", avg);
    let mut stddev = 0.0;
	for value in z {
		let diff = value - avg;
		stddev += diff*diff;
	}
	stddev = stddev/256.0;
	stddev = stddev.sqrt();
	println!("{:10}", stddev);
}
