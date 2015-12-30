# CHIME-x
OpenCL Kernels for 4-bit Cross Correlations for Astrophysics

Usage: sudo ./correlator_test [opts]

Options:
  --help (-h)                               Display the available run options.
  --device (-d) [device_number]             Default: 0. For multi-GPU computers, can choose larger values.
  --iterations (-i) [number]                Default: 100. Number of iterations to test the code.
  --time_accum (-t) [number]                Default: 256. Range [1,291] for version from conference, [1,1912] for new alg.
  --time_steps (-T) [number]                Default: Automatically generated. Number of time steps of element data.
  --num_freq (-f) [number]                  Default: 1. Number of frequency channels to process simultaneously.
  --num_elements (-e) [number]              Default: 2048. Number of elements to correlate.
  --timer_without_copies (-w) [number]      Default: off. When on, it separates the copy section from the iterations. 
                                                     Not realistic behaviour, but helpful for timing without using profiler tools.
  --upper_triangle_convention (-U) [number] Default: 1. (range: [0,1]). 1 uses the standard pairwise correlation convention. 0 does not (i.e. complex conjugate of expected results).
  --check_results (-c)                      Default: off. Calculates and checks GPU results with CPU calculations.
  --verbose (-v)                            Default: off. Verbose calculation check. (Dumps all correlation products).
  --gen_type (-g) [number]                  Default: 4. (1 = Constant, 2 = Ramp up, 3 = Ramp down, 4 = Random (seeded)).
  --random_seed (-r) [number]               Default: 42. The seed for the pseudorandom generator.
  --no_repeat_random (-p)                   Default: off. Whether the random sequence repeats at each time step (and frequency channel).
  --generate_frequency -q [number]          Default: -1 (All frequencies). Other numbers generate non-default values for that frequency channel.
  --default_real (-x) [number]              Default: 0. (range: [-8, 7]). Only used for higher frequency channels when generate_freq != ALL_FREQUENCIES.
  --default_imaginary (-y) [number]         Default: 0. (range: [-8, 7]). Only used for higher frequency channels when generate_freq != ALL_FREQUENCIES.
  --initial_real (-X) [number]              Default: 0. (range: [-8, 7]). Only matters for ramped modes.
  --initial_imaginary (-Y) [number]         Default: 0. (range: [-8, 7]). Only matters for ramped modes.
  --kernel_batch (-k) [number]              Default: 0. (0= Kernels from IEEE conference, 1= New more-packed version).

