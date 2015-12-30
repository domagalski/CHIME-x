//cpu_corr_test.h
//functions to test results
#ifndef CPU_CORR_TEST_H
#define CPU_CORR_TEST_H
#include <stdlib.h>

int cpu_data_generate_and_correlate(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose);

int cpu_data_generate_and_correlate_upper_triangle_only(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_triangle, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose);

int cpu_data_generate_and_correlate_nonstandard_convention(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose);

int cpu_data_generate_and_correlate_upper_triangle_only_nonstandard_convention(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_triangle, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose);

void compare_NSquared_correlator_results ( int *num_err, int64_t *err_2, int num_frequencies, int num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity);

void compare_NSquared_correlator_results_data_has_upper_triangle_only ( int *num_err, int64_t *err_2, int actual_num_frequencies, int actual_num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity);

#endif
