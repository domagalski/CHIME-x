//definitions for use with the generator
#ifndef INPUT_GENERATOR_H
#define INPUT_GENERATOR_H

#define GENERATE_DATASET_CONSTANT       1u
#define GENERATE_DATASET_RAMP_UP        2u
#define GENERATE_DATASET_RAMP_DOWN      3u
#define GENERATE_DATASET_RANDOM_SEEDED  4u
#define ALL_FREQUENCIES                -1

int offset_and_clip_value(int input_value, int offset_value, int min_val, int max_val);

void generate_char_data_set(int generation_Type,
                            int random_seed,
                            int default_real,
                            int default_imaginary,
                            int initial_real,
                            int initial_imaginary,
                            int single_frequency,
                            int num_timesteps,
                            int num_frequencies,
                            int num_elements,
                            int no_repeat_random,
                            unsigned char *packed_data_set);

#endif
