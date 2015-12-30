// cpu_corr_test.c
// contains functions to correlate data in straight-forward manner on the cpu for comparison with GPU results
#include "cpu_corr_test.h"
#include <stdio.h> // printf
#include <stdlib.h> // malloc, etc.
#include <math.h>
#include "four_bit_macros.h"
#include "input_generator.h"
#include "gpu_cpu_helpers.h"

int cpu_data_generate_and_correlate_nonstandard_convention(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(gen_type,default_seed,default_real,default_imaginary,initial_real,initial_imaginary,generate_frequency, num_timesteps, num_frequencies, num_elements, no_repeat_random, generated);

    if (verbose){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = 0; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
                    int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                    int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    if (k != 0){
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   += element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] += element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                    else{
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   = element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] = element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}

int cpu_data_generate_and_correlate_upper_triangle_only_nonstandard_convention(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_triangle, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(gen_type,default_seed,default_real,default_imaginary,initial_real,initial_imaginary,generate_frequency, num_timesteps, num_frequencies, num_elements, no_repeat_random, generated);

    if (verbose){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        int output_counter = 0;
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = element_y; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
                    int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                    int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    if (k != 0){
                        correlated_data_triangle[output_counter++] += element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data_triangle[output_counter++] += element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                    else{
                        correlated_data_triangle[output_counter++] = element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data_triangle[output_counter++] = element_x_im*element_y_re - element_x_re*element_y_im;
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}

int cpu_data_generate_and_correlate(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(gen_type,default_seed,default_real,default_imaginary,initial_real,initial_imaginary,generate_frequency, num_timesteps, num_frequencies, num_elements, no_repeat_random, generated);

    if (verbose){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = 0; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
                    int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                    int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    if (k != 0){
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   += element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] += element_x_re*element_y_im - element_x_im*element_y_re ;
                    }
                    else{
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2]   = element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data[(j*num_elements*num_elements+element_y*num_elements+element_x)*2+1] = element_x_re*element_y_im - element_x_im*element_y_re;
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}

int cpu_data_generate_and_correlate_upper_triangle_only(int num_timesteps, int num_frequencies, int num_elements, int *correlated_data_triangle, int gen_type, int default_seed, int default_real, int default_imaginary, int initial_real, int initial_imaginary, int generate_frequency, int no_repeat_random, int verbose){
    //correlatedData will be returned as num_frequencies blocks, each num_elements x num_elements x 2

    //generate a dataset that should be the same as what the gpu is testing
    //dataset will be num_timesteps x num_frequencies x num_elements large
    unsigned char *generated = (unsigned char *)malloc(num_timesteps*num_frequencies*num_elements*sizeof(unsigned char));
    //check the array was allocated properly
    if (generated == NULL){
        printf ("Error allocating memory: cpu_data_generate_and_correlate\n");
        return (-1);
    }

    generate_char_data_set(gen_type,default_seed,default_real,default_imaginary,initial_real,initial_imaginary,generate_frequency, num_timesteps, num_frequencies, num_elements, no_repeat_random, generated);

    if (verbose){
        print_element_data(1, num_frequencies, num_elements, ALL_FREQUENCIES, generated);
    }

    unsigned char temp_char;
    //correlate based on generated data
    for (int k = 0; k < num_timesteps; k++){
        int output_counter = 0;
        for (int j = 0; j < num_frequencies; j++){
            for (int element_y = 0; element_y < num_elements; element_y++){
                temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_y];
                int element_y_re = (int)(HI_NIBBLE(temp_char)) - 8; //-8 is to put the number back in the range -8 to 7 from 0 to 15
                int element_y_im = (int)(LO_NIBBLE(temp_char)) - 8;
                for (int element_x = element_y; element_x < num_elements; element_x++){
                    temp_char = generated[k*num_frequencies*num_elements+j*num_elements+element_x];
                    int element_x_re = (int)(HI_NIBBLE(temp_char)) - 8;
                    int element_x_im = (int)(LO_NIBBLE(temp_char)) - 8;
                    if (k != 0){
                        correlated_data_triangle[output_counter++] += element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data_triangle[output_counter++] += element_x_re*element_y_im - element_x_im*element_y_re;
                    }
                    else{
                        correlated_data_triangle[output_counter++] = element_x_re*element_y_re + element_x_im*element_y_im;
                        correlated_data_triangle[output_counter++] = element_x_re*element_y_im - element_x_im*element_y_re;
                    }
                }
            }
        }
    }

    //clean up parameters as needed
    free(generated);
    return (0);
}


void compare_NSquared_correlator_results ( int *num_err, int64_t *err_2, int num_frequencies, int num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
    //this will compare the values of the two arrays and give information about the comparison
    int address = 0;
    int local_Address = 0;
    *num_err = 0;
    *err_2 = 0;
    int max_error = 0;
    int amplitude_squared_error;
    double amplitude_squared_CPU;
    double amplitude_squared_GPU;
    double phase_angle_CPU;
    double phase_angle_GPU;
    for (int freq = 0; freq < num_frequencies; freq++){
        for (int element_y = 0; element_y < num_elements; element_y++){
            for (int element_x = 0; element_x < num_elements; element_x++){
                //compare real results
                int data_Real_GPU = data_set_GPU[address];
                int data_Real_CPU = data_set_CPU[address++];
                int difference_real = data_Real_GPU - data_Real_CPU;
                //compare imaginary results
                int data_Imag_GPU = data_set_GPU[address];
                int data_Imag_CPU = data_set_CPU[address++];
                int difference_imag = data_Imag_GPU - data_Imag_CPU;

                //get amplitude_squared
                amplitude_squared_CPU = data_Real_CPU*data_Real_CPU + data_Imag_CPU*data_Imag_CPU;
                amplitude_squared_GPU = data_Real_GPU*data_Real_GPU + data_Imag_GPU*data_Imag_GPU;
                phase_angle_CPU = atan2((double)data_Imag_CPU,(double)data_Real_CPU);
                phase_angle_GPU = atan2((double)data_Imag_GPU,(double)data_Real_GPU);

                if (amplitude_squared_CPU != 0){
                    ratio_GPU_div_CPU[local_Address] = amplitude_squared_GPU/amplitude_squared_CPU;
                }
                else{
                    ratio_GPU_div_CPU[local_Address] = -1;//amplitude_squared_GPU/amplitude_squared_CPU;
                }

                phase_difference[local_Address++] = phase_angle_GPU - phase_angle_CPU;

                if (difference_real != 0 || difference_imag !=0){
                    (*num_err)++;
                    if (verbosity ){
                        printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d ERR: %7d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU, *num_err);
                    }
                    amplitude_squared_error = difference_imag*difference_imag+difference_real*difference_real;
                    *err_2 += amplitude_squared_error;
                    if (amplitude_squared_error > max_error)
                        max_error = amplitude_squared_error;
                }
                else{
                    if (verbosity){
                        printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU);
                    }
                }
            }
        }
    }
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}

void compare_NSquared_correlator_results_data_has_upper_triangle_only ( int *num_err, int64_t *err_2, int actual_num_frequencies, int actual_num_elements, int *data_set_GPU, int *data_set_CPU, double *ratio_GPU_div_CPU, double *phase_difference, int verbosity){
    //this will compare the values of the two arrays and give information about the comparison
    int address = 0;
    int local_Address = 0;
    *num_err = 0;
    *err_2 = 0;
    int max_error = 0;
    int amplitude_squared_error;
    double amplitude_squared_CPU;
    double amplitude_squared_GPU;
    double phase_angle_CPU;
    double phase_angle_GPU;
    for (int freq = 0; freq < actual_num_frequencies; freq++){
        for (int element_y = 0; element_y < actual_num_elements; element_y++){
            for (int element_x = element_y; element_x < actual_num_elements; element_x++){
                //compare real results
                int data_Real_GPU = data_set_GPU[address];
                int data_Real_CPU = data_set_CPU[address++];
                int difference_real = data_Real_GPU - data_Real_CPU;
                //compare imaginary results
                int data_Imag_GPU = data_set_GPU[address];
                int data_Imag_CPU = data_set_CPU[address++];
                int difference_imag = data_Imag_GPU - data_Imag_CPU;

                //get amplitude_squared
                amplitude_squared_CPU = data_Real_CPU*data_Real_CPU + data_Imag_CPU*data_Imag_CPU;
                amplitude_squared_GPU = data_Real_GPU*data_Real_GPU + data_Imag_GPU*data_Imag_GPU;
                phase_angle_CPU = atan2((double)data_Imag_CPU,(double)data_Real_CPU);
                phase_angle_GPU = atan2((double)data_Imag_GPU,(double)data_Real_GPU);

                if (amplitude_squared_CPU != 0){
                    ratio_GPU_div_CPU[local_Address] = amplitude_squared_GPU/amplitude_squared_CPU;
                }
                else{
                    ratio_GPU_div_CPU[local_Address] = -1;//amplitude_squared_GPU/amplitude_squared_CPU;
                }

                phase_difference[local_Address++] = phase_angle_GPU - phase_angle_CPU;

                if (difference_real != 0 || difference_imag !=0){
                    (*num_err)++;
                    if (verbosity ){
                        printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d ERR: %7d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU, *num_err);
                    }
                    amplitude_squared_error = difference_imag*difference_imag+difference_real*difference_real;
                    *err_2 += amplitude_squared_error;
                    if (amplitude_squared_error > max_error)
                        max_error = amplitude_squared_error;
                }
                else{
                    if (verbosity){
                        printf ("freq: %6d element_x: %6d element_y: %6d Real CPU/GPU %8d %8d Imaginary CPU/GPU %8d %8d\n",freq, element_x, element_y, data_Real_CPU, data_Real_GPU, data_Imag_CPU, data_Imag_GPU);
                    }
                }
            }
        }
    }
    printf("\nTotal number of errors: %d, Sum of Squared Differences: %lld \n",*num_err, (long long int) *err_2);
    printf("sqrt(sum of squared differences/numberElements): %f \n", sqrt((*err_2)*1.0/local_Address));//add some more data--find maximum error, figure out other statistical properties
    printf("Maximum amplitude squared error: %d\n", max_error);
    return;
}
