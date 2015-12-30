// gpu_cpu_helpers.c

#include "gpu_cpu_helpers.h"
#include <stdio.h>
#include <sys/time.h>
#include "four_bit_macros.h"
#include "input_generator.h"

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}



void print_element_data(int num_timesteps, int num_frequencies, int num_elements, int particular_frequency, unsigned char *data){
    printf("Number of timesteps to print: %d, ", num_timesteps);
    if (particular_frequency == ALL_FREQUENCIES)
        printf("number of frequency bands: %d, number of elements: %d\n", num_frequencies, num_elements);
    else
        printf("frequency band: %d, number of elements: %d\n", particular_frequency, num_elements);

    for (int k = 0; k < num_timesteps; k++){
        if (num_timesteps > 1){
            printf("Time Step %d\n", k);
        }
        printf("            ");
        for (int header_i = 0; header_i < num_elements; header_i++){
            printf("%3dR %3dI ", header_i, header_i);
        }
        printf("\n");
        for (int j = 0; j < num_frequencies; j++){
            if (particular_frequency == ALL_FREQUENCIES || particular_frequency == j){
                if (particular_frequency != j)
                    printf("Freq: %4d: ", j);

                for (int i = 0; i < num_elements; i++){
                    unsigned char temp = data[k*num_frequencies*num_elements+j*num_elements+i];
                    printf("%4d %4d ",(int)(HI_NIBBLE(temp))-8,(int)(LO_NIBBLE(temp))-8);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}
