//gpu_data_reorg.h

#ifndef GPU_DATA_REORG_H
#define GPU_DATA_REORG_H

void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies, int actual_num_elements, int *correlated_data);

void reorganize_GPU_to_full_Matrix_for_comparison(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix);

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix);

void reorganize_data_16_element_with_triangle_conversion (int num_frequencies_final, int actual_num_frequencies, int *input_data, int *output_data);

#endif
