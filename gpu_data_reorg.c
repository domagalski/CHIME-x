// gpu_data_reorg.c
// The correlation data output from the gpu algorithms is organized in tiles that cover the upper triangle
// These functions reorganize the data to other forms that are more useful on the cpu side


void reorganize_32_to_16_feed_GPU_Correlated_Data(int actual_num_frequencies, int actual_num_elements, int *correlated_data){
    //data is processed as 32 elements x 32 elements to fit the kernel even though only 16 elements exist.
    //This is equivalent to processing 2 elements at the same time, where the desired correlations live in the first and fourth quadrants
    //This function is to reorganize the data so that comparisons can be done more easily

    //The input dataset is larger than the output, so can reorganize in the same array

    int input_frequencies = actual_num_frequencies/2;
    int input_elements = actual_num_elements*2;
    int address = 0;
    int address_out = 0;
    for (int freq = 0; freq < input_frequencies; freq++){
        for (int element_y = 0; element_y < input_elements; element_y++){
            for (int element_x = 0; element_x < input_elements; element_x++){
                if (element_x < actual_num_elements && element_y < actual_num_elements){
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] = correlated_data[address++]; //real and imaginary at each spot
                }
                else if (element_x >=actual_num_elements && element_y >=actual_num_elements){
                    correlated_data[address_out++] = correlated_data[address++];
                    correlated_data[address_out++] = correlated_data[address++];
                }
                else
                    address += 2;
            }
        }
    }
    return;
}

void reorganize_GPU_to_full_Matrix_for_comparison(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix){
    //takes the output data, grouped in blocks of block_dim x block_dim x 2 (complex pairs (ReIm)of ints), and fills a num_elements x num_elements x 2
    //
    for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
        int block_x_ID = 0;
        int block_y_ID = 0;
        int num_blocks_x = actual_num_elements/block_side_length;
        int block_check = num_blocks_x;

        for (int block_ID = 0; block_ID < num_blocks; block_ID++){
            if (block_ID == block_check){
                num_blocks_x--;
                block_check += num_blocks_x;
                block_y_ID++;
                block_x_ID = block_y_ID;
            }
            for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++){
                int y_ID_global = block_y_ID * block_side_length + y_ID_local;
                for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++){
                    int GPU_address = frequency_bin*(num_blocks*block_side_length*block_side_length*2) + block_ID *(block_side_length*block_side_length*2) + y_ID_local*block_side_length*2+x_ID_local*2; ///TO DO :simplify this statement after getting everything working
                    int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                    if (x_ID_global >= y_ID_global){
                        if (x_ID_global > y_ID_global){ //store the conjugate: x and y addresses get swapped and the imaginary value is the negative of the original value
                            final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2]   =  gpu_data[GPU_address];
                            final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+x_ID_global*actual_num_elements+y_ID_global)*2+1] = -gpu_data[GPU_address+1];
                        }
                        //store the
                        final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2]   = gpu_data[GPU_address];
                        final_matrix[(frequency_bin*actual_num_elements*actual_num_elements+y_ID_global*actual_num_elements+x_ID_global)*2+1] = gpu_data[GPU_address+1];
                    }
                }
            }
            //printf("block_ID: %d, block_y_ID: %d, block_x_ID: %d\n", block_ID, block_y_ID, block_x_ID);
            //update block offset values
            block_x_ID++;
        }
    }
    return;
}

void reorganize_GPU_to_upper_triangle(int block_side_length, int num_blocks, int actual_num_frequencies, int actual_num_elements, int *gpu_data, int *final_matrix){
    int GPU_address = 0; //we go through the gpu data sequentially and map it to the proper locations in the output array
    for (int frequency_bin = 0; frequency_bin < actual_num_frequencies; frequency_bin++ ){
        int block_x_ID = 0;
        int block_y_ID = 0;
        int num_blocks_x = actual_num_elements/block_side_length;
        int block_check = num_blocks_x;
        int frequency_offset = frequency_bin * (actual_num_elements* (actual_num_elements+1))/2;// frequency_bin * number of items in an upper triangle

        for (int block_ID = 0; block_ID < num_blocks; block_ID++){
            if (block_ID == block_check){
                num_blocks_x--;
                block_check += num_blocks_x;
                block_y_ID++;
                block_x_ID = block_y_ID;
            }

            for (int y_ID_local = 0; y_ID_local < block_side_length; y_ID_local++){

                for (int x_ID_local = 0; x_ID_local < block_side_length; x_ID_local++){

                    int x_ID_global = block_x_ID * block_side_length + x_ID_local;
                    int y_ID_global = block_y_ID * block_side_length + y_ID_local;

                    /// address_1d_output = frequency_offset, plus the number of entries in the rectangle area (y_ID_global*actual_num_elements), minus the number of elements in lower triangle to that row (((y_ID_global-1)*y_ID_global)/2), plus the contributions to the address from the current row (x_ID_global - y_ID_global)
                    int address_1d_output = frequency_offset + y_ID_global*actual_num_elements - ((y_ID_global-1)*y_ID_global)/2 + (x_ID_global - y_ID_global);

                    if (block_x_ID != block_y_ID){ //when we are not in the diagonal blocks
                        final_matrix[address_1d_output*2  ] = gpu_data[GPU_address++];
                        final_matrix[address_1d_output*2+1] = gpu_data[GPU_address++];
                    }
                    else{ // the special case needed to deal with the diagonal pieces
                        if (x_ID_local >= y_ID_local){
                            final_matrix[address_1d_output*2  ] = gpu_data[GPU_address++];
                            final_matrix[address_1d_output*2+1] = gpu_data[GPU_address++];
                        }
                        else{
                            GPU_address += 2;
                        }
                    }
                }
            }
            //offset_GPU += (block_side_length*block_side_length);
            //update block offset values
            block_x_ID++;
        }
    }
    return;
}

void reorganize_data_16_element_with_triangle_conversion (int num_frequencies_final, int actual_num_frequencies, int *input_data, int *output_data){
    //input data should be arranged as (num_elements*(num_elements+1))/2 (real,imag) pairs of complex visibilities for frequencies
    //output array will be sparsely to moderately filled, so loop such that writing is done in sequential order
    //int num_complex_visibilities = 136;//16*(16+1)/2; //(n*(n+1)/2)
    int output_counter = 0;
    for (int freq_count = 0; freq_count < num_frequencies_final; freq_count++){
        for (int y = 0; y < 16; y++){
            for (int x = y; x < 16; x++){
                if (freq_count < actual_num_frequencies){
                    int input_index = (freq_count * 256 + y*16 + x)*2; //blocks of data are 16 x 16 = 256 and row_stride is 16
                    output_data [output_counter++] = input_data[input_index];
                    output_data [output_counter++] = input_data[input_index+1];
                    //output_data [(data_count*num_frequencies_final + freq_count)] = (double)input_data[input_index] + I * (double)input_data[input_index+1];
                }
                else{
                    output_data [output_counter++] = 0;
                    output_data [output_counter++] = 0;
                    //output_data [(data_count*num_frequencies_final + freq_count)] = (double)0.0 + I * (double)0.0;
                }
            }
        }
    }
    return;
}
