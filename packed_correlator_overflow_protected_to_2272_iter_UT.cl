//Keith Vaderlinde's reduced-calculation-count version, coded by Peter Klages
#define NUM_ELEMENTS_div_4                          (NUM_ELEMENTS/4u)  // N/4
#define NUM_BLOCKS_x_2048                           (NUM_BLOCKS*2048u) //each block size is 32 x 32 x 2 = 2048

#define LOCAL_SIZE                                  8u
#define BLOCK_DIM_div_4                             8u
#define N_TIME_CHUNKS_LOCAL                         NUM_TIME_ACCUM

#define TIME_STEP_DIV_N_TIMESTEPS                   (get_global_id(2)/NUM_BLOCKS)
#define BLOCK_ID_LOCAL                              (get_global_id(2)%NUM_BLOCKS)
#define LOCAL_X                                     (get_local_id(0))
#define LOCAL_Y                                     (get_local_id(1))


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, LOCAL_SIZE, 1)))
void corr ( __global const uint *packed, //packed data loaded in groups of 4x(4+4bit) data to use the bus most efficiently
            __global  int *corr_buf, //buffer to save correlated results into
            __constant uint *id_x_map,
            __constant uint *id_y_map,
            __global int *block_lock)
{
    __local uint stillPackedX[256]; //1kB
    __local uint stillPackedY[256]; //1kB
    __local uint temp_y_real[256]; //1kB

    const uint block_x = id_x_map[BLOCK_ID_LOCAL]; //column of output block
    const uint block_y = id_y_map[BLOCK_ID_LOCAL]; //row of output block  //if NUM_BLOCKS = 1, then BLOCK_ID = 0 then block_x = block_y = 0

    /// The address for the x elements for the data
    //the common part that can be used for x and y is calculated first
    uint addr_x = (   LOCAL_Y*NUM_ELEMENTS_div_4 //LOCAL_Y used for grabbing a time offset here
                    + TIME_STEP_DIV_N_TIMESTEPS * N_TIME_CHUNKS_LOCAL *NUM_ELEMENTS_div_4); //frequency offset

    /// The address for the y elements for the data
    uint addr_y = ( (BLOCK_DIM_div_4*block_y + LOCAL_X) + addr_x); //offset into 1D input for the y lookup, using LOCAL X as the 0-7 offset for the group

    addr_x += (BLOCK_DIM_div_4*block_x + LOCAL_X); //the x address

    uint corr_a0=0u;
    uint corr_b0=0u;
    uint corr_c0=0u;
    uint corr_d0=0u;
    uint corr_e0=0u;
    uint corr_f0=0u;
    uint corr_g0=0u;
    uint corr_h0=0u;
    uint corr_a1=0u;
    uint corr_b1=0u;
    uint corr_c1=0u;
    uint corr_d1=0u;
    uint corr_e1=0u;
    uint corr_f1=0u;
    uint corr_g1=0u;
    uint corr_h1=0u;
    uint corr_a2=0u;
    uint corr_b2=0u;
    uint corr_c2=0u;
    uint corr_d2=0u;
    uint corr_e2=0u;
    uint corr_f2=0u;
    uint corr_g2=0u;
    uint corr_h2=0u;
    uint overflow_a=0u;
    uint overflow_b=0u;
    uint overflow_c=0u;
    uint overflow_d=0u;
    uint overflow_e=0u;
    uint overflow_f=0u;
    uint overflow_g=0u;
    uint overflow_h=0u;
    //vectors (i.e., uint4) make clearer code, but empirically had a very slight performance cost

    uint4 temp_stillPackedX;
    uint4 temp_stillPackedY;
    uint4 temp_Y; //new for this method--means only a reduction of 4 vgprs...
    uint temp_pa;
    uint pa;
    uint la;
    uint addr_o;

    uint extra_counter = 0; //should be a scalar, so no extra cost

//        uint address_offset= TIME_STEP_DIV_N_TIMESTEPS*2*N_TIME_CHUNKS_LOCAL*NUM_ELEMENTS_div_4 +repeat_count*N_TIME_CHUNKS_LOCAL*NUM_ELEMENTS_div_4;
        for (uint i = 0; i < N_TIME_CHUNKS_LOCAL; i += LOCAL_SIZE){ //256 is a number of timesteps to do a local accum before saving to global memory
            pa=packed[i * NUM_ELEMENTS_div_4 + addr_y];// + address_offset]; //add an additional time offset
            la=((LOCAL_Y<<5)| (LOCAL_X<<2)); //a short form for LOCAL_Y * 32 + LOCAL_X * 4 which may or may not be faster...

            barrier(CLK_LOCAL_MEM_FENCE);
            //unpack y values slightly (from 2 values per byte to 2 values per 4 bytes and fill local memory
            //first 'invert' the imaginary values (i.e. 15 - offset-encoded imaginary values)
            pa = pa ^0x0f0f0f0f; //exclusive or--flip the bits for the imaginary parts to 'invert'
            stillPackedY[la]    = ((pa & 0x000000f0) << 12u) | ((pa & 0x0000000f) >>  0u);
            stillPackedY[la+1u] = ((pa & 0x0000f000) <<  4u) | ((pa & 0x00000f00) >>  8u);
            stillPackedY[la+2u] = ((pa & 0x00f00000) >>  4u) | ((pa & 0x000f0000) >> 16u);
            stillPackedY[la+3u] = ((pa & 0xf0000000) >> 12u) | ((pa & 0x0f000000) >> 24u);

            //prepare the y array for quick lookup from local (and minimal number of calculations in the inner loop)
            temp_y_real[la]     = (pa & 0x000000f0)  >>  4u;
            temp_y_real[la+1u]  = (pa & 0x0000f000)  >> 12u;
            temp_y_real[la+2u]  = (pa & 0x00f00000)  >> 20u;
            temp_y_real[la+3u]  = (pa & 0xf0000000)  >> 28u;

            //barrier(CLK_LOCAL_MEM_FENCE);//does not appear to be needed on the current AMD architectures--things go in lockstep

            //unpack x values slightly (from 2 values per byte to 2 values per 4 bytes)
            temp_pa=packed[i * NUM_ELEMENTS_div_4 + addr_x ];

            stillPackedX[la]    = ((temp_pa & 0x000000f0) << 12u) | ((temp_pa & 0x0000000f) >>  0u);
            stillPackedX[la+1u] = ((temp_pa & 0x0000f000) <<  4u) | ((temp_pa & 0x00000f00) >>  8u);
            stillPackedX[la+2u] = ((temp_pa & 0x00f00000) >>  4u) | ((temp_pa & 0x000f0000) >> 16u);
            stillPackedX[la+3u] = ((temp_pa & 0xf0000000) >> 12u) | ((temp_pa & 0x0f000000) >> 24u);
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint j=0; j< LOCAL_SIZE; j++){
                temp_stillPackedX = vload4(mad24(j, 8u, LOCAL_X), stillPackedX);
                temp_stillPackedY = vload4(mad24(j, 8u, LOCAL_Y), stillPackedY);

                temp_Y = vload4(mad24(j, 8u, LOCAL_Y), temp_y_real);

                pa = temp_stillPackedX.s0;
                temp_pa = pa & 0x000f0000;
                corr_a0 = mad24(pa, temp_stillPackedY.s0, corr_a0);
                corr_b0 = mad24(pa, temp_stillPackedY.s1, corr_b0);
                corr_c0 = mad24(pa, temp_stillPackedY.s2, corr_c0);
                corr_d0 = mad24(pa, temp_stillPackedY.s3, corr_d0);

                pa = temp_stillPackedX.s1;
                temp_pa = temp_pa | ((pa >> 16u) & 0xf); //both reals repacked //do this outside the loop?
                corr_a1 = mad24(pa, temp_stillPackedY.s0, corr_a1);
                corr_b1 = mad24(pa, temp_stillPackedY.s1, corr_b1);
                corr_c1 = mad24(pa, temp_stillPackedY.s2, corr_c1);
                corr_d1 = mad24(pa, temp_stillPackedY.s3, corr_d1);

                //the extra term needed to get the overall real part in the correlation
                //real component 1 for two elements against the temp_Y
                corr_a2 = mad24(temp_pa, temp_Y.s0, corr_a2);
                corr_b2 = mad24(temp_pa, temp_Y.s1, corr_b2);
                corr_c2 = mad24(temp_pa, temp_Y.s2, corr_c2);
                corr_d2 = mad24(temp_pa, temp_Y.s3, corr_d2);

                //next two
                pa = temp_stillPackedX.s2;
                temp_pa = pa & 0x000f0000;
                corr_e0 = mad24(pa, temp_stillPackedY.s0, corr_e0);
                corr_f0 = mad24(pa, temp_stillPackedY.s1, corr_f0);
                corr_g0 = mad24(pa, temp_stillPackedY.s2, corr_g0);
                corr_h0 = mad24(pa, temp_stillPackedY.s3, corr_h0);

                pa = temp_stillPackedX.s3;
                temp_pa = temp_pa | ((pa >> 16u) & 0xf); //both reals packed
                corr_e1 = mad24(pa, temp_stillPackedY.s0, corr_e1);
                corr_f1 = mad24(pa, temp_stillPackedY.s1, corr_f1);
                corr_g1 = mad24(pa, temp_stillPackedY.s2, corr_g1);
                corr_h1 = mad24(pa, temp_stillPackedY.s3, corr_h1);

                corr_e2 = mad24(temp_pa, temp_Y.s0, corr_e2);
                corr_f2 = mad24(temp_pa, temp_Y.s1, corr_f2);
                corr_g2 = mad24(temp_pa, temp_Y.s2, corr_g2);
                corr_h2 = mad24(temp_pa, temp_Y.s3, corr_h2);

                extra_counter++;
            }
            // add code to keep track of possible overflows
            // it is known that overflows will only possibly occur after 145 iterations.

            if (extra_counter >= 120){ // 120 because taking only top 3 bits for the Im part
                //overflow data is packed 8 4 4 (Im Re-Pt1 Re-Pt2) x 2 and accumulated.  These
                //overflow bits limit the number of iterations in a single kernel.
                //15 overflows max in the 1 b nibbles: 15 * 120 = 1800 iterations max,
                //but it can't overflow 15 times in 15 iterations.
                //Max iter = 2272 for complete correctness
                //(note that having larger overflow section would mean more registers would be used and
                //the kernel would slow down--we are VGPR limited at the moment)
                overflow_a  += (((corr_a0 & 0xE0000000)>>5)  | ((corr_a0 & 0x00008000)<<5)
                              | ((corr_a1 & 0xE0000000)>>21) | ((corr_a1 & 0x00008000)>>11)
                              | ((corr_a2 & 0x80008000)>>15) );

                overflow_b  += (((corr_b0 & 0xE0000000)>>5)  | ((corr_b0 & 0x00008000)<<5)
                              | ((corr_b1 & 0xE0000000)>>21) | ((corr_b1 & 0x00008000)>>11)
                              | ((corr_b2 & 0x80008000)>>15) );

                overflow_c  += (((corr_c0 & 0xE0000000)>>5)  | ((corr_c0 & 0x00008000)<<5)
                              | ((corr_c1 & 0xE0000000)>>21) | ((corr_c1 & 0x00008000)>>11)
                              | ((corr_c2 & 0x80008000)>>15) );

                overflow_d  += (((corr_d0 & 0xE0000000)>>5)  | ((corr_d0 & 0x00008000)<<5)
                              | ((corr_d1 & 0xE0000000)>>21) | ((corr_d1 & 0x00008000)>>11)
                              | ((corr_d2 & 0x80008000)>>15) );

                overflow_e  += (((corr_e0 & 0xE0000000)>>5)  | ((corr_e0 & 0x00008000)<<5)
                              | ((corr_e1 & 0xE0000000)>>21) | ((corr_e1 & 0x00008000)>>11)
                              | ((corr_e2 & 0x80008000)>>15) );

                overflow_f  += (((corr_f0 & 0xE0000000)>>5)  | ((corr_f0 & 0x00008000)<<5)
                              | ((corr_f1 & 0xE0000000)>>21) | ((corr_f1 & 0x00008000)>>11)
                              | ((corr_f2 & 0x80008000)>>15) );

                overflow_g  += (((corr_g0 & 0xE0000000)>>5)  | ((corr_g0 & 0x00008000)<<5)
                              | ((corr_g1 & 0xE0000000)>>21) | ((corr_g1 & 0x00008000)>>11)
                              | ((corr_g2 & 0x80008000)>>15) );

                overflow_h  += (((corr_h0 & 0xE0000000)>>5)  | ((corr_h0 & 0x00008000)<<5)
                              | ((corr_h1 & 0xE0000000)>>21) | ((corr_h1 & 0x00008000)>>11)
                              | ((corr_h2 & 0x80008000)>>15) );

                corr_a0 = corr_a0 &0x1FFF7FFF;
                corr_a1 = corr_a1 &0x1FFF7FFF;
                corr_a2 = corr_a2 &0x7FFF7FFF;
                corr_b0 = corr_b0 &0x1FFF7FFF;
                corr_b1 = corr_b1 &0x1FFF7FFF;
                corr_b2 = corr_b2 &0x7FFF7FFF;
                corr_c0 = corr_c0 &0x1FFF7FFF;
                corr_c1 = corr_c1 &0x1FFF7FFF;
                corr_c2 = corr_c2 &0x7FFF7FFF;
                corr_d0 = corr_d0 &0x1FFF7FFF;
                corr_d1 = corr_d1 &0x1FFF7FFF;
                corr_d2 = corr_d2 &0x7FFF7FFF;
                corr_e0 = corr_e0 &0x1FFF7FFF;
                corr_e1 = corr_e1 &0x1FFF7FFF;
                corr_e2 = corr_e2 &0x7FFF7FFF;
                corr_f0 = corr_f0 &0x1FFF7FFF;
                corr_f1 = corr_f1 &0x1FFF7FFF;
                corr_f2 = corr_f2 &0x7FFF7FFF;
                corr_g0 = corr_g0 &0x1FFF7FFF;
                corr_g1 = corr_g1 &0x1FFF7FFF;
                corr_g2 = corr_g2 &0x7FFF7FFF;
                corr_h0 = corr_h0 &0x1FFF7FFF;
                corr_h1 = corr_h1 &0x1FFF7FFF;
                corr_h2 = corr_h2 &0x7FFF7FFF;

                extra_counter = 0;
            }

        }


        //output: 32 numbers--> 16 pairs of real/imag numbers
        //32 * 8 (local_size(0)) * 8 (local_size(1)) = 2048 ints / block
        addr_o = ((BLOCK_ID_LOCAL * 2048u) + (LOCAL_Y * 256u) + (LOCAL_X * 8u));// +((TIME_STEP_DIV_N_TIMESTEPS*SIZE_PER_SET)&0xf)) ; //extra part cycles through outputs


    //     custom block spin-lock section
        if (LOCAL_X == 0 && LOCAL_Y == 0){
            while(atomic_cmpxchg(&block_lock[BLOCK_ID_LOCAL],0,1)); //wait until unlocked
        }

            barrier(CLK_GLOBAL_MEM_FENCE); //sync point for the group
            //note that to be careful, each output needs to include their overflow protection values

            corr_buf[addr_o+0u]   += (((corr_a2 >> 16u)& 0xffff) + ((overflow_a  & 0x000F0000)>>  1))  - ((corr_a0 & 0xFFFF) + ((overflow_a &0x00F00000)>> 5)) ; //real value
            corr_buf[addr_o+1u]   -= (((corr_a0 >> 16u)& 0xffff) + ((overflow_a  & 0xFF000000)>> 11)); //imag value
            corr_buf[addr_o+2u]   += (((corr_a2 >>  0u)& 0xffff) + ((overflow_a  & 0x0000000F)<< 15))  - ((corr_a1 & 0xFFFF) + ((overflow_a &0x000000F0)<<11)) ;
            corr_buf[addr_o+3u]   -= (((corr_a1 >> 16u)& 0xffff) + ((overflow_a  & 0x0000FF00)<<  5));
            corr_buf[addr_o+4u]   += (((corr_e2 >> 16u)& 0xffff) + ((overflow_e  & 0x000F0000)>>  1))  - ((corr_e0 & 0xFFFF) + ((overflow_e &0x00F00000)>> 5)) ;
            corr_buf[addr_o+5u]   -= (((corr_e0 >> 16u)& 0xffff) + ((overflow_e  & 0xFF000000)>> 11));
            corr_buf[addr_o+6u]   += (((corr_e2 >>  0u)& 0xffff) + ((overflow_e  & 0x0000000F)<< 15))  - ((corr_e1 & 0xFFFF) + ((overflow_e &0x000000F0)<<11)) ;
            corr_buf[addr_o+7u]   -= (((corr_e1 >> 16u)& 0xffff) + ((overflow_e  & 0x0000FF00)<<  5));

            //next 4 complex numbers from the next row
            corr_buf[addr_o+64u]  += (((corr_b2 >> 16u)& 0xffff) + ((overflow_b  & 0x000F0000)>>  1))  - ((corr_b0 & 0xFFFF) + ((overflow_b &0x00F00000)>> 5)) ; //real value
            corr_buf[addr_o+65u]  -= (((corr_b0 >> 16u)& 0xffff) + ((overflow_b  & 0xFF000000)>> 11)); //imag value
            corr_buf[addr_o+66u]  += (((corr_b2 >>  0u)& 0xffff) + ((overflow_b  & 0x0000000F)<< 15))  - ((corr_b1 & 0xFFFF) + ((overflow_b &0x000000F0)<<11)) ;
            corr_buf[addr_o+67u]  -= (((corr_b1 >> 16u)& 0xffff) + ((overflow_b  & 0x0000FF00)<<  5));
            corr_buf[addr_o+68u]  += (((corr_f2 >> 16u)& 0xffff) + ((overflow_f  & 0x000F0000)>>  1))  - ((corr_f0 & 0xFFFF) + ((overflow_f &0x00F00000)>> 5)) ;
            corr_buf[addr_o+69u]  -= (((corr_f0 >> 16u)& 0xffff) + ((overflow_f  & 0xFF000000)>> 11));
            corr_buf[addr_o+70u]  += (((corr_f2 >>  0u)& 0xffff) + ((overflow_f  & 0x0000000F)<< 15))  - ((corr_f1 & 0xFFFF) + ((overflow_f &0x000000F0)<<11)) ;
            corr_buf[addr_o+71u]  -= (((corr_f1 >> 16u)& 0xffff) + ((overflow_f  & 0x0000FF00)<<  5));

            corr_buf[addr_o+128u] += (((corr_c2 >> 16u)& 0xffff) + ((overflow_c  & 0x000F0000)>>  1))  - ((corr_c0 & 0xFFFF) + ((overflow_c &0x00F00000)>> 5)) ; //real value
            corr_buf[addr_o+129u] -= (((corr_c0 >> 16u)& 0xffff) + ((overflow_c  & 0xFF000000)>> 11)); //imag value
            corr_buf[addr_o+130u] += (((corr_c2 >>  0u)& 0xffff) + ((overflow_c  & 0x0000000F)<< 15))  - ((corr_c1 & 0xFFFF) + ((overflow_c &0x000000F0)<<11)) ;
            corr_buf[addr_o+131u] -= (((corr_c1 >> 16u)& 0xffff) + ((overflow_c  & 0x0000FF00)<<  5));
            corr_buf[addr_o+132u] += (((corr_g2 >> 16u)& 0xffff) + ((overflow_g  & 0x000F0000)>>  1))  - ((corr_g0 & 0xFFFF) + ((overflow_g &0x00F00000)>> 5)) ;
            corr_buf[addr_o+133u] -= (((corr_g0 >> 16u)& 0xffff) + ((overflow_g  & 0xFF000000)>> 11));
            corr_buf[addr_o+134u] += (((corr_g2 >>  0u)& 0xffff) + ((overflow_g  & 0x0000000F)<< 15))  - ((corr_g1 & 0xFFFF) + ((overflow_g &0x000000F0)<<11)) ;
            corr_buf[addr_o+135u] -= (((corr_g1 >> 16u)& 0xffff) + ((overflow_g  & 0x0000FF00)<<  5));

            corr_buf[addr_o+192u] += (((corr_d2 >> 16u)& 0xffff) + ((overflow_d  & 0x000F0000)>>  1))  - ((corr_d0 & 0xFFFF) + ((overflow_d &0x00F00000)>> 5)) ; //real value
            corr_buf[addr_o+193u] -= (((corr_d0 >> 16u)& 0xffff) + ((overflow_d  & 0xFF000000)>> 11)); //imag value
            corr_buf[addr_o+194u] += (((corr_d2 >>  0u)& 0xffff) + ((overflow_d  & 0x0000000F)<< 15))  - ((corr_d1 & 0xFFFF) + ((overflow_d &0x000000F0)<<11)) ;
            corr_buf[addr_o+195u] -= (((corr_d1 >> 16u)& 0xffff) + ((overflow_d  & 0x0000FF00)<<  5));
            corr_buf[addr_o+196u] += (((corr_h2 >> 16u)& 0xffff) + ((overflow_h  & 0x000F0000)>>  1))  - ((corr_h0 & 0xFFFF) + ((overflow_h &0x00F00000)>> 5)) ;
            corr_buf[addr_o+197u] -= (((corr_h0 >> 16u)& 0xffff) + ((overflow_h  & 0xFF000000)>> 11));
            corr_buf[addr_o+198u] += (((corr_h2 >>  0u)& 0xffff) + ((overflow_h  & 0x0000000F)<< 15))  - ((corr_h1 & 0xFFFF) + ((overflow_h &0x000000F0)<<11)) ;
            corr_buf[addr_o+199u] -= (((corr_h1 >> 16u)& 0xffff) + ((overflow_h  & 0x0000FF00)<<  5));

            barrier(CLK_GLOBAL_MEM_FENCE); //make sure everyone is done

        if (LOCAL_X == 0 && LOCAL_Y == 0)
            block_lock[BLOCK_ID_LOCAL]=0;

}
