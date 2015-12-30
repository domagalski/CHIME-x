//NUM_ELEMENTS, NUM_BLOCKS, NUM_TIMESAMPLES defined at compile time
//#define NUM_ELEMENTS                    32u // must be 32 or larger, 2560u eventually
//#define NUM_BLOCKS                      1u  // N(N+1)/2 where N=(NUM_ELEMENTS/32)
//#define NUM_TIMESAMPLES                 10u*1024u


#define NUM_TIMESAMPLES_x_128           (NUM_TIMESAMPLES*128u)// need the total number of iterations for the offset outputs: this 128 is from the offset calcs--not part of the smallest accum period
#define NUM_BLOCKS_x_2048               (NUM_BLOCKS*2048u) //each block size is 32 x 32 x 2 = 2048

#define LOCAL_SIZE                      8u
#define BLOCK_DIM                       32u

#define FREQUENCY_BAND                  (get_group_id(1))
#define BLOCK_ID                        (get_group_id(2))
#define LOCAL_X                         (get_local_id(0))
#define LOCAL_Y                         (get_local_id(1))


__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, LOCAL_SIZE, 1)))
void preseed( __global const uint *dataIn,
              __global        int *corr_buf,
              __constant     uint *id_x_map,
              __constant     uint *id_y_map,
              __local        uint *localDataX,
              __local        uint *localDataY)
{
    uint block_x = id_x_map[BLOCK_ID]; //column of output block
    uint block_y = id_y_map[BLOCK_ID]; //row of output block  //if NUM_BLOCKS = 1, then BLOCK_ID = 0 then block_x = block_y = 0

    uint local_index = LOCAL_X + LOCAL_Y*LOCAL_SIZE; //0-63

    uint base_addr_x = ( (BLOCK_DIM*block_x)
                    + FREQUENCY_BAND*NUM_ELEMENTS)*2u; //times 2 because there are pairs of numbers for the complex values
    uint base_addr_y = ( (BLOCK_DIM*block_y)
                    + FREQUENCY_BAND*NUM_ELEMENTS)*2u;

    uint8 xVals;
    uint8 yVals; //using vectors since data will be in contiguous memory spaces and can then load 8 items at a time (in the hopes of getting coalesced loads
    int  xValOffsets[8u];
    int  yValOffsets[8u];

    //synchronize then load
    barrier(CLK_LOCAL_MEM_FENCE);
    //want to load 32 complex values (i.e. 64 values)
    localDataX[local_index] = dataIn[base_addr_x+local_index]; //local_index has 64 contiguous entries
    localDataY[local_index] = dataIn[base_addr_y+local_index];
    barrier(CLK_LOCAL_MEM_FENCE);

    //load relevant values for this work item
    xVals = vload8(LOCAL_X,localDataX); //offsets are in sizes of the vector, so 8 uints big
    yVals = vload8(LOCAL_Y,localDataY);

    //if registers are the slowing point of this algorithm, then can possibly reuse variables, though this should be okay as a test
    xValOffsets[0u] = -8*((int)xVals.s0) + 7*((int)xVals.s1);
    xValOffsets[1u] =  8*((int)xVals.s1) + 7*((int)xVals.s0);
    xValOffsets[2u] = -8*((int)xVals.s2) + 7*((int)xVals.s3);
    xValOffsets[3u] =  8*((int)xVals.s3) + 7*((int)xVals.s2);
    xValOffsets[4u] = -8*((int)xVals.s4) + 7*((int)xVals.s5);
    xValOffsets[5u] =  8*((int)xVals.s5) + 7*((int)xVals.s4);
    xValOffsets[6u] = -8*((int)xVals.s6) + 7*((int)xVals.s7);
    xValOffsets[7u] =  8*((int)xVals.s7) + 7*((int)xVals.s6);

    yValOffsets[0u] = -8*(((int)yVals.s0) + ((int)yVals.s1));
    yValOffsets[1u] =  8*(((int)yVals.s0) - ((int)yVals.s1));
    yValOffsets[2u] = -8*(((int)yVals.s2) + ((int)yVals.s3));
    yValOffsets[3u] =  8*(((int)yVals.s2) - ((int)yVals.s3));
    yValOffsets[4u] = -8*(((int)yVals.s4) + ((int)yVals.s5));
    yValOffsets[5u] =  8*(((int)yVals.s4) - ((int)yVals.s5));
    yValOffsets[6u] = -8*(((int)yVals.s6) + ((int)yVals.s7));
    yValOffsets[7u] =  8*(((int)yVals.s6) - ((int)yVals.s7));

    //output results
    //Each work item outputs 4 x 4 complex values (so 32 values rather than 16)
    //
    //offset to the next row is 8 (local_x vals) x 8 vals = 64
    //each y takes care of 4 values, so y * 4*64
    //
    //16 pairs * 8 (local_size(0)) * 8 (local_size(1)) = 1024
    uint addr_o = ((BLOCK_ID * 2048u) + (LOCAL_Y * 256u) + (LOCAL_X * 8u)) + (FREQUENCY_BAND * NUM_BLOCKS_x_2048);
    //row 0
    corr_buf[addr_o+0u]   =   NUM_TIMESAMPLES_x_128 + xValOffsets[0] + yValOffsets[0]; //real value correction
    corr_buf[addr_o+1u]   =                           xValOffsets[1] + yValOffsets[1]; //imaginary value correction (the extra subtraction in the notes has been performed by swapping order above
    corr_buf[addr_o+2u]   =   NUM_TIMESAMPLES_x_128 + xValOffsets[2] + yValOffsets[0]; //note that x changes, but y stays the same
    corr_buf[addr_o+3u]   =                           xValOffsets[3] + yValOffsets[1];
    corr_buf[addr_o+4u]   =   NUM_TIMESAMPLES_x_128 + xValOffsets[4] + yValOffsets[0];
    corr_buf[addr_o+5u]   =                           xValOffsets[5] + yValOffsets[1];
    corr_buf[addr_o+6u]   =   NUM_TIMESAMPLES_x_128 + xValOffsets[6] + yValOffsets[0];
    corr_buf[addr_o+7u]   =                           xValOffsets[7] + yValOffsets[1];
    //row 1
    corr_buf[addr_o+64u]  =   NUM_TIMESAMPLES_x_128 + xValOffsets[0] + yValOffsets[2]; //real value correction
    corr_buf[addr_o+65u]  =                           xValOffsets[1] + yValOffsets[3]; //imaginary value correction (the extra subtraction in the notes has been performed by swapping order above
    corr_buf[addr_o+66u]  =   NUM_TIMESAMPLES_x_128 + xValOffsets[2] + yValOffsets[2]; //note that x changes, but y stays the same
    corr_buf[addr_o+67u]  =                           xValOffsets[3] + yValOffsets[3];
    corr_buf[addr_o+68u]  =   NUM_TIMESAMPLES_x_128 + xValOffsets[4] + yValOffsets[2];
    corr_buf[addr_o+69u]  =                           xValOffsets[5] + yValOffsets[3];
    corr_buf[addr_o+70u]  =   NUM_TIMESAMPLES_x_128 + xValOffsets[6] + yValOffsets[2];
    corr_buf[addr_o+71u]  =                           xValOffsets[7] + yValOffsets[3];

    //row 2
    corr_buf[addr_o+128u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[0] + yValOffsets[4]; //real value correction
    corr_buf[addr_o+129u] =                           xValOffsets[1] + yValOffsets[5]; //imaginary value correction (the extra subtraction in the notes has been performed by swapping order above
    corr_buf[addr_o+130u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[2] + yValOffsets[4]; //note that x changes, but y stays the same
    corr_buf[addr_o+131u] =                           xValOffsets[3] + yValOffsets[5];
    corr_buf[addr_o+132u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[4] + yValOffsets[4];
    corr_buf[addr_o+133u] =                           xValOffsets[5] + yValOffsets[5];
    corr_buf[addr_o+134u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[6] + yValOffsets[4];
    corr_buf[addr_o+135u] =                           xValOffsets[7] + yValOffsets[5];

    //row 3
    corr_buf[addr_o+192u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[0] + yValOffsets[6]; //real value correction
    corr_buf[addr_o+193u] =                           xValOffsets[1] + yValOffsets[7]; //imaginary value correction (the extra subtraction in the notes has been performed by swapping order above
    corr_buf[addr_o+194u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[2] + yValOffsets[6]; //note that x changes, but y stays the same
    corr_buf[addr_o+195u] =                           xValOffsets[3] + yValOffsets[7];
    corr_buf[addr_o+196u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[4] + yValOffsets[6];
    corr_buf[addr_o+197u] =                           xValOffsets[5] + yValOffsets[7];
    corr_buf[addr_o+198u] =   NUM_TIMESAMPLES_x_128 + xValOffsets[6] + yValOffsets[6];
    corr_buf[addr_o+199u] =                           xValOffsets[7] + yValOffsets[7];

}
