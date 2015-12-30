//ACTUAL_NUM_ELEMENTS, ACTUAL_NUM_FREQUENCIES defined at compile time
//#define ACTUAL_NUM_ELEMENTS         16u
//#define ACTUAL_NUM_FREQUENCIES      256u
#define NUM_TIMESTEPS_LOCAL         BASE_ACCUM
#define OFFSET_FOR_1_TIMESTEP       (NUM_ELEMENTS*NUM_FREQUENCIES/4u) //divide by 4 to account for the 4 B per uint

__kernel void offsetAccumulateElements (__global const uint *inputData,
                                        __global uint *outputData){
    uint    data;
    uint4   dataExpanded = (uint4)(0u,0u,0u,0u);
    uint4   temp;
    //we load 4 Byte words, addresses are based on that size
    uint    address = get_local_id(0) + //0-63
                      get_group_id(1)*64u;//   the 64 comes from the fact we're using 4 B words... that is we're mutiplying by 256 B / 4 B

    //only compute values if the output address is going to be valid
    //address * 8 accounts for the 8 values (4 pairs of complex numbers) accessed by loading 4 B of packed data
    if (address * 8u + 7u < NUM_ELEMENTS*NUM_FREQUENCIES*2u){ //check to see if the output address falls in a useful range (i.e. < Num_Elements x Num_Freq)
        address += get_group_id(2)*NUM_TIMESTEPS_LOCAL*OFFSET_FOR_1_TIMESTEP;

        for (int i = 0; i < NUM_TIMESTEPS_LOCAL; i++){
            data = inputData[address]; //should be a coalesced load with local work group

            //unpack
            temp.s0 = ((data & 0x000000f0) << 12u) | ((data & 0x0000000f) >>   0u);
            temp.s1 = ((data & 0x0000f000) <<  4u) | ((data & 0x00000f00) >>   8u);
            temp.s2 = ((data & 0x00f00000) >>  4u) | ((data & 0x000f0000) >>  16u);
            temp.s3 = ((data & 0xf0000000) >> 12u) | ((data & 0x0f000000) >>  24u);

            //accumulate
            dataExpanded += temp;

            //update address for next iteration
            address += OFFSET_FOR_1_TIMESTEP;
        }

        //output reduced data set, expanding one more time (store as uint to avoid a cast to int)--recasting will be done when expanding the smaller N*M*2 matrix to N(N+1)/2*M*2
        //8 output values
        address = (get_local_id(0) + get_group_id(1)*64u)*8u; //([0,504] + [0,M*N/256)*512)
        atomic_add(&outputData[address+0u], (dataExpanded.s0>>16) );         //real value
        atomic_add(&outputData[address+1u], (dataExpanded.s0&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+2u], (dataExpanded.s1>>16) );         //real value
        atomic_add(&outputData[address+3u], (dataExpanded.s1&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+4u], (dataExpanded.s2>>16) );         //real value
        atomic_add(&outputData[address+5u], (dataExpanded.s2&0x0000ffff) );  //imaginary
        atomic_add(&outputData[address+6u], (dataExpanded.s3>>16) );         //real value
        atomic_add(&outputData[address+7u], (dataExpanded.s3&0x0000ffff) );  //imaginary
    }
}
