// This testing code for the AMD correlator kernel is being released under the MIT License

// For this to work, you will need an AMD GPU and the AMD APP SDK for OpenCL installed: Note this code was written
// with OpenCL 1.2 in mind, not 2.0

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
//#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <assert.h>
#include <getopt.h>
#include "amd_firepro_error_code_list_for_opencl.h"
#include "input_generator.h"
#include "four_bit_macros.h"
#include "gpu_data_reorg.h"
#include "gpu_cpu_helpers.h"
#include "cpu_corr_test.h"


#define NUM_CL_FILES                    3
#define OPENCL_FILENAME_PACKED1_1       "pairwise_correlator.cl"
#define OPENCL_FILENAME_PACKED1_2       "offset_accumulator.cl"
#define OPENCL_FILENAME_PACKED1_3       "preseed_multifreq.cl"

#define OPENCL_FILENAME_PACKED1_UT_1    "pairwise_correlator_UT.cl"
#define OPENCL_FILENAME_PACKED1_UT_2    "offset_accumulator.cl"
#define OPENCL_FILENAME_PACKED1_UT_3    "preseed_multifreq_UT.cl"

#define OPENCL_FILENAME_PACKED2_1       "packed_correlator_overflow_protected_to_2272_iter.cl"   //limiting factor-1 bit overflow counter: 145*15+112=2287
#define OPENCL_FILENAME_PACKED2_2       "offset_accumulator.cl"
#define OPENCL_FILENAME_PACKED2_3       "preseed_multifreq_highly_packed_correlator_method.cl"

#define OPENCL_FILENAME_PACKED2_UT_1    "packed_correlator_overflow_protected_to_2272_iter_UT.cl"   //limiting factor-1 bit overflow counter: 145*15+112=2287
#define OPENCL_FILENAME_PACKED2_UT_2    "offset_accumulator.cl"
#define OPENCL_FILENAME_PACKED2_UT_3    "preseed_multifreq_highly_packed_correlator_method_UT.cl"

#define N_STAGES                        2 //write to CL_Mem, Kernel (Read is done after many runs since answers are accumulated)
#define N_QUEUES                        2 //have 2 separate queues so transfer and process paths can be queued nicely
#define PAGESIZE_MEM                    4096u
#define BASE_TIMESAMPLES_ACCUM          32u

#define SDK_SUCCESS                     0u
#define TRIANGLE                        1

void print_help() {
    printf("Usage: sudo ./correlator_test [opts]\n\n");
    printf("Options:\n");
    printf("  --help (-h)                               Display the available run options.\n");
    printf("  --device (-d) [device_number]             Default: 0. For multi-GPU computers, can choose larger values.\n");
    printf("  --iterations (-i) [number]                Default: 100. Number of iterations to test the code.\n");
    printf("  --time_accum (-t) [number]                Default: 256. Range [1,291] for version from conference, [1,1912] for new alg.\n");
    printf("  --time_steps (-T) [number]                Default: Automatically generated. Number of time steps of element data.\n");
    printf("  --num_freq (-f) [number]                  Default: 1. Number of frequency channels to process simultaneously.\n");
    printf("  --num_elements (-e) [number]              Default: 2048. Number of elements to correlate.\n");
    printf("  --timer_without_copies (-w) [number]      Default: off. When on, it separates the copy section from the iterations. \n");
    printf("                                                     Not realistic behaviour, but helpful for timing without using profiler tools.\n");
    printf("  --upper_triangle_convention (-U) [number] Default: 1. (range: [0,1]). 1 uses the standard pairwise correlation convention. 0 does not (i.e. complex conjugate of expected results).\n");
    printf("  --check_results (-c)                      Default: off. Calculates and checks GPU results with CPU calculations.\n");
    printf("  --verbose (-v)                            Default: off. Verbose calculation check. (Dumps all correlation products).\n");
    printf("  --gen_type (-g) [number]                  Default: 4. (1 = Constant, 2 = Ramp up, 3 = Ramp down, 4 = Random (seeded)).\n");
    printf("  --random_seed (-r) [number]               Default: 42. The seed for the pseudorandom generator.\n");
    printf("  --no_repeat_random (-p)                   Default: off. Whether the random sequence repeats at each time step (and frequency channel).\n");
    printf("  --generate_frequency -q [number]          Default: -1 (All frequencies). Other numbers generate non-default values for that frequency channel.\n");
    printf("  --default_real (-x) [number]              Default: 0. (range: [-8, 7]). Only used for higher frequency channels when generate_freq != ALL_FREQUENCIES.\n");
    printf("  --default_imaginary (-y) [number]         Default: 0. (range: [-8, 7]). Only used for higher frequency channels when generate_freq != ALL_FREQUENCIES.\n");
    printf("  --initial_real (-X) [number]              Default: 0. (range: [-8, 7]). Only matters for ramped modes.\n");
    printf("  --initial_imaginary (-Y) [number]         Default: 0. (range: [-8, 7]). Only matters for ramped modes.\n");
    printf("  --kernel_batch (-k) [number]              Default: 0. (0= Kernels from IEEE conference, 1= New more-packed version).\n");
}


int main(int argc, char ** argv) {

    int opt_val = 0;
    //parse entries

    // Default values:
    int device_number = 0;
    int iterations = 100;
    int num_freq = 1;
    int num_elem = 2048;
    int time_accum = 256;
    int time_steps =  256*128; //dependent on the freqs and elements for the one accum phase, and the accum length, num freq, etc... need to keep under 1 GB.
    int timer_without_loop_copying = 0;
    int check_results = 0; //default as 0 because the check is slooowww (at times).
    int verbose = 0;
    int gen_type = GENERATE_DATASET_RANDOM_SEEDED;
    int random_seed = 42;
    int no_repeat_random = 0;
    int generate_frequency = ALL_FREQUENCIES;
    int default_real = 0;
    int default_imaginary = 0;
    int initial_real = 0;
    int initial_imaginary = 0;
    int kernel_batch = 0;
    int T_changed = 0;
    int upper_triangle_convention = 1;

    for (;;) {
        static struct option long_options[] = {
            {"device",              required_argument, 0, 'd'},
            {"iterations",          required_argument, 0, 'i'},
            {"num_freq",            required_argument, 0, 'f'},
            {"num_elem",            required_argument, 0, 'e'},
            {"time_accum",          required_argument, 0, 't'},
            {"time_steps",          required_argument, 0, 'T'},
            {"timer_without_copies",no_argument,       0, 'w'},
            {"upper_triangle_convention", required_argument, 0, 'U'},
            {"check_results",       no_argument,       0, 'c'},
            {"verbose",             no_argument,       0, 'v'},
            {"gen_type",            required_argument, 0, 'g'},
            {"random_seed",         required_argument, 0, 'r'},
            {"no_repeat_random",    no_argument,       0, 'p'},
            {"generate_frequency",  required_argument, 0, 'q'},
            {"default_real",        required_argument, 0, 'x'},
            {"default_imaginary",   required_argument, 0, 'y'},
            {"initial_real",        required_argument, 0, 'X'},
            {"initial_imaginary",   required_argument, 0, 'Y'},
            {"kernel_batch",        required_argument, 0, 'k'},
            {"help",                no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int option_index = 0;

        opt_val = getopt_long (argc, argv, "d:i:f:e:t:T:wcvg:r:pq:x:y:X:Y:hk:U:",
                               long_options, &option_index);

        // End of args
        if (opt_val == -1) {
            break;
        }

        switch (opt_val) {
            case 'h':
                print_help();
                return 0;
                break;
            case 'd':
                device_number = atoi(optarg);
                break;
            case 'i':
                iterations = atoi(optarg);
                break;
            case 'f':
                num_freq = atoi(optarg);
                break;
            case 'e':
                num_elem = atoi(optarg);
                break;
            case 't':
                time_accum = atoi(optarg);
                if (T_changed == 0){
                    time_steps = 128*time_accum;
                }
                break;
            case 'T':
                time_steps = atoi(optarg);
                T_changed = 1;
                break;
            case 'w':
                timer_without_loop_copying =1;
                break;
            case 'c':
                check_results = 1;
                break;
            case 'v':
                verbose = 1;
                check_results = 1;
                break;
            case 'g':
                gen_type = atoi(optarg);
                if (gen_type < 1 || gen_type >>4){
                    printf("Invalid parameter for gen_type.  See help for options\n");
                    print_help();
                    return -1;
                }
                break;
            case 'r':
                random_seed = atoi(optarg);
                break;
            case 'p':
                no_repeat_random = 1;
                break;
            case 'q':
                generate_frequency = atoi(optarg);
                break;
            case 'x':
                default_real = atoi(optarg);
                break;
            case 'y':
                default_imaginary = atoi(optarg);
                break;
            case 'X':
                initial_real = atoi(optarg);
                break;
            case 'Y':
                initial_real = atoi(optarg);
                break;
            case 'k':
                kernel_batch = atoi(optarg);
                if (kernel_batch < 0 || kernel_batch > 1){
                    printf("Invalid parameter for kernel_batch.  See help for options\n");
                    print_help();
                    return -1;
                }
                break;
            case 'u':
                upper_triangle_convention = atoi(optarg);
                if (upper_triangle_convention <0 || upper_triangle_convention > 1){
                    printf("Invalid parameter for upper_triangle_convention.  See help for options\n");
                    print_help();
                    return -1;
                }
                break;
            default:
                printf("Invalid option, run with -h to see options");
                return -1;
                break;
        }
    }

    //end of parsing

    double cputime=0;


    //basic setup of CL devices
    cl_int err;

    // 1. Get a platform.
    cl_platform_id platform;
    clGetPlatformIDs( 1, &platform, NULL );

    // 2. Find a gpu device.
    cl_device_id deviceID[5];

    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 4, deviceID, NULL);

    if (err != CL_SUCCESS){
        printf("Error getting device IDs\n");
        return (-1);
    }
    cl_ulong lm;
    err = clGetDeviceInfo(deviceID[device_number], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lm, NULL);
    if (err != CL_SUCCESS){
        printf("Error getting device info\n");
        return (-1);
    }

    cl_uint mcl,mcm;
    clGetDeviceInfo(deviceID[device_number], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mcl, NULL);
    clGetDeviceInfo(deviceID[device_number], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mcm, NULL);
    float card_tflops = mcl*1e6 * mcm*16*4*2 / 1e12;

    // 3. Create a context and command queues on that device.
    cl_context context = clCreateContext( NULL, 1, &deviceID[device_number], NULL, NULL, NULL);
    cl_command_queue queue[N_QUEUES];
    for (int i = 0; i < N_QUEUES; i++){
        queue[i] = clCreateCommandQueue( context, deviceID[device_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err );
        //queue[i] = clCreateCommandQueue( context, deviceID[device_number], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE , &err );
        if (err){ //success returns a 0
            printf("Error initializing queues.  Exiting program.\n");
            return (-1);
        }

    }

    // 4. Perform runtime source compilation, and obtain kernel entry point.
    int size1_block = 32;
    int num_blocks = (num_elem / size1_block) * (num_elem / size1_block + 1) / 2.; // 256/32 = 8, so 8 * 9/2 (= 36) //needed for the define statement

    // 4a load the source files //this load routine is based off of example code in OpenCL in Action by Matthew Scarpino
    char cl_fileNames[3][256];
    if (upper_triangle_convention == 0){ //original code did the pairwise correlations with a non-standard convention...  The code is retained here, but in general it should be done as in the UT kernels
        if (kernel_batch == 0){
            sprintf(cl_fileNames[0],OPENCL_FILENAME_PACKED1_1);
            sprintf(cl_fileNames[1],OPENCL_FILENAME_PACKED1_2);
            sprintf(cl_fileNames[2],OPENCL_FILENAME_PACKED1_3);
        }
        else if (kernel_batch == 1){
            sprintf(cl_fileNames[0],OPENCL_FILENAME_PACKED2_1);
            sprintf(cl_fileNames[1],OPENCL_FILENAME_PACKED2_2);
            sprintf(cl_fileNames[2],OPENCL_FILENAME_PACKED2_3);
        }
    }
    else{ //UT kernels
        if (kernel_batch == 0){
            sprintf(cl_fileNames[0],OPENCL_FILENAME_PACKED1_UT_1);
            sprintf(cl_fileNames[1],OPENCL_FILENAME_PACKED1_UT_2);
            sprintf(cl_fileNames[2],OPENCL_FILENAME_PACKED1_UT_3);
        }
        else if (kernel_batch == 1){
            sprintf(cl_fileNames[0],OPENCL_FILENAME_PACKED2_UT_1);
            sprintf(cl_fileNames[1],OPENCL_FILENAME_PACKED2_UT_2);
            sprintf(cl_fileNames[2],OPENCL_FILENAME_PACKED2_UT_3);
        }
    }

    printf("Using the following kernels: \n  \"%s\"\n  \"%s\"\n  \"%s\"\n", cl_fileNames[0],cl_fileNames[1],cl_fileNames[2]);

    char cl_options[1024];
    sprintf(cl_options,"-D NUM_ELEMENTS=%du -D NUM_FREQUENCIES=%du -D NUM_BLOCKS=%du -D NUM_TIMESAMPLES=%du -D NUM_TIME_ACCUM=%du -D BASE_ACCUM=%du -D SIZE_PER_SET=%du", num_elem, num_freq, num_blocks, time_steps, time_accum, BASE_TIMESAMPLES_ACCUM,num_blocks*32*32*2*num_freq);
    printf("Dynamic define statements for GPU OpenCL kernels\n");
    printf("-D NUM_ELEMENTS=%du \n-D NUM_FREQUENCIES=%du \n-D NUM_BLOCKS=%du \n-D NUM_TIMESAMPLES=%du\n-D NUM_TIME_ACCUM=%du\n-D BASE_ACCUM=%du\n-D SIZE_PER_SET=%du\n", num_elem, num_freq,num_blocks, time_steps, time_accum, BASE_TIMESAMPLES_ACCUM, num_blocks*32*32*2*num_freq);

    size_t cl_programSize[NUM_CL_FILES];
    FILE *fp;
    char *cl_programBuffer[NUM_CL_FILES];


    for (int i = 0; i < NUM_CL_FILES; i++){
        fp = fopen(cl_fileNames[i], "r");
        if (fp == NULL){
            printf("error loading file: %s\n", cl_fileNames[i]);
            return (-1);
        }
        fseek(fp, 0, SEEK_END);
        cl_programSize[i] = ftell(fp);
        rewind(fp);
        cl_programBuffer[i] = (char*)malloc(cl_programSize[i]+1);
        cl_programBuffer[i][cl_programSize[i]] = '\0';
        int sizeRead = fread(cl_programBuffer[i], sizeof(char), cl_programSize[i], fp);
        if (sizeRead < cl_programSize[i])
            printf("Error reading the file!!!");
        fclose(fp);
    }

    cl_program program = clCreateProgramWithSource( context, NUM_CL_FILES, (const char**)cl_programBuffer, cl_programSize, &err );
    if (err){
        printf("Error in clCreateProgramWithSource: %i\n",err);
        return(-1);
    }

    err = clBuildProgram( program, 1, &deviceID[device_number], cl_options, NULL, NULL );
    if (err){
        printf("Error in clBuildProgram: %i\n",err);
        size_t log_size;
        clGetProgramBuildInfo(program,deviceID[device_number], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log;
        program_log = (char*)malloc(log_size+1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program,deviceID[device_number], CL_PROGRAM_BUILD_LOG, log_size+1,program_log,NULL);
        printf("%s\n",program_log);
        free(program_log);
        return(-1);
    }

    cl_kernel corr_kernel = clCreateKernel( program, "corr", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return (-1);
    }

    cl_kernel offsetAccumulate_kernel = clCreateKernel( program, "offsetAccumulateElements", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return -1;
    }

    cl_kernel preseed_kernel = clCreateKernel( program, "preseed", &err );
    if (err){
        printf("Error in clCreateKernel: %i\n",err);
        return -1;
    }

    for (int i =0; i < NUM_CL_FILES; i++){
        free(cl_programBuffer[i]);
    }

    // 5. set up arrays and initilize if required
    unsigned char *host_PrimaryInput    [N_STAGES]; //where things are brought from, ultimately. Code runs fastest when we create the aligned memory and then pin it to the device
    int *host_PrimaryOutput             [N_STAGES];
    cl_mem device_CLinput_pinnedBuffer  [N_STAGES];
    cl_mem device_CLoutput_pinnedBuffer [N_STAGES];
    cl_mem device_CLinput_kernelData    [N_STAGES];
    cl_mem device_CLoutput_kernelData   [N_STAGES];
    cl_mem device_block_lock;
    cl_mem device_CLoutputAccum         [N_STAGES];


    int len=num_freq*num_blocks*(size1_block*size1_block)*2.;//NUM_TIMESAMPLES/TIME_ACCUM;// *2 because of real and imag
    printf("Num_blocks %d ", num_blocks);
    printf("Output Length %d and size %ld B\n", len, len*sizeof(cl_int));
    cl_int *zeros=calloc(num_blocks*num_freq,sizeof(cl_int)); //for the output buffers
    //printf("zeros %d\n",zeros[num_blocks*num_freq-1]);
    device_block_lock = clCreateBuffer (context,
                                        CL_MEM_COPY_HOST_PTR,
                                        num_blocks*num_freq*sizeof(cl_int),
                                        zeros,
                                        &err);
    free(zeros);

    zeros=calloc(len,sizeof(cl_int)); //for the output buffers

    printf("Size of Data block = %i B\n", time_steps*num_elem*num_freq);
    if (timer_without_loop_copying){
        printf("Setting up and transferring data to GPU...\n");
    }

    // Set up arrays so that they can be used later on
    for (int i = 0; i < N_STAGES; i++){
        //preallocate memory for pinned buffers
        err = posix_memalign ((void **)&host_PrimaryInput[i], PAGESIZE_MEM, time_steps*num_elem*num_freq);
        //check if an extra command is needed to pre pin this--this might just make sure it is
        //aligned in memory space.
        if (err){
            printf("error in creating memory buffers: Inputa, stage: %i, err: %i. Exiting program.\n",i, err);
            return (err);
        }
        err = mlock(host_PrimaryInput[i], time_steps*num_elem*num_freq);
        if (err){
            printf("error in creating memory buffers: Inputb, stage: %i, err: %i. Exiting program.\n",i, err);
            printf("%s",strerror(errno));
            return (err);
        }

        device_CLinput_pinnedBuffer[i] = clCreateBuffer ( context,
                                    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,//
                                    time_steps*num_elem*num_freq,
                                    host_PrimaryInput[i],
                                    &err); //create the clBuffer, using pre-pinned host memory

        if (err){
            printf("error in mapping pin pointers. Exiting program.\n");
            return (err);
        }

        err = posix_memalign ((void **)&host_PrimaryOutput[i], PAGESIZE_MEM, len*sizeof(cl_int));
        err |= mlock(host_PrimaryOutput[i],len*sizeof(cl_int));
        if (err){
            printf("error in creating memory buffers: Output, stage: %i. Exiting program.\n",i);
            return (err);
        }

        device_CLoutput_pinnedBuffer[i] = clCreateBuffer (context,
                                    CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                                    len*sizeof(cl_int),
                                    host_PrimaryOutput[i],
                                    &err); //create the output buffer and allow cl to allocate host memory

        if (err){
            printf("error in mapping pin pointers. Exiting program.\n");
            return (err);
        }


        device_CLinput_kernelData[i] = clCreateBuffer (context,
                                    CL_MEM_READ_ONLY,
                                    time_steps*num_elem*num_freq,
                                    0,
                                    &err); //cl memory that can only be read by kernel

        if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
        }


        device_CLoutput_kernelData[i] = clCreateBuffer (context,
                                    CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
                                    len*sizeof(cl_int),
                                    zeros,
                                    &err); //cl memory that can only be written to by kernel--preset to 0s everywhere

        if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
        }

    } //end for
    free(zeros);

    //initialize an array for the accumulator of offsets (borrowed this buffer from an old version of code--check this)
    zeros=calloc(num_freq*num_elem*2,sizeof(cl_uint));
    device_CLoutputAccum[0] = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          num_freq*num_elem*2*sizeof(cl_uint),
                                          zeros,
                                          &err);
    if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
    }

    device_CLoutputAccum[1] = clCreateBuffer(context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          num_freq*num_elem*2*sizeof(cl_uint),
                                          zeros,
                                          &err);
    if (err){
            printf("error in allocating memory. Exiting program.\n");
            return (err);
    }

    //arrays have been allocated

    //--------------------------------------------------------------
    //Generate Data Set!

    generate_char_data_set(gen_type,
                           random_seed, //random seed
                           default_real,//default_real,
                           default_imaginary,//default_imaginary,
                           initial_real,//initial_real,
                           initial_imaginary,//initial_imaginary,
                           generate_frequency,//int single_frequency,
                           time_steps,//int num_timesteps,
                           num_freq,//int num_frequencies,
                           num_elem,//int num_elements,
                           no_repeat_random,
                           host_PrimaryInput[0]);

    memcpy(host_PrimaryInput[1], host_PrimaryInput[0], time_steps*num_elem*num_freq);

    //--------------------------------------------------------------


    // 6. Set up Kernel parameters

    //upper triangular address mapping --converting 1d addresses to 2d addresses
    unsigned int global_id_x_map[num_blocks];
    unsigned int global_id_y_map[num_blocks];

    int largest_num_blocks_1D = num_elem/size1_block;
    int index_1D = 0;
    for (int j = 0; j < largest_num_blocks_1D; j++){
        for (int i = j; i < largest_num_blocks_1D; i++){
            global_id_x_map[index_1D] = i;
            global_id_y_map[index_1D] = j;
            index_1D++;
        }
    }

    cl_mem id_x_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    num_blocks * sizeof(cl_uint), global_id_x_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    cl_mem id_y_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    num_blocks * sizeof(cl_uint), global_id_y_map, &err);
    if (err){
        printf("Error in clCreateBuffer %i\n", err);
    }

    //set other parameters that will be fixed for the kernels (changeable parameters will be set in run loops)
    clSetKernelArg(corr_kernel, 2, sizeof(void *), (void*) &id_x_map);
    clSetKernelArg(corr_kernel, 3, sizeof(void *), (void*) &id_y_map);
    clSetKernelArg(corr_kernel, 4, sizeof(void *), (void*) &device_block_lock);

    clSetKernelArg(preseed_kernel, 2, sizeof(void *), (void*) &id_x_map);
    clSetKernelArg(preseed_kernel, 3, sizeof(void *), (void*) &id_y_map);
    clSetKernelArg(preseed_kernel, 4, 64* sizeof(cl_uint), NULL);
    clSetKernelArg(preseed_kernel, 5, 64* sizeof(cl_uint), NULL);

    unsigned int n_cAccum=time_steps/time_accum; //n_cAccum == number_of_compressedAccum
    size_t gws_corr[3]={8,8*num_freq,num_blocks*n_cAccum}; //global work size array
    size_t lws_corr[3]={8,8,1}; //local work size array

    size_t gws_accum[3]={64, (int)ceil(num_elem*num_freq/256.0),time_steps/BASE_TIMESAMPLES_ACCUM};
    size_t lws_accum[3]={64, 1, 1};

    size_t gws_preseed[3]={8, 8*num_freq, num_blocks};
    size_t lws_preseed[3]={8, 8, 1};

    //setup and start loop to process data in parallel
    int spinCount = 0; //we rotate through values to launch processes in order for the command queues. This helps keep track of what position, and can run indefinitely without overflow
    int writeToDevStageIndex;
    int kernelStageIndex;
    //int readFromDevStageIndex;

    cl_int numWaitEventWrite = 0;
    cl_event* eventWaitPtr = NULL;

    cl_event lastWriteEvent[N_STAGES]  = { 0 }; // All entries initialized to 0, since unspecified entries are set to 0
    cl_event lastKernelEvent[N_STAGES] = { 0 };
    cl_event copyInputDataEvent;
    cl_event offsetAccumulateEvent;
    cl_event preseedEvent;

    if (timer_without_loop_copying){
        for (int i = 0; i < N_STAGES; i++){
             err = clEnqueueWriteBuffer(queue[0],
                                    device_CLinput_kernelData[i], //to here
                                    CL_TRUE,
                                    0, //offset
                                    time_steps * num_elem*num_freq, //8 for multifreq interleaving
                                    host_PrimaryInput[i], //from here
                                    0,
                                    NULL,
                                    NULL);
            if (err){
                printf("Error in transfer to device memory. Error in loop %d, error: %s\n",i,oclGetOpenCLErrorCodeStr(err));
                exit(err);
            }
        }
        clFinish(queue[0]);
        printf("Data transfer to GPU complete\n");
    }

    printf("Running %i iterations of full corr (%i time samples (%i Ki time samples), %i elements, %i frequencies)\n", iterations, time_steps, time_steps/1024, num_elem, num_freq);

    //note that releasing events (while preventing memory leaks) can cause havoc on the CodeXL profiler--it needs the events for its analysis--if things act weird in CodeXL, this is a place to look
    ///////////////////////////////////////////////////////////////////////////////
    cputime = e_time();
    for (int i=0; i<=iterations; i++){//if we were truly streaming data, for each correlation, we would need to change what arrays are used for input/output
        writeToDevStageIndex =  (spinCount ); // + 0) % N_STAGES;
        kernelStageIndex =      (spinCount + 1 ) % N_STAGES; //had been + 2 when it was 3 stages

        //transfer section
        if (i < iterations){ //Start at 0, Stop before the last loop
            //check if it needs to wait on anything
            if(lastKernelEvent[writeToDevStageIndex] != 0){ //only equals 0 when it hasn't yet been defined i.e. the first run through the loop with N_STAGES == 2
                numWaitEventWrite = 1;
                eventWaitPtr = &lastKernelEvent[writeToDevStageIndex]; //writes must wait on the last kernel operation since
            }
            else {
                numWaitEventWrite = 0;
                eventWaitPtr = NULL;
                }

            //copy necessary buffers to device memory
            if (timer_without_loop_copying){
                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLoutputAccum[writeToDevStageIndex],
                                        CL_FALSE,
                                        0,
                                        num_freq*num_elem*2*sizeof(cl_int),
                                        zeros,
                                        numWaitEventWrite,
                                        eventWaitPtr,
                                        &lastWriteEvent[writeToDevStageIndex]);
                if (err){
                    printf("Error in transfer to device memory. Error in loop %d\n",i);
                    exit(err);
                }
                if (eventWaitPtr != NULL)
                    clReleaseEvent(*eventWaitPtr);
                //err = clFlush(queue[0]);
                if (err){
                    printf("Error in flushing transfer to device memory. Error in loop %d\n",i);
                    exit(err);
                }

            }
            else{
                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLinput_kernelData[writeToDevStageIndex], //to here
                                        CL_FALSE,
                                        0, //offset
                                        time_steps * num_elem*num_freq, //8 for multifreq interleaving
                                        host_PrimaryInput[writeToDevStageIndex], //from here
                                        numWaitEventWrite,
                                        eventWaitPtr,
                                        &copyInputDataEvent);
                if (err){
                    printf("Error in transfer to device memory. Error in loop %d, error: %s\n",i,oclGetOpenCLErrorCodeStr(err));
                    exit(err);
                }
                if (eventWaitPtr != NULL)
                    clReleaseEvent(*eventWaitPtr);

                err = clEnqueueWriteBuffer(queue[0],
                                        device_CLoutputAccum[writeToDevStageIndex],
                                        CL_FALSE,
                                        0,
                                        num_freq*num_elem*2*sizeof(cl_int),
                                        zeros,
                                        1,
                                        &copyInputDataEvent,
                                        &lastWriteEvent[writeToDevStageIndex]);

                clReleaseEvent(copyInputDataEvent);
                //err = clFlush(queue[0]);
                if (err){
                    printf("Error in flushing transfer to device memory. Error in loop %d\n",i);
                    exit(err);
                }
            }
        }

        //processing section
        if (lastWriteEvent[kernelStageIndex] !=0 && i <= iterations){//insert additional steps for processing here
            //accumulateFeeds_kernel--set 2 arguments--input array and zeroed output array
            err = clSetKernelArg(offsetAccumulate_kernel,
                                 0,
                                 sizeof(void*),
                                 (void*) &device_CLinput_kernelData[kernelStageIndex]);

            err |= clSetKernelArg(offsetAccumulate_kernel,
                                  1,
                                  sizeof(void *),
                                  (void *) &device_CLoutputAccum[kernelStageIndex]); //make sure this array is zeroed initially!
            if (err){
                printf("Error setting the kernel 0 arguments in loop %d\n", i);
                exit(err);
            }


            err = clEnqueueNDRangeKernel(queue[1],
                                         offsetAccumulate_kernel,
                                         3,
                                         NULL,
                                         gws_accum,
                                         lws_accum,
                                         1,
                                         &lastWriteEvent[kernelStageIndex], /* make sure data is present, first*/
                                         &offsetAccumulateEvent);
            if (err){
                printf("Error accumulating in loop %d\n", i);
                exit(err);
            }
            clReleaseEvent(lastWriteEvent[kernelStageIndex]);
            //preseed_kernel--set only 2 of the 6 arguments (the other 4 stay the same)
            err = clSetKernelArg(preseed_kernel,
                                 0,
                                 sizeof(void *),
                                 (void *) &device_CLoutputAccum[kernelStageIndex]);//assign the accumulated data as input

            err = clSetKernelArg(preseed_kernel,
                                 1,
                                 sizeof(void *),
                                 (void *) &device_CLoutput_kernelData[kernelStageIndex]); //set the output for preseeding the correlator array

            err = clEnqueueNDRangeKernel(queue[1],
                                         preseed_kernel,
                                         3, //3d global dimension, also worksize
                                         NULL, //no offsets
                                         gws_preseed,
                                         lws_preseed,
                                         1,
                                         &offsetAccumulateEvent,/*dependent on previous step so don't use &lastWriteEvent[kernelStageIndex],*/
                                         &preseedEvent);
            if (err){
                printf("Error performing preseed kernel operation in loop %d: error %d\n", i,err);
                exit(err);
            }
            clReleaseEvent(offsetAccumulateEvent);
            //corr_kernel--set the input and output buffers (the other parameters stay the same).
            err =  clSetKernelArg(corr_kernel,
                                    0,
                                    sizeof(void *),
                                    (void*) &device_CLinput_kernelData[kernelStageIndex]);
            if (err){
                printf("Error setting the kernel 0 arguments in loop %d\n", i);
                exit(err);
            }
            err = clSetKernelArg(corr_kernel,
                                    1,
                                    sizeof(void *),
                                    (void*) &device_CLoutput_kernelData[kernelStageIndex]);
            if (err){
                printf("Error setting the kernel 1 arguments in loop %d\n", i);
                exit(err);
            }

            err = clEnqueueNDRangeKernel(queue[1],
                                         corr_kernel,
                                         3, //3d global dimension, also worksize
                                         NULL, //no offsets
                                         gws_corr,
                                         lws_corr,
                                         1,
                                         &preseedEvent,/*dependent on previous step so don't use &lastWriteEvent[kernelStageIndex],*/
                                         &lastKernelEvent[kernelStageIndex]);
            if (err){
                printf("Error performing corr kernel operation in loop %d, err: %d\n", i,err);
                exit(err);
            }
            clReleaseEvent(preseedEvent);


        }

        spinCount++;
        spinCount = (spinCount < N_STAGES) ? spinCount : 0; //keeps the value of spinCount small, always, and then saves 1 remainder calculation earlier in the loop.
    }

    //since there are only 2, simplify things (i.e. no need for a loop).
    err =  clFinish(queue[0]);
    err |= clFinish(queue[1]);

    if (err){
        printf("Error while finishing up the queue after the loops.\n");
        return (err);
    }
    cputime = e_time()-cputime;

    // 7. Look at the results
    err = clEnqueueReadBuffer(queue[0], device_CLoutput_kernelData[0], CL_TRUE, 0, len*sizeof(cl_int), host_PrimaryOutput[0], 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue[0], device_CLoutput_kernelData[1], CL_TRUE, 0, len*sizeof(cl_int), host_PrimaryOutput[1], 0, NULL, NULL);

    if (err){
        printf("Error reading data back to host.\n");
        //return (err);
    }

    err = clFinish(queue[0]);

    if (err){
        printf("Error while finishing up the queue after the loops.\n");
        //return (err);
    }




    if (iterations > 1){
        for (int i = 0; i < len; i++){
            host_PrimaryOutput[0][i] += host_PrimaryOutput[1][i];
            host_PrimaryOutput[0][i] /=2; //the results in output 0 and 1 should be identical--this is just to check (in a rough way) that they are.
            //if the average of the two arrays is the correct answer, and one expects both of them to have an answer, then the answers of both should be correct
            //-could skip this and just compare against the first array's results
        }
    }

    //--------------------------------------------------------------

    printf("Correlation matrices computation time: %6.4fs on GPU (%.1f kHz of 400 MHz band, or %.1fx10^3 correlation matrices/s)\n",cputime,time_steps*num_freq/cputime/1000*iterations,time_steps*num_freq/cputime/1000*iterations);
    printf("    [Theoretical max: @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n", card_tflops,
                                    card_tflops*1e12 / (num_elem/2.*(num_elem+1.) * 2. * 2.) / 1e3,
                                    100.*iterations*time_steps/cputime / (card_tflops*1e12) * num_elem/2.*(num_elem+1.) * 2. * 2.*num_freq );
    printf("    [Algorithm max:   @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n", card_tflops,
                                    card_tflops*1e12 / (num_blocks * size1_block * size1_block * 2. * 2.) / 1e3,
                                    100.*iterations*time_steps/cputime / (card_tflops*1e12) * num_blocks * size1_block * size1_block * 2. * 2.*num_freq);


    if (check_results){
        printf("Checking results. Please wait...\n");
        // start using calls to do the comparisons
        cputime = e_time();
        int *correlated_CPU = calloc((num_elem*(num_elem))*num_freq*2,sizeof(int)); //made for the largest possible size (one size fits all)
        if (correlated_CPU == NULL){
            printf("failed to allocate memory\n");
            return(-1);
        }

        if (upper_triangle_convention == 0){
            if (TRIANGLE){
                err = cpu_data_generate_and_correlate_upper_triangle_only_nonstandard_convention(time_steps, num_freq, num_elem, correlated_CPU,gen_type, random_seed, default_real, default_imaginary, initial_real, initial_imaginary,generate_frequency, no_repeat_random,verbose);
            }
            else{
                err = cpu_data_generate_and_correlate_nonstandard_convention(time_steps, num_freq, num_elem, correlated_CPU,gen_type, random_seed, default_real, default_imaginary, initial_real, initial_imaginary,generate_frequency, no_repeat_random,verbose);
            }
        }
        else{
            if (TRIANGLE){
                err = cpu_data_generate_and_correlate_upper_triangle_only(time_steps, num_freq, num_elem, correlated_CPU,gen_type, random_seed, default_real, default_imaginary, initial_real, initial_imaginary,generate_frequency, no_repeat_random,verbose);
            }
            else{
                err = cpu_data_generate_and_correlate(time_steps, num_freq, num_elem, correlated_CPU,gen_type, random_seed, default_real, default_imaginary, initial_real, initial_imaginary,generate_frequency, no_repeat_random,verbose);
            }
        }

        int *correlated_GPU = (int *)malloc((num_elem*(num_elem))*num_freq*2*sizeof(int));

        if (correlated_GPU == NULL){
            printf("failed to allocate memory\n");
            return(-1);
        }


        if (TRIANGLE){
            reorganize_GPU_to_upper_triangle(size1_block, num_blocks, num_freq, num_elem, host_PrimaryOutput[0], correlated_GPU);
        }
        else{
            reorganize_GPU_to_full_Matrix_for_comparison(size1_block, num_blocks, num_freq, num_elem, host_PrimaryOutput[0], correlated_GPU);
        }

        int number_errors = 0;
        int64_t errors_squared;
        double *amp2_ratio_GPU_div_CPU = (double *)malloc(num_elem*num_elem*num_freq*sizeof(double));
        if (amp2_ratio_GPU_div_CPU == NULL){
            printf("ran out of memory\n");
            return (-1);
        }
        double *phaseAngleDiff_GPU_m_CPU = (double *)malloc(num_elem*num_elem*num_freq*sizeof(double));
        if (phaseAngleDiff_GPU_m_CPU == NULL){
            printf("2ran out of memory\n");
            return (-1);
        }

        if (TRIANGLE){
            compare_NSquared_correlator_results_data_has_upper_triangle_only ( &number_errors, &errors_squared, num_freq, num_elem, correlated_GPU, correlated_CPU, amp2_ratio_GPU_div_CPU, phaseAngleDiff_GPU_m_CPU, verbose);
        }
        else{
            compare_NSquared_correlator_results ( &number_errors, &errors_squared, num_freq, num_elem, correlated_GPU, correlated_CPU, amp2_ratio_GPU_div_CPU, phaseAngleDiff_GPU_m_CPU, verbose);
        }

        if (number_errors > 0)
            printf("Error with correlation/accumulation! Num Err: %d and length of correlated data: %d\n",number_errors, num_elem*num_elem*num_freq);
        else
            printf("Correlation/accumulation successful! CPU matches GPU.\n");
        cputime=e_time()-cputime;
        printf("Full Corr: %4.2fs on CPU (%.2f kHz)\n",cputime,time_steps/cputime/1e3);

        free(correlated_CPU);
        free(correlated_GPU);
        free(amp2_ratio_GPU_div_CPU);
        free(phaseAngleDiff_GPU_m_CPU);
    }
    else{
        printf("\nGPU calculations have not been verified. If kernels have been changed, be careful regarding these results.\n\n");
    }
    err = munlockall();


    if (err != SDK_SUCCESS) {
        printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
        printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
        exit(err);
    }

    for (int ns=0; ns < N_STAGES; ns++){
        err = clReleaseMemObject(device_CLinput_pinnedBuffer[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject( device_CLoutput_pinnedBuffer[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject(device_CLinput_kernelData[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        err = clReleaseMemObject(device_CLoutput_kernelData[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
        assert(host_PrimaryInput[ns]!=NULL);
        free(host_PrimaryInput[ns]);
        free(host_PrimaryOutput[ns]);

        err = clReleaseMemObject(device_CLoutputAccum[ns]);
        if (err != SDK_SUCCESS) {
            printf("clReleaseMemObject() failed with %d (%s)\n",err,oclGetOpenCLErrorCodeStr(err));
            printf("Error at line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(err);
        }
    }

    free(zeros);

    //--------------------------------------------------------------

    clReleaseKernel(corr_kernel);
    clReleaseProgram(program);
    clReleaseMemObject(device_block_lock);
    clReleaseMemObject(id_x_map);
    clReleaseMemObject(id_y_map);
    clReleaseCommandQueue(queue[0]);
    clReleaseCommandQueue(queue[1]);
    clReleaseContext(context);
    return 0;
}
