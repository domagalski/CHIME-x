
CC	= gcc
OPTIMIZE	= -Wall -O4 -std=gnu99 -msse3 -ggdb
INC	= -I$(AMDAPPSDKROOT)/include -I$(AMDAPPSDKROOT)/include/CAL
LIBS	= -lOpenCL -lm -L$(AMDAPPSDKROOT)/lib/x86_64/
CFLAGS	= $(OPTIMIZE) $(INC)
SOURCES	=main_wrapper.c amd_firepro_error_code_list_for_opencl.c input_generator.c gpu_data_reorg.c gpu_cpu_helpers.c cpu_corr_test.c
OBJECTS	=$(SOURCES:.c=.o)
EXECUTABLE=correlator_test

all: $(SOURCES) $(EXECUTABLE)


$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LIBS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o *~ $(EXECUTABLE)
