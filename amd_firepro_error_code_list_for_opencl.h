/**
* @file amd_firepro_error_code_list_for_opencl.h
* @brief changes an error code into an error message that is human-readable/understandable
**/

#ifndef AMD_FIREPRO_ERROR_CODE_LIST_FOR_OPENCL_H
#define AMD_FIREPRO_ERROR_CODE_LIST_FOR_OPENCL_H

#include <CL/cl.h>
#include <CL/cl_ext.h>

char* oclGetOpenCLErrorCodeStr(cl_int input);

#endif
