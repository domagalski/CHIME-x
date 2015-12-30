//local_macros.h
//commonly used macros with packed 4-bit data

#ifndef FOUR_BIT_MACROS_H
#define FOUR_BIT_MACROS_H

//macros to return the low and high 4 bits of a 1 B char
#define HI_NIBBLE(b)                    (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b)                    ((b) & 0x0F)

#endif
