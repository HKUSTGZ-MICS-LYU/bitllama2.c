#include "custom_asm.h"

// For BitNet SIMD Above 8 elements, the arguments are the data pointers
.global __bitnetadd16_unroll
__bitnetadd16_unroll:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, x0, a1) // BNSTORE a1 -> buffer
    li a0, 0 // result buffer
    lw t0, 0(a4)
    lw t1, 4(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 8(a4)
    lw t1, 12(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    ret

.global __bitnetadd32_unroll
__bitnetadd32_unroll:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, a2, a1) // BNSTORE a1 -> buffer
    li a0, 0 // result buffer
    lw t0, 0(a4)
    lw t1, 4(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 8(a4)
    lw t1, 12(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 16(a4)
    lw t1, 20(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 24(a4)
    lw t1, 28(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    ret

// Available for 1-bit BitNet Only
.global __bitnetadd64_unroll
__bitnetadd64_unroll:
    opcode_R(CUSTOM0, 0x2, 0x00, x0, a2, a1) // BNSTORE a1 -> buffer
    li a0, 0 // result buffer
    lw t0, 0(a4)
    lw t1, 4(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 8(a4)
    lw t1, 12(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 16(a4)
    lw t1, 20(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 24(a4)
    lw t1, 28(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 32(a4)
    lw t1, 36(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 40(a4)
    lw t1, 44(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 48(a4)
    lw t1, 52(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    lw t0, 56(a4)
    lw t1, 60(a4)
    opcode_R(CUSTOM0, 0x1, 0x00, t2, t0, t1) // BNADD8 a0
    add a0, a0, t2
    ret