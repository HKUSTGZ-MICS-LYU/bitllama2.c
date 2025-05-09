ARCHS=rv32ic rv32imc
# ifneq ($(RV32),no)
# ifneq ($(NEED_FPU),yes)
# ARCHS+=rv32i rv32ima rv32imac
# endif
# ARCHS+=rv32imaf rv32imafc rv32imafd rv32imafdc
# endif

# ifneq ($(RV64),no)
# ifneq ($(NEED_FPU),yes)
# ARCHS+=rv64i rv64ima rv64imac
# endif
# ARCHS+=rv64imaf rv64imafc rv64imafd rv64imafdc
# endif

rv32ic=MARCH=rv32ic MABI=ilp32 USE_SIMD=0
rv32imc=MARCH=rv32imc MABI=ilp32 USE_SIMD=0

# rv32i=MARCH=rv32i MABI=ilp32
# rv32ima=MARCH=rv32ima MABI=ilp32
# rv32imac=MARCH=rv32imac MABI=ilp32
# rv32imaf=MARCH=rv32imaf MABI=ilp32f
# rv32imafc=MARCH=rv32imafc MABI=ilp32f
# rv32imafd=MARCH=rv32imafd MABI=ilp32d
# rv32imafdc=MARCH=rv32imafdc MABI=ilp32d
# rv64i=MARCH=rv64i MABI=lp64
# rv64ima=MARCH=rv64ima MABI=lp64
# rv64imac=MARCH=rv64imac MABI=lp64
# rv64imaf=MARCH=rv64imaf MABI=lp64f
# rv64imafc=MARCH=rv64imafc MABI=lp64f
# rv64imafd=MARCH=rv64imafd MABI=lp64d
# rv64imafdc=MARCH=rv64imafdc MABI=lp64d

