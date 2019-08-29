CFLAGS = -std=c99 -Wall
OPTIMIZATIONS = -O3 -xHost
LDFLAGS= -lOpenCL -lm

d2q9-bgk:
	mpiicc $(CFLAGS) $(OPTIMIZATIONS) d2q9-bgk.c -o d2q9-bgk $(LDFLAGS)
