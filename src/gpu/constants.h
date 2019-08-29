#pragma once

// ======= CONSTANTS ===================================================================
#define NSPEEDS 9
#define HALOS 2
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define MASTER 0
#define IN_DEGREE 2
#define OUT_DEGREE 2

// ======= COLLISION CONSTANTS ===================================================================
#define C_SQ       (1.f / 3.f)                          /* square of speed of sound */
#define INV_C_SQ   (1.0f / C_SQ)
#define INV_2C_SQ  (1.0f / (2.f * C_SQ))
#define INV_2C_SQ2 (1.0f / (2.f * C_SQ * C_SQ))
#define W0         (4.f / 9.f)                          /* weighting factor */
#define W1         (1.f / 9.f)                          /* weighting factor */
#define W2         (1.f / 36.f)                         /* weighting factor */