#pragma once
#include <hip/hip_complex.h>

typedef hipFloatComplex cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;
typedef hipComplex cuComplex;

#define make_cuFloatComplex make_hipFloatComplex
#define make_cuDoubleComplex make_hipDoubleComplex
#define make_cuComplex make_hipComplex

#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuConjf hipConjf
#define cuCsqabsf hipCsqabsf
#define cuCaddf hipCaddf
#define cuCsubf hipCsubf
#define cuCmulf hipCmulf
#define cuCdivf hipCdivf
#define cuCabsf hipCabsf

#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuConj hipConj
#define cuCsqabs hipCsqabs
#define cuCadd hipCadd
#define cuCsub hipCsub
#define cuCmul hipCmul
#define cuCdiv hipCdiv
#define cuCabs hipCabs

#define cuComplexDoubleToFloat hipComplexDoubleToFloat
#define cuComplexFloatToDouble hipComplexFloatToDouble
#define cuCfmaf hipCfmaf
#define cuCfma hipCfma
