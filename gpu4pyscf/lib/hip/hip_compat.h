/*
 * Copyright 2024-2026 The PySCF Developers and Gallicchio-Lab.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

#ifndef __HIP_CPU__
  #if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
    #define FULL_WARP_MASK 0xffffffffffffffffULL
    typedef unsigned long long warp_mask_t;
  #else
    #define FULL_WARP_MASK 0xffffffffU
    typedef unsigned int warp_mask_t;
  #endif
#else
  #define FULL_WARP_MASK 0xffffffffU
  typedef unsigned int warp_mask_t;
#endif

#ifdef __HIPCC__
template <typename T>
__device__ inline T __shfl_down_sync(unsigned long long mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_down(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_down_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_down(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_down_sync(int mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_down(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_sync(unsigned long long mask, T var, int srcLane, int width = warpSize) {
    return ::__shfl(var, srcLane, width);
}
template <typename T>
__device__ inline T __shfl_sync(unsigned int mask, T var, int srcLane, int width = warpSize) {
    return ::__shfl(var, srcLane, width);
}
template <typename T>
__device__ inline T __shfl_sync(int mask, T var, int srcLane, int width = warpSize) {
    return ::__shfl(var, srcLane, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(unsigned long long mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_up(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(unsigned int mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_up(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(int mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_up(var, delta, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned long long mask, T var, int laneMask, int width = warpSize) {
    return ::__shfl_xor(var, laneMask, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned int mask, T var, int laneMask, int width = warpSize) {
    return ::__shfl_xor(var, laneMask, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(int mask, T var, int laneMask, int width = warpSize) {
    return ::__shfl_xor(var, laneMask, width);
}
__device__ inline unsigned int __ballot_sync(unsigned long long mask, int predicate) {
    return ::__ballot(predicate);
}
__device__ inline unsigned int __ballot_sync(unsigned int mask, int predicate) {
    return ::__ballot(predicate);
}
__device__ inline unsigned int __ballot_sync(int mask, int predicate) {
    return ::__ballot(predicate);
}
__device__ inline int __any_sync(unsigned long long mask, int predicate) {
    return ::__any(predicate);
}
__device__ inline int __any_sync(unsigned int mask, int predicate) {
    return ::__any(predicate);
}
__device__ inline int __any_sync(int mask, int predicate) {
    return ::__any(predicate);
}
__device__ inline int __all_sync(unsigned long long mask, int predicate) {
    return ::__all(predicate);
}
__device__ inline int __all_sync(unsigned int mask, int predicate) {
    return ::__all(predicate);
}
__device__ inline int __all_sync(int mask, int predicate) {
    return ::__all(predicate);
}
#endif

// CUDA to HIP aliases for common types & functions
#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif

#ifndef cudaError_t
#define cudaError_t hipError_t
#endif

#ifndef cudaGetLastError
#define cudaGetLastError hipGetLastError
#endif

#ifndef cudaPeekAtLastError
#define cudaPeekAtLastError hipPeekAtLastError
#endif

#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif

#ifndef cudaGetErrorName
#define cudaGetErrorName hipGetErrorName
#endif

#ifndef cudaDeviceReset
#define cudaDeviceReset hipDeviceReset
#endif

#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif

#ifndef cudaStreamDefault
#define cudaStreamDefault hipStreamDefault
#endif

#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking hipStreamNonBlocking
#endif

#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#endif

#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif

#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#endif

#ifndef cudaMemcpy
#define cudaMemcpy hipMemcpy
#endif

#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif

#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif

#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif

#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

#ifndef cudaMemcpyDefault
#define cudaMemcpyDefault hipMemcpyDefault
#endif

#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif

#ifndef cudaFree
#define cudaFree hipFree
#endif

#ifndef cudaMemset
#define cudaMemset hipMemset
#endif

#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif

#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif

#ifndef cudaSetDevice
#define cudaSetDevice hipSetDevice
#endif

#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount hipGetDeviceCount
#endif

#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#endif

#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer(pdev, phst, flags) hipHostGetDevicePointer((void**)(pdev), (void*)(phst), flags)
#endif

#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif

#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif

#ifndef cudaMemcpyToSymbol
#define cudaMemcpyToSymbol(sym, ...) hipMemcpyToSymbol(HIP_SYMBOL(sym), __VA_ARGS__)
#endif

#ifndef CUDA_VERSION
#define CUDA_VERSION 12000
#endif

#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute(func, attr, val) hipFuncSetAttribute((const void*)(func), (hipFuncAttribute)(attr), val)
#endif

#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#endif

#ifndef cudaFuncAttributes
#define cudaFuncAttributes hipFuncAttributes
#endif

#ifndef cudaFuncGetAttributes
#define cudaFuncGetAttributes(attr, func) hipFuncGetAttributes(attr, (const void*)(func))
#endif

#ifndef cudaGetSymbolAddress
#define cudaGetSymbolAddress(devPtr, sym) hipGetSymbolAddress((void**)(devPtr), HIP_SYMBOL(sym))
#endif

#ifndef register
#define register
#endif

#ifdef I
#undef I
#endif

#else // Native CUDA

#include <cuda_runtime.h>
#include <cuComplex.h>

#define FULL_WARP_MASK 0xffffffffU
typedef unsigned int warp_mask_t;

#endif
