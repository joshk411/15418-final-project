#include "kernel_helpers.cuh"
#include <cstdlib>
#include <stdio.h>

__global__ void _kernel1_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus4 = modulii[4];
	auto barrett_k4 = barrett_k[4];
	auto barrett_ratio4 = barrett_ratios[4];
	auto in4_0 = args_inputs[2];
	auto in4_1 = args_inputs[3];
	auto out4 = args_outputs[1];
	auto modulus7 = modulii[7];
	auto barrett_k7 = barrett_k[7];
	auto barrett_ratio7 = barrett_ratios[7];
	auto in7_0 = args_inputs[4];
	auto in7_1 = args_inputs[5];
	auto out7 = args_outputs[2];
	auto modulus10 = modulii[10];
	auto barrett_k10 = barrett_k[10];
	auto barrett_ratio10 = barrett_ratios[10];
	auto in10_0 = args_inputs[6];
	auto in10_1 = args_inputs[7];
	auto out10 = args_outputs[3];
	auto modulus13 = modulii[13];
	auto barrett_k13 = barrett_k[13];
	auto barrett_ratio13 = barrett_ratios[13];
	auto in13_0 = args_inputs[8];
	auto in13_1 = args_inputs[9];
	auto out13 = args_outputs[4];
	auto modulus16 = modulii[16];
	auto barrett_k16 = barrett_k[16];
	auto barrett_ratio16 = barrett_ratios[16];
	auto in16_0 = args_inputs[10];
	auto in16_1 = args_inputs[11];
	auto out16 = args_outputs[5];
	auto modulus19 = modulii[19];
	auto barrett_k19 = barrett_k[19];
	auto barrett_ratio19 = barrett_ratios[19];
	auto in19_0 = args_inputs[12];
	auto in19_1 = args_inputs[13];
	auto out19 = args_outputs[6];
	auto modulus22 = modulii[22];
	auto barrett_k22 = barrett_k[22];
	auto barrett_ratio22 = barrett_ratios[22];
	auto in22_0 = args_inputs[14];
	auto in22_1 = args_inputs[15];
	auto out22 = args_outputs[7];
	auto modulus25 = modulii[25];
	auto barrett_k25 = barrett_k[25];
	auto barrett_ratio25 = barrett_ratios[25];
	auto in25_0 = args_inputs[16];
	auto in25_1 = args_inputs[17];
	auto out25 = args_outputs[8];
	auto modulus28 = modulii[28];
	auto barrett_k28 = barrett_k[28];
	auto barrett_ratio28 = barrett_ratios[28];
	auto in28_0 = args_inputs[18];
	auto in28_1 = args_inputs[19];
	auto out28 = args_outputs[9];
	auto modulus29 = modulii[29];
	auto barrett_k29 = barrett_k[29];
	auto barrett_ratio29 = barrett_ratios[29];
	auto in29_0 = args_inputs[20];
	auto in29_1 = args_inputs[21];
	auto out29 = args_outputs[10];
	auto modulus30 = modulii[30];
	auto barrett_k30 = barrett_k[30];
	auto barrett_ratio30 = barrett_ratios[30];
	auto in30_0 = args_inputs[22];
	auto in30_1 = args_inputs[23];
	auto out30 = args_outputs[11];
	auto modulus31 = modulii[31];
	auto barrett_k31 = barrett_k[31];
	auto barrett_ratio31 = barrett_ratios[31];
	auto in31_0 = args_inputs[24];
	auto in31_1 = args_inputs[25];
	auto out31 = args_outputs[12];
	auto modulus32 = modulii[32];
	auto barrett_k32 = barrett_k[32];
	auto barrett_ratio32 = barrett_ratios[32];
	auto in32_0 = args_inputs[26];
	auto in32_1 = args_inputs[27];
	auto out32 = args_outputs[13];
	auto modulus33 = modulii[33];
	auto barrett_k33 = barrett_k[33];
	auto barrett_ratio33 = barrett_ratios[33];
	auto in33_0 = args_inputs[28];
	auto in33_1 = args_inputs[29];
	auto out33 = args_outputs[14];
	auto modulus34 = modulii[34];
	auto barrett_k34 = barrett_k[34];
	auto barrett_ratio34 = barrett_ratios[34];
	auto in34_0 = args_inputs[30];
	auto in34_1 = args_inputs[31];
	auto out34 = args_outputs[15];
	auto modulus35 = modulii[35];
	auto barrett_k35 = barrett_k[35];
	auto barrett_ratio35 = barrett_ratios[35];
	auto in35_0 = args_inputs[32];
	auto in35_1 = args_inputs[33];
	auto out35 = args_outputs[16];
	auto modulus36 = modulii[36];
	auto barrett_k36 = barrett_k[36];
	auto barrett_ratio36 = barrett_ratios[36];
	auto in36_0 = args_inputs[34];
	auto in36_1 = args_inputs[35];
	auto out36 = args_outputs[17];
	auto modulus37 = modulii[37];
	auto barrett_k37 = barrett_k[37];
	auto barrett_ratio37 = barrett_ratios[37];
	auto in37_0 = args_inputs[36];
	auto in37_1 = args_inputs[37];
	auto out37 = args_outputs[18];
	auto modulus38 = modulii[38];
	auto barrett_k38 = barrett_k[38];
	auto barrett_ratio38 = barrett_ratios[38];
	auto in38_0 = args_inputs[38];
	auto in38_1 = args_inputs[39];
	auto out38 = args_outputs[19];
	auto modulus39 = modulii[39];
	auto barrett_k39 = barrett_k[39];
	auto barrett_ratio39 = barrett_ratios[39];
	auto in39_0 = args_inputs[40];
	auto in39_1 = args_inputs[41];
	auto out39 = args_outputs[20];
	auto modulus40 = modulii[40];
	auto barrett_k40 = barrett_k[40];
	auto barrett_ratio40 = barrett_ratios[40];
	auto in40_0 = args_inputs[42];
	auto in40_1 = args_inputs[43];
	auto out40 = args_outputs[21];
	auto modulus41 = modulii[41];
	auto barrett_k41 = barrett_k[41];
	auto barrett_ratio41 = barrett_ratios[41];
	auto in41_0 = args_inputs[44];
	auto in41_1 = args_inputs[45];
	auto out41 = args_outputs[22];
	auto modulus42 = modulii[42];
	auto barrett_k42 = barrett_k[42];
	auto barrett_ratio42 = barrett_ratios[42];
	auto in42_0 = args_inputs[46];
	auto in42_1 = args_inputs[47];
	auto out42 = args_outputs[23];
	auto modulus43 = modulii[43];
	auto barrett_k43 = barrett_k[43];
	auto barrett_ratio43 = barrett_ratios[43];
	auto in43_0 = args_inputs[48];
	auto in43_1 = args_inputs[49];
	auto out43 = args_outputs[24];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r2: r0, r1 | 4
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r5: r3, r4 | 3
		out4[i] = multiply(in4_0[i],in4_1[i],modulus4,barrett_ratio4,barrett_k4);
		// mul: r8: r6, r7 | 2
		out7[i] = multiply(in7_0[i],in7_1[i],modulus7,barrett_ratio7,barrett_k7);
		// mul: r11: r9, r10 | 1
		out10[i] = multiply(in10_0[i],in10_1[i],modulus10,barrett_ratio10,barrett_k10);
		// mul: r14: r12, r13 | 0
		out13[i] = multiply(in13_0[i],in13_1[i],modulus13,barrett_ratio13,barrett_k13);
		// mul: r17: r15, r16 | 4
		out16[i] = multiply(in16_0[i],in16_1[i],modulus16,barrett_ratio16,barrett_k16);
		// mul: r20: r18, r19 | 3
		out19[i] = multiply(in19_0[i],in19_1[i],modulus19,barrett_ratio19,barrett_k19);
		// mul: r23: r21, r22 | 2
		out22[i] = multiply(in22_0[i],in22_1[i],modulus22,barrett_ratio22,barrett_k22);
		// mul: r26: r24, r25 | 1
		out25[i] = multiply(in25_0[i],in25_1[i],modulus25,barrett_ratio25,barrett_k25);
		// mul: r29: r27, r28 | 0
		out28[i] = multiply(in28_0[i],in28_1[i],modulus28,barrett_ratio28,barrett_k28);
		// mul: r30: r15[X], r1[X] | 4
		out29[i] = multiply(in29_0[i],in29_1[i],modulus29,barrett_ratio29,barrett_k29);
		// mul: r31: r0[X], r16[X] | 4
		out30[i] = multiply(in30_0[i],in30_1[i],modulus30,barrett_ratio30,barrett_k30);
		// add: r32: r30[X], r31[X] | 4
		out31[i] = add(in31_0[i],in31_1[i],modulus31);
		// mul: r33: r18[X], r4[X] | 3
		out32[i] = multiply(in32_0[i],in32_1[i],modulus32,barrett_ratio32,barrett_k32);
		// mul: r34: r3[X], r19[X] | 3
		out33[i] = multiply(in33_0[i],in33_1[i],modulus33,barrett_ratio33,barrett_k33);
		// add: r35: r33[X], r34[X] | 3
		out34[i] = add(in34_0[i],in34_1[i],modulus34);
		// mul: r36: r21[X], r7[X] | 2
		out35[i] = multiply(in35_0[i],in35_1[i],modulus35,barrett_ratio35,barrett_k35);
		// mul: r37: r6[X], r22[X] | 2
		out36[i] = multiply(in36_0[i],in36_1[i],modulus36,barrett_ratio36,barrett_k36);
		// add: r38: r36[X], r37[X] | 2
		out37[i] = add(in37_0[i],in37_1[i],modulus37);
		// mul: r39: r24[X], r10[X] | 1
		out38[i] = multiply(in38_0[i],in38_1[i],modulus38,barrett_ratio38,barrett_k38);
		// mul: r40: r9[X], r25[X] | 1
		out39[i] = multiply(in39_0[i],in39_1[i],modulus39,barrett_ratio39,barrett_k39);
		// add: r41: r39[X], r40[X] | 1
		out40[i] = add(in40_0[i],in40_1[i],modulus40);
		// mul: r42: r27[X], r13[X] | 0
		out41[i] = multiply(in41_0[i],in41_1[i],modulus41,barrett_ratio41,barrett_k41);
		// mul: r43: r12[X], r28[X] | 0
		out42[i] = multiply(in42_0[i],in42_1[i],modulus42,barrett_ratio42,barrett_k42);
		// add: r44: r42[X], r43[X] | 0
		out43[i] = add(in43_0[i],in43_1[i],modulus43);
	}
}

__host__ void _function1_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel1_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel9_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus3 = modulii[3];
	auto barrett_k3 = barrett_k[3];
	auto barrett_ratio3 = barrett_ratios[3];
	auto in3_0 = args_inputs[2];
	auto in3_1 = args_inputs[3];
	auto out3 = args_outputs[1];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r47: r45, r46[X] | 63
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r49: r45[X], r48[X] | 63
		out3[i] = multiply(in3_0[i],in3_1[i],modulus3,barrett_ratio3,barrett_k3);
	}
}

__host__ void _function9_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel9_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel11_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus3 = modulii[3];
	auto barrett_k3 = barrett_k[3];
	auto barrett_ratio3 = barrett_ratios[3];
	auto in3_0 = args_inputs[2];
	auto in3_1 = args_inputs[3];
	auto out3 = args_outputs[1];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r52: r50, r51[X] | 62
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r54: r50[X], r53[X] | 62
		out3[i] = multiply(in3_0[i],in3_1[i],modulus3,barrett_ratio3,barrett_k3);
	}
}

__host__ void _function11_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel11_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel13_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus3 = modulii[3];
	auto barrett_k3 = barrett_k[3];
	auto barrett_ratio3 = barrett_ratios[3];
	auto in3_0 = args_inputs[2];
	auto in3_1 = args_inputs[3];
	auto out3 = args_outputs[1];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r57: r55, r56[X] | 61
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r59: r55[X], r58[X] | 61
		out3[i] = multiply(in3_0[i],in3_1[i],modulus3,barrett_ratio3,barrett_k3);
	}
}

__host__ void _function13_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel13_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel15_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus3 = modulii[3];
	auto barrett_k3 = barrett_k[3];
	auto barrett_ratio3 = barrett_ratios[3];
	auto in3_0 = args_inputs[2];
	auto in3_1 = args_inputs[3];
	auto out3 = args_outputs[1];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r62: r60, r61[X] | 60
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r64: r60[X], r63[X] | 60
		out3[i] = multiply(in3_0[i],in3_1[i],modulus3,barrett_ratio3,barrett_k3);
	}
}

__host__ void _function15_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel15_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel17_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus1 = modulii[1];
	auto barrett_k1 = barrett_k[1];
	auto barrett_ratio1 = barrett_ratios[1];
	auto in1_0 = args_inputs[0];
	auto in1_1 = args_inputs[1];
	auto out1 = args_outputs[0];
	auto modulus3 = modulii[3];
	auto barrett_k3 = barrett_k[3];
	auto barrett_ratio3 = barrett_ratios[3];
	auto in3_0 = args_inputs[2];
	auto in3_1 = args_inputs[3];
	auto out3 = args_outputs[1];
	auto modulus4 = modulii[4];
	auto barrett_k4 = barrett_k[4];
	auto barrett_ratio4 = barrett_ratios[4];
	auto in4 = args_inputs[4];
	auto out4 = args_outputs[2];
	auto modulus6 = modulii[6];
	auto barrett_k6 = barrett_k[6];
	auto barrett_ratio6 = barrett_ratios[6];
	auto in6_0 = args_inputs[5];
	auto in6_1 = args_inputs[6];
	auto out6 = args_outputs[3];
	auto modulus8 = modulii[8];
	auto barrett_k8 = barrett_k[8];
	auto barrett_ratio8 = barrett_ratios[8];
	auto in8_0 = args_inputs[7];
	auto in8_1 = args_inputs[8];
	auto out8 = args_outputs[4];
	auto modulus9 = modulii[9];
	auto barrett_k9 = barrett_k[9];
	auto barrett_ratio9 = barrett_ratios[9];
	auto in9 = args_inputs[9];
	auto out9 = args_outputs[5];
	auto modulus11 = modulii[11];
	auto barrett_k11 = barrett_k[11];
	auto barrett_ratio11 = barrett_ratios[11];
	auto in11_0 = args_inputs[10];
	auto in11_1 = args_inputs[11];
	auto out11 = args_outputs[6];
	auto modulus13 = modulii[13];
	auto barrett_k13 = barrett_k[13];
	auto barrett_ratio13 = barrett_ratios[13];
	auto in13_0 = args_inputs[12];
	auto in13_1 = args_inputs[13];
	auto out13 = args_outputs[7];
	auto modulus14 = modulii[14];
	auto barrett_k14 = barrett_k[14];
	auto barrett_ratio14 = barrett_ratios[14];
	auto in14 = args_inputs[14];
	auto out14 = args_outputs[8];
	auto modulus16 = modulii[16];
	auto barrett_k16 = barrett_k[16];
	auto barrett_ratio16 = barrett_ratios[16];
	auto in16_0 = args_inputs[15];
	auto in16_1 = args_inputs[16];
	auto out16 = args_outputs[9];
	auto modulus18 = modulii[18];
	auto barrett_k18 = barrett_k[18];
	auto barrett_ratio18 = barrett_ratios[18];
	auto in18_0 = args_inputs[17];
	auto in18_1 = args_inputs[18];
	auto out18 = args_outputs[10];
	auto modulus19 = modulii[19];
	auto barrett_k19 = barrett_k[19];
	auto barrett_ratio19 = barrett_ratios[19];
	auto in19 = args_inputs[19];
	auto out19 = args_outputs[11];
	auto modulus21 = modulii[21];
	auto barrett_k21 = barrett_k[21];
	auto barrett_ratio21 = barrett_ratios[21];
	auto in21_0 = args_inputs[20];
	auto in21_1 = args_inputs[21];
	auto out21 = args_outputs[12];
	auto modulus23 = modulii[23];
	auto barrett_k23 = barrett_k[23];
	auto barrett_ratio23 = barrett_ratios[23];
	auto in23_0 = args_inputs[22];
	auto in23_1 = args_inputs[23];
	auto out23 = args_outputs[13];
	auto modulus24 = modulii[24];
	auto barrett_k24 = barrett_k[24];
	auto barrett_ratio24 = barrett_ratios[24];
	auto in24 = args_inputs[24];
	auto out24 = args_outputs[14];
	auto modulus26 = modulii[26];
	auto barrett_k26 = barrett_k[26];
	auto barrett_ratio26 = barrett_ratios[26];
	auto in26_0 = args_inputs[25];
	auto in26_1 = args_inputs[26];
	auto out26 = args_outputs[15];
	auto modulus28 = modulii[28];
	auto barrett_k28 = barrett_k[28];
	auto barrett_ratio28 = barrett_ratios[28];
	auto in28_0 = args_inputs[27];
	auto in28_1 = args_inputs[28];
	auto out28 = args_outputs[16];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// mul: r67: r65, r66[X] | 59
		out1[i] = multiply(in1_0[i],in1_1[i],modulus1,barrett_ratio1,barrett_k1);
		// mul: r69: r65[X], r68[X] | 59
		out3[i] = multiply(in3_0[i],in3_1[i],modulus3,barrett_ratio3,barrett_k3);
		// mov: r70: r2[X] | 4
		out4[i] = in4[i];
		// mul: r72: r70, r71[X] | 4
		out6[i] = multiply(in6_0[i],in6_1[i],modulus6,barrett_ratio6,barrett_k6);
		// mul: r74: r70[X], r73[X] | 4
		out8[i] = multiply(in8_0[i],in8_1[i],modulus8,barrett_ratio8,barrett_k8);
		// mov: r75: r5[X] | 3
		out9[i] = in9[i];
		// mul: r77: r75, r76[X] | 3
		out11[i] = multiply(in11_0[i],in11_1[i],modulus11,barrett_ratio11,barrett_k11);
		// mul: r79: r75[X], r78[X] | 3
		out13[i] = multiply(in13_0[i],in13_1[i],modulus13,barrett_ratio13,barrett_k13);
		// mov: r80: r8[X] | 2
		out14[i] = in14[i];
		// mul: r82: r80, r81[X] | 2
		out16[i] = multiply(in16_0[i],in16_1[i],modulus16,barrett_ratio16,barrett_k16);
		// mul: r84: r80[X], r83[X] | 2
		out18[i] = multiply(in18_0[i],in18_1[i],modulus18,barrett_ratio18,barrett_k18);
		// mov: r85: r11[X] | 1
		out19[i] = in19[i];
		// mul: r87: r85, r86[X] | 1
		out21[i] = multiply(in21_0[i],in21_1[i],modulus21,barrett_ratio21,barrett_k21);
		// mul: r89: r85[X], r88[X] | 1
		out23[i] = multiply(in23_0[i],in23_1[i],modulus23,barrett_ratio23,barrett_k23);
		// mov: r90: r14[X] | 0
		out24[i] = in24[i];
		// mul: r92: r90, r91[X] | 0
		out26[i] = multiply(in26_0[i],in26_1[i],modulus26,barrett_ratio26,barrett_k26);
		// mul: r94: r90[X], r93[X] | 0
		out28[i] = multiply(in28_0[i],in28_1[i],modulus28,barrett_ratio28,barrett_k28);
	}
}

__host__ void _function17_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel17_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel40_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r105: r17[X], r95[X] | 4
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function40_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel40_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel42_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r106: r20[X], r96[X] | 3
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function42_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel42_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel44_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r107: r23[X], r97[X] | 2
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function44_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel44_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel46_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r108: r26[X], r98[X] | 1
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function46_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel46_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel48_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r109: r29[X], r99[X] | 0
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function48_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel48_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel50_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r110: r32[X], r100[X] | 4
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function50_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel50_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel52_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r111: r35[X], r101[X] | 3
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function52_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel52_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel54_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r112: r38[X], r102[X] | 2
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function54_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel54_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel56_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r113: r41[X], r103[X] | 1
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function56_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel56_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

__global__ void _kernel58_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	auto modulus0 = modulii[0];
	auto barrett_k0 = barrett_k[0];
	auto barrett_ratio0 = barrett_ratios[0];
	auto in0_0 = args_inputs[0];
	auto in0_1 = args_inputs[1];
	auto out0 = args_outputs[0];
	for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {
		// add: r114: r44[X], r104[X] | 0
		out0[i] = add(in0_0[i],in0_1[i],modulus0);
	}
}

__host__ void _function58_ (LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	dim3 gridSize(GRID_DIM_X,1,1);
	dim3 blockSize(BLOCK_DIM_X,1,1);
	_kernel58_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);
CHECK_CUDA_ERROR();
}

// End of Kernel Definitions

void execute_fused_kernels(int idx, LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {
	switch (idx) {
		case 1: _function1_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 9: _function9_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 11: _function11_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 13: _function13_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 15: _function15_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 17: _function17_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 40: _function40_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 42: _function42_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 44: _function44_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 46: _function46_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 48: _function48_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 50: _function50_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 52: _function52_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 54: _function54_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 56: _function56_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		case 58: _function58_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;
		default:
			printf("Error: idx out of range\n");
			break;
	}
}
// End of File
