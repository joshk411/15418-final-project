#include <map>
#include <unordered_map>
#include <string>
#include <variant>
#include <regex>
#include <chrono>
#include <unordered_set>
#include <fstream>

#include "fusion.h"

namespace fusionFHE {
    class Codegen {
        public: 
            std::string indent(int indent_depth) {
                std::string tab = "";
                for (auto i = 0; i < indent_depth; i++) {
                    tab += "\t";
                }
                return tab;
            }
            void write_kernel_header(std::ofstream &kernelFile, int fused_idx) {
                kernelFile << "__global__ void _kernel" << fused_idx << "_" << " ("
                    << "LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map"
                    << ") {\n";
            }
            void write_host_header(std::ofstream &kernelFile, int fused_idx) {
                kernelFile << "__host__ void _function" << fused_idx << "_" << " ("
                    << "LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map"
                    << ") {\n";
            }
            void write_host_body(std::ofstream &kernelFile, int fused_idx) {
                std::string fn_indent = indent(1);
                kernelFile << fn_indent << "dim3 gridSize(GRID_DIM_X,1,1);\n";
	            kernelFile << fn_indent << "dim3 blockSize(BLOCK_DIM_X,1,1);\n";
	            kernelFile << fn_indent << "_kernel" << fused_idx << "_<<<gridSize,blockSize>>>(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map);\n";
                kernelFile << "CHECK_CUDA_ERROR();\n";
            }
            void handle_preload(std::ofstream &kernelFile, std::string indent, const Fusion::FusedInstr &fused_instr, int kernel_idx);
            void handle_instructions(std::ofstream &kernelFile, std::string indent, const Fusion::FusedInstr &fused_instr, int kernel_idx);
            void generate_fused_kernels(const std::string &dest_file_base, std::vector<Fusion::FusedInstr> instruction_list, const std::size_t partition);
    };
}