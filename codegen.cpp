#include <fstream>
#include <iomanip>
#include <regex>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <assert.h>
#include <cstdlib>

#include "codegen.h"

namespace fusionFHE {
    void Codegen::handle_preload(std::ofstream &kernelFile, std::string indent, const Fusion::FusedInstr &fused_instr, int kernel_idx) {
        int num_instr = fused_instr.opcode_list.size();
        int input_counter = 0;
        int output_counter = 0;
        int scalar_counter = 0;
        int rot_counter = 0;

        for (size_t instr_idx = 0; instr_idx < num_instr; instr_idx++) {
            std::string opcode = fused_instr.opcode_list.at(instr_idx);
            if (opcode == "store" || opcode == "spill") {
                throw std::runtime_error("Should not preload for: " + opcode);
            } else if (opcode == "load" || opcode == "loas" ||  opcode == "evg") {
                // Can appear in fused instruction, but should not create kernel instructions
            } else if (opcode == "dis" || opcode == "rcv" || opcode == "joi" || opcode == "bci") {
                throw std::runtime_error("Should not preload for: " + opcode);
            } else if (opcode == "int" || opcode == "ntt" || opcode == "sud" || opcode == "pl1") {
                throw std::runtime_error("Should not preload for: " + opcode);
            } else if (opcode == "rsv" || opcode == "mod") {
                throw std::runtime_error("Should not preload for: " + opcode);
            } else if (opcode == "add") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << "_0 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto in" << instr_idx << "_1 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "sub") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << "_0 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto in" << instr_idx << "_1 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "mul" || opcode == "mup") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << "_0 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto in" << instr_idx << "_1 = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "ads") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto scalar" << instr_idx << " = args_scalars[" << scalar_counter << "];\n";
                scalar_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "sus") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto scalar" << instr_idx << " = args_scalars[" << scalar_counter << "];\n";
                scalar_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "mus") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto scalar" << instr_idx << " = args_scalars[" << scalar_counter << "];\n";
                scalar_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "neg") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "mov") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "rot") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto rot_map" << instr_idx << " = rotation_map[" << rot_counter << "];\n";
                rot_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "con") {
                kernelFile << indent << "auto modulus" << instr_idx << " = modulii[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_k" << instr_idx << " = barrett_k[" << instr_idx << "];\n";
                kernelFile << indent << "auto barrett_ratio" << instr_idx << " = barrett_ratios[" << instr_idx << "];\n";
                kernelFile << indent << "auto in" << instr_idx << " = args_inputs[" << input_counter << "];\n";
                input_counter++;
                kernelFile << indent << "auto rot_map" << instr_idx << " = rotation_map[" << rot_counter << "];\n";
                rot_counter++;
                kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                output_counter++;
            } else if (opcode == "rsi") {
                // kernelFile << indent << "auto out" << instr_idx << " = args_outputs[" << output_counter << "];\n";
                // output_counter++;
                throw std::runtime_error("Should not preload for: " + opcode);
            } else {
                throw std::runtime_error("Should not preload for: " + opcode);
            }
        }
        return;
    }

    void Codegen::handle_instructions(std::ofstream &kernelFile, std::string indent, const Fusion::FusedInstr &fused_instr, int kernel_idx) {
        int num_instr = fused_instr.opcode_list.size();
        for (size_t instr_idx = 0; instr_idx < num_instr; instr_idx++) {
            std::string opcode = fused_instr.opcode_list.at(instr_idx);
            if (opcode == "store" || opcode == "spill") {
                throw std::runtime_error("Should not codegen for: " + opcode);
            } else if (opcode == "load" || opcode == "loas" ||  opcode == "evg") {
                // Can appear in fused instruction, but should not create kernel instructions
            } else if (opcode == "dis" || opcode == "rcv" || opcode == "joi" || opcode == "bci") {
                throw std::runtime_error("Should not codegen for: " + opcode);
            } else if (opcode == "int" || opcode == "ntt" || opcode == "sud" || opcode == "pl1") {
                throw std::runtime_error("Should not codegen for: " + opcode);
            } else if (opcode == "rsv" || opcode == "mod") {
                throw std::runtime_error("Should not codegen for: " + opcode);
            } else if (opcode == "add") {
                // ***
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "add(in" << instr_idx << "_0[i],in" << instr_idx << "_1[i],modulus" << instr_idx << ");\n";
            } else if (opcode == "sub") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "subtract(in" << instr_idx << "_0[i],in" << instr_idx << "_1[i],modulus" << instr_idx << ");\n";
            } else if (opcode == "mul" || opcode == "mup") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "multiply(in" << instr_idx << "_0[i],in" << instr_idx << "_1[i],modulus" << instr_idx << ",barrett_ratio" << instr_idx << ",barrett_k" << instr_idx << ");\n";
            } else if (opcode == "ads") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "add(in" << instr_idx << "[i],scalar" << instr_idx << ",modulus" << instr_idx << ");\n";
            } else if (opcode == "sus") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "subtract(in" << instr_idx << "[i],scalar" << instr_idx << ",modulus" << instr_idx << ");\n";
            } else if (opcode == "mus") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "multiply(in" << instr_idx << "[i],scalar" << instr_idx << ",modulus" << instr_idx << ",barrett_ratio" << instr_idx << ",barrett_k" << instr_idx << ");\n";
            } else if (opcode == "neg") {  
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "neg(in" << instr_idx << "[i],modulus" << instr_idx << ");\n";
            } else if (opcode == "mov") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";\
                kernelFile << indent << "out" << instr_idx << "[i] = " << "in" << instr_idx << "[i];\n";
            } else if (opcode == "rot") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "in" << instr_idx << "[rot_map" << instr_idx << "[i]];\n";
            } else if (opcode == "con") {
                kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                kernelFile << indent << "out" << instr_idx << "[i] = " << "in" << instr_idx << "[rot_map" << instr_idx << "[i]];\n";
            } else if (opcode == "rsi") {
                // kernelFile << indent << "// " << opcode << ": " << fused_instr.instr_list.at(instr_idx) << "\n";
                // kernelFile << indent << "out" << instr_idx << "[i] = 0;\n";
                throw std::runtime_error("Should not codegen for: " + opcode);
            } else {
                throw std::runtime_error("Should not codegen for: " + opcode);
            }
        }
        return;
    }
    
    void Codegen::generate_fused_kernels(const std::string &dest_file_base, std::vector<Fusion::FusedInstr> instruction_list, const std::size_t partition) {
        std::string file_path = dest_file_base + "fused_kernels" + std::to_string(partition) + ".cu";
        std::ofstream kernelFile(file_path);
        if (!kernelFile.is_open()) {
            std::cerr << "Error opening file\n";
        }
        kernelFile << "#include \"kernel_helpers.cuh\"\n";
        kernelFile << "#include <cstdlib>\n";
        kernelFile << "#include <stdio.h>\n" << "\n";
        std::string fn_indent = indent(1);
        std::string for_indent = indent(2);

        for (auto &fused_instr : instruction_list) {
            // Note all standalone functions do not need separate functions generated
            std::size_t num_instr = fused_instr.opcode_list.size();
            int kernel_idx = fused_instr.fused_idx;
            if (!fused_instr.fusable) continue;

            // If we are in a fused kernel, create the header
            write_kernel_header(kernelFile, kernel_idx);
 
            // Pass 1 - prepending all parameter loads
            handle_preload(kernelFile, fn_indent, fused_instr, kernel_idx);

            // Insert standard thread_idx for loop
            kernelFile << fn_indent << "for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; i < LENGTH; i+= blockDim.x * gridDim.x) {\n";

            // Pass 2 - inserting actual instructions
            handle_instructions(kernelFile, for_indent, fused_instr, kernel_idx);

            // Close for loop
            kernelFile << fn_indent << "}\n";
            // Close kernel definition
            kernelFile << "}\n" << "\n";

            // Create host function that launches kernel
            write_host_header(kernelFile, kernel_idx);
            write_host_body(kernelFile, kernel_idx);
            // Close host definition
            kernelFile << "}\n" << "\n";
        }
        kernelFile << "// End of Kernel Definitions\n\n";

        // Create a function that can be used to launch these kernels
        kernelFile << "void execute_fused_kernels(int idx, LimbDataType ** args_outputs, const LimbDataType ** args_inputs, const LimbDataType * args_scalars, const LimbDataType * modulii, const LimbDataType * barrett_ratios, const uint32_t * barrett_k, const uint32_t ** rotation_map) {\n";
        kernelFile << indent(1) << "switch (idx) {\n";
        for (int i = 0; i < instruction_list.size(); i++) { 
            auto &fused_instr = instruction_list.at(i);
            if (fused_instr.fusable) {
                kernelFile << indent(2) << "case " << i << ": _function" << i << "_(args_outputs, args_inputs, args_scalars, modulii, barrett_ratios, barrett_k, rotation_map); break;\n";
            }
        }
        kernelFile << indent(2) << "default:\n";
        kernelFile << indent(3) << "printf(\"Error: idx out of range\\n\");\n";
        kernelFile << indent(3) << "break;\n";
        kernelFile << indent(1) << "}\n";
        kernelFile << "}\n";
        kernelFile << "// End of File\n";
    }
}