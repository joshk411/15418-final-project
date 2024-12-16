#include <chrono>
#include <iostream>

#include "codegen.h"
#define num_partitions 1

int main() {
    std::string file_base = "fused_files/";
    fusionFHE::Fusion fusion;
    fusionFHE::Codegen codegen;
    std::chrono::steady_clock::time_point begin_ = std::chrono::steady_clock::now();
    // fusion.parse_instructions("tests/bootstrap/bootstrap51.instructions", num_partitions);
    fusion.parse_instructions("tests/mul_rot1/instructions", num_partitions);
    // fusion.print_instruction_list(0);
    codegen.generate_fused_kernels(file_base,fusion.get_instruction_list(0),0);
    std::chrono::steady_clock::time_point end_ = std::chrono::steady_clock::now();
    std::cout << "Execution Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count() << " milliseconds" << std::endl;
    return 0;
}