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
#include <thread>

#include "fusion.h"

namespace fusionFHE {
    std::string get_opcode(std::string &instruction) {
        auto pos = instruction.find(" ");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid Instruction: " + instruction);
        }
        auto opcode = instruction.substr(0, pos);
        instruction.erase(0, pos + 1);
        return opcode;
    };

    Fusion::FusedInstr Fusion::create_new_fusedInstr(std::string &opcode, std::string &instruction, const int idx, const bool fusable) {
        Fusion::FusedInstr new_fused_op;
        new_fused_op.opcode_list = {opcode};
        new_fused_op.instr_list = {instruction};
        new_fused_op.fused_idx = idx;
        new_fused_op.fusable = fusable;
        return new_fused_op;
    }
    
    /* 
     * Takes in one instruction stream and generates fused instructions. This function implements naive fusion,
     * where all kernels get fused with NTT and INTT and SUD (kernels with inter-block dependencies) serve as breakpoints
     * Note we isolate functions like load and store (non-device ops) and dis/rcv/joi (specialized communication ops).
     */
    void Fusion::generate_fused_instructions(std::string &opcode, std::string &instruction, std::size_t partition) {
        auto &fused_instr_list = fused_instruction_partitions.at(partition);
        if (opcode == "store" || opcode == "spill") {
            // breaking memory operations
            fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),false));
        } else if (opcode == "load" || opcode == "loas" ||  opcode == "evg") {
            // memory operations - note we will push all load instructions to the top and then execute the fused kernel
            if (fused_instr_list.size() == 0) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),false));
            }
            else if ((!fused_instr_list[fused_instr_list.size() - 1].fusable)) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),true));
            }
            else {
                auto &curr_fused_instr = fused_instr_list.at(fused_instr_list.size() - 1);
                curr_fused_instr.opcode_list.push_back(opcode);
                curr_fused_instr.instr_list.push_back(instruction);
                curr_fused_instr.fusable = true;
            }
        } else if (opcode == "dis" || opcode == "rcv" || opcode == "joi" || opcode == "bci") {
            // not a CUDA kernel
            fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),false));
        } else if (opcode == "int" || opcode == "ntt" || opcode == "sud" || opcode == "pl1") {
            // blocking kernels
            fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),false));
        } else if (opcode == "add" || opcode == "ads" || opcode == "sub" || opcode == "sus" || opcode == "mup" || opcode == "mus" || opcode == "mul") {
            // binops
            if (fused_instr_list.size() == 0) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),true));
            }
            else if ((!fused_instr_list[fused_instr_list.size() - 1].fusable)) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),true));
            }
            else {
                auto &curr_fused_instr = fused_instr_list.at(fused_instr_list.size() - 1);
                curr_fused_instr.opcode_list.push_back(opcode);
                curr_fused_instr.instr_list.push_back(instruction);
                curr_fused_instr.fusable = true;
            }
        } else if (opcode == "neg" || opcode == "mov" || opcode == "rot" || opcode == "con") {
            if (fused_instr_list.size() == 0) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),true));
            }
            else if ((!fused_instr_list[fused_instr_list.size() - 1].fusable)) {
                fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),true));
            }
            else {
                auto &curr_fused_instr = fused_instr_list.at(fused_instr_list.size() - 1);
                curr_fused_instr.opcode_list.push_back(opcode);
                curr_fused_instr.instr_list.push_back(instruction);
                curr_fused_instr.fusable = true;
            }
        } else if (opcode == "rsv" || opcode == "mod" || opcode == "rsi") {
            // multiple dests and srcs - making instruction creation difficult
            fused_instr_list.push_back(create_new_fusedInstr(opcode,instruction,fused_instr_list.size(),false));
        } else {
            throw std::runtime_error("Invalid opcode: " + opcode);
        }
    }

    void Fusion::parse_instructions(const std::string &instruction_file_base, const uint8_t partitions) {
        std::vector<std::ifstream> instruction_files;
        std::vector<uint64_t> instruction_count;
        std::vector<std::string> instructions;
        std::vector<bool> advance;
        std::vector<bool> completed;
        size_t complete_count = 0;

        fused_instruction_partitions.resize(partitions);

        for (size_t i = 0; i < partitions; i++) {
            auto instruction_file_name = instruction_file_base + std::to_string(i);
            std::ifstream ifile(instruction_file_name, std::ios::in);
            instruction_files.push_back(std::move(ifile));
            instruction_count.push_back(0);
            std::string instruction;
            std::getline(instruction_files[i], instruction);
            instructions.push_back("");
            advance.push_back(true);
            completed.push_back(false);
        }

        auto thread_fn = [&](std::size_t partition){
            std::string instruction;
            while(true){
                if(advance[partition]){
                    if(!std::getline(instruction_files.at(partition), instruction)){
                        if(completed[partition] == false){
                            complete_count++;
                            completed[partition] = true;
                        }
                        if(complete_count == partitions){
                            break;
                        } else {
                            continue;
                        }
                    }
                    instruction_count[partition]++;
                    instructions[partition] = instruction;
                }
                instruction = instructions[partition];
                auto opcode = get_opcode(instruction);
                generate_fused_instructions(opcode, instruction, partition);
            }
        };

        std::vector<std::thread> threads;
        for(size_t tid = 0; tid < partitions; tid++) {
            threads.push_back(std::thread(thread_fn,tid));
        }

        for(size_t tid = 0; tid < partitions; tid++) {
            threads[tid].join();
        }
    }

    std::vector<Fusion::FusedInstr>& Fusion::get_instruction_list(std::size_t partition) {
        return fused_instruction_partitions.at(partition);
    }

    void Fusion::print_instruction_list(std::size_t partition) {
        printf("entered\n");
        for (auto& instr : fused_instruction_partitions.at(partition)) {
            printf("**********************\n");
            printf("Fused instruction %d\n", instr.fused_idx);
            printf("Fused?: %d\n", instr.fusable);
            printf("opcode: ");
            for (auto& op : instr.opcode_list) {
                std::cout << op << ", ";
            }
            printf("\n");
            for (auto& in : instr.instr_list) {
                std::cout << in << ", ";
            }
            printf("\n");
            printf("**********************\n");
        }
    }
}