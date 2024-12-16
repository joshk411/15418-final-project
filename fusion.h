#include <map>
#include <unordered_map>
#include <string>
#include <variant>
#include <regex>
#include <chrono>
#include <unordered_set>

namespace fusionFHE {
    class Fusion {
        public:
            struct FusedInstr {
                std::vector<std::string> opcode_list;
                std::vector<std::string> instr_list;
                bool fusable;
                int fused_idx;
            };

            FusedInstr create_new_fusedInstr(std::string &opcode, std::string &instruction, const int idx, const bool fusable);
            void parse_instructions(const std::string &instruction_file_base, const uint8_t partitions);
            void generate_fused_instructions(std::string &opcode, std::string &instruction, std::size_t partition);
            std::vector<FusedInstr>& get_instruction_list(std::size_t partition);
            void print_instruction_list(std::size_t partition);
        private:
            std::vector<std::vector<FusedInstr>> fused_instruction_partitions;
    };
}