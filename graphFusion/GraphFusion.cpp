#include "GraphFusion.h"

#include <fstream>
#include <iomanip>

#include <regex>
#include <iostream>
#include <string>

#include <chrono>
#include <optional>
#include <assert.h>
#include <thread>
#include <cstdlib>

// #define BUTTERSCOTCH_INTERPRETER_VERBOSE
// #define USE_CUDA_GRAPHS

namespace GraphFusion {

    std::string GraphFusion::get_opcode(std::string &instruction) {
        auto pos = instruction.find(" ");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid Instruction: " + instruction);
        }
        auto opcode = instruction.substr(0, pos);
        instruction.erase(0, pos + 1);
        return opcode;
    };

    std::string GraphFusion::get_dest(std::string &instruction) {
        auto pos = instruction.find(":");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid Instruction xx: " + instruction);
        }
        auto dest = instruction.substr(0, pos);
        instruction.erase(0, pos + 2);
        return dest;
    };

    std::string GraphFusion::get_src(std::string &instruction) {
        auto pos = instruction.find(",");
        auto src = instruction.substr(0, pos);
        instruction.erase(0, pos + 2);
        return src;
    };

    int GraphFusion::get_rns_id(std::string &instruction) {
        auto pos = instruction.find("|");
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid Instruction xx: " + instruction);
        }
        auto id = std::stoi(instruction.substr(pos + 2));
        instruction.erase(pos - 1);
        return id;
    };

    size_t get_register_idx(const std::string &reg_str_) {
        std::string reg_str(reg_str_);
        auto pos = reg_str.find("[X]");
        bool dead = false;
        if (pos != std::string::npos) {
            dead = true;
            reg_str.erase(pos);
        }
        if (reg_str[0] != 'r') {
            throw std::runtime_error("Invalid Register: " + reg_str);
        }
        reg_str[0] = '0';
        size_t reg_id = std::stoul(reg_str);
        return reg_id;
    };

    // GraphFusion::LimbT::Element_t GraphFusion::get_register_scalar(const std::string &reg_str_, GraphFusion::ScalarRegisterFileType &srf) {
    //     std::string reg_str(reg_str_);
    //     auto pos = reg_str.find("[X]");
    //     bool dead = false;
    //     if (pos != std::string::npos) {
    //         dead = true;
    //         reg_str.erase(pos);
    //     }
    //     if (reg_str[0] != 's') {
    //         throw std::runtime_error("Invalid Scalar Register: " + reg_str);
    //     }
    //     reg_str[0] = '0';
    //     size_t reg_id = std::stoul(reg_str);
    //     auto scalar = srf[reg_id];
    //     return scalar;
    // };

    // void add_edge_from_set(CUDA::GraphPtr &graph, const size_t partition, Node &curr_node, const GraphFusion::DependencySet &edge_set) {
    void GraphFusion::add_edge_from_set(const size_t partition, std::shared_ptr<Node> curr_node, const GraphFusion::DependencySet &edge_set) {

        // our root should store each connected component
        if (edge_set.empty()) {
            roots_[partition]->insert(curr_node);
        }

        for (auto &edge : edge_set) {
            edge.first->insert(edge.second);
            // have two nodes, second one is child. add it to children of first
            // CUDA::addSingleEdge(graph, edge.first, edge.second);
        }
    }

    void set_union(GraphFusion::DependencySet &a, const GraphFusion::DependencySet &b) {
        for (auto &elem : b) {
            a.insert(elem);
        }
    }

    // GraphFusion::LimbPtrT get_src_reg(const unsigned long reg, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem) {
    //     if (rfMem[reg]) {
    //         return rfMem[reg];
    //     } else {
    //         return rf[reg];
    //     }
    // }

    GraphFusion::DependencySet add_write_dependencies(const unsigned long dest_reg, GraphFusion::NodePtr nodeIn, GraphFusion::NodePtr nodeOut, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads) {
        GraphFusion::DependencySet edge_set;
        std::pair<GraphFusion::NodePtr,GraphFusion::NodePtr> edge;

        // Write-Write Dependency
         auto prev_dest_write = rDep[dest_reg];
         if (prev_dest_write) {
            edge = std::make_pair(prev_dest_write,nodeIn);
            edge_set.insert(edge);
        }
        rDep[dest_reg] = nodeOut;
        // Read-Write Dependency
        auto all_prev_reads = rReads[dest_reg];
        auto num_read_nodes = all_prev_reads.size();
        if (num_read_nodes > 0) {
            for (int i = 0; i < num_read_nodes; i++) {
                edge = std::make_pair(all_prev_reads[i],nodeIn);
                edge_set.insert(edge);
            }
            rReads[dest_reg].clear();
        }
        return edge_set;
    }
    
    // void GraphFusion::handle_load(const std::string &instruction, const std::string &opcode, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
    //                                   GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition) {
    void GraphFusion::handle_load(const std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                                      GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition) {

        std::smatch match;
        // the idea for the copy is that we may parse the original instruction argument passed in
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, load_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif
            GraphFusion::DependencySet edge_set;
            std::pair<NodePtr,NodePtr> edge;

            auto dest_reg = std::stoul(match[1]);
            auto term = match[2];
            bool free_from_mem = false;
            if(match[3].length() != 0){
                free_from_mem = true;
            }

            // LimbPtrT src = nullptr;
            // {
            //     // std::lock_guard<std::mutex> lock(mem_lock);

            //     try {
            //         src = program_memory_local_[partition].at(term);
            //     } catch (const std::out_of_range &err) {
            //         throw std::runtime_error("Load not found: " + std::string(term));
            //     }
            //     // if(free_from_mem){
            //     //     program_memory_.erase(term);
            //     // }
            // }
            // if(free_from_mem){
            if(false){
                // NOTE: current case never reached, come back and implement if needed
            } else {
                // Assumption: no node needed for successor edges as no data is being changed
                // rfMem[dest_reg] = src;
                // rReads[dest_reg].clear();
                // Unsure: I believe the last write node to dest_reg should be the previous write of src to local_mem (if exists)
                // currently clearing the register dependency and then reassigning it if the src write node exists
                // rDep[dest_reg] = nullptr;
                // auto search = localDep.find(term);
                // if (search != localDep.end()) {
                //     Node src_node = search->second;
                //     rDep[dest_reg] = src_node;
                // }
                // auto new_node = CUDA::createNode();
                auto new_node = std::make_shared<Node>(instruction_copy, opcode);

                // evaluators_[partition].dummyNode(new_node);
                // evaluators_[partition].copy(new_node,rf[dest_reg],src,src->rns_base_id());
                set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));

                auto search = localDep.find(term);
                if (search != localDep.end()) {
                    NodePtr src_node = search->second;
                    edge = std::make_pair(src_node,new_node);
                    edge_set.insert(edge);
                }

                auto local_read_nodes = localReads.find(term); // update set of reads for local mem
                if (local_read_nodes != localReads.end()) {
                    auto read_vec = local_read_nodes->second;
                    read_vec.push_back(new_node);
                }
                else {
                    localReads[term] = std::vector<NodePtr>({new_node});
                }
                add_edge_from_set(partition, new_node, edge_set);
            }
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            std::cout << "LOAD " << dest_reg << " : " << term << "\n";
#endif
        } else {
            throw std::runtime_error("Invalid Instruction for load: " + instruction);
        }
    }

    // void GraphFusion::handle_loas(const std::string &instruction, GraphFusion::ScalarRegisterFileType &srf, const size_t partition) {
    void GraphFusion::handle_loas(const std::string &instruction, const size_t partition) {

        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, loas_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto dest_reg = std::stoul(match[1]);
            auto term = match[2];
            bool free_from_mem = false;
            if(match[3].length() != 0){
                free_from_mem = true;
            }

            auto new_node = std::make_shared<Node>(instruction_copy, "loas");

            // LimbT::Element_t src = 0;
            // {
            //     std::lock_guard<std::mutex> lock(mem_lock);
            //     src = program_memory_scalar_.at(term);
            //     if(free_from_mem){
            //         program_memory_scalar_.erase(term);
            //     }
            // }
            // srf[dest_reg] = src;
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            std::cout << "LOAS " << dest_reg << " : " << term << "\n";
#endif
        } else {
            throw std::runtime_error("Invalid Instruction for load: " + instruction);
        }
    }


    void GraphFusion::handle_evg(const std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                                     GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition) {
        // handle_load(instruction, opcode, rf,rfMem,rDep,localDep,rReads,localReads,partition);
        handle_load(instruction, opcode, rDep,localDep,rReads,localReads,partition);
    };
    
    // void GraphFusion::handle_store(std::string &instruction, const std::string &opcode, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
    //                                     GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition) {
    void GraphFusion::handle_store(std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                                        GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;

        auto instruction_copy = instruction;
        auto dest = get_dest(instruction);
        auto term = instruction;
        dest[0] = '0';
        auto reg = std::stoul(dest);
        // auto src = rf[reg];
        // auto src = get_src_reg(reg,rf,rfMem);
        // if(program_memory_local_[partition].find(term) != program_memory_local_[partition].end()) {
        //     return;
        // }

        // Assumption: src must have been previously written to
        auto prev_src_write = rDep[reg];
        if (!prev_src_write) {
            throw std::runtime_error("Src not found in rDep (in store op): " + std::to_string(reg));
        }

        // auto new_node = CUDA::createNode();
        auto new_node = std::make_shared<Node>(instruction_copy, opcode);
        // auto copy = evaluators_[partition].copy(new_node,src,src->rns_base_id());

        if (prev_src_write) {
            edge = std::make_pair(prev_src_write,new_node);
            edge_set.insert(edge);
        }
        // if (!prev_src_write) {
        //     throw std::runtime_error("Src not found in rDep (in store op): " + std::to_string(reg));
        // }
        {
            // std::lock_guard<std::mutex> lock(mem_lock);
            // program_memory_local_[partition][term] = std::move(copy);
            auto search = localDep.find(term);
            if (search != localDep.end()) {
                auto prev_local_write = search->second;
                edge = std::make_pair(prev_local_write,new_node);
                edge_set.insert(edge);
            }
            localDep[term] = new_node;

            // False Dependencies
            auto all_prev_reads = localReads.find(term);
            if (all_prev_reads != localReads.end()) {
                auto read_vec = all_prev_reads->second;
                for (int i = 0; i < read_vec.size(); i++) {
                    edge = std::make_pair(read_vec[i],new_node);
                    edge_set.insert(edge);
                }
                read_vec.clear();
            }
            rReads[reg].push_back(new_node);
        }
        add_edge_from_set(partition, new_node, edge_set);
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
        std::cout << "STORE " << reg << " : " << term << "\n";
#endif
    };

    // void GraphFusion::handle_binop(const std::string &opcode, std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_binop(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {  
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        auto instruction_copy = instruction;
        auto dest = get_dest(instruction);
        auto rns_id = get_rns_id(instruction);
        dest[0] = '0';
        auto dest_reg = std::stoul(dest);
        auto src1_str = get_src(instruction);
        auto src2_str = get_src(instruction);
        auto reg1 = get_register_idx(src1_str);
        auto reg2 = get_register_idx(src2_str);
        // auto src1 = get_src_reg(reg1,rf,rfMem);
        // auto src2 = get_src_reg(reg2,rf,rfMem);

        // auto new_node = CUDA::createNode();
        auto new_node = std::make_shared<Node>(instruction_copy, opcode);
        // if (opcode == "add") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].add(new_node, dest, src1, src2, rns_id);
        // } else if (opcode == "sub") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].subtract(new_node, dest, src1, src2, rns_id);
        // } else if (opcode == "mup") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].multiply(new_node, dest, src1, src2, rns_id);
        // } else if (opcode == "mul") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].multiply(new_node, dest, src1, src2, rns_id);
        // }
        // rfMem[dest_reg] = nullptr;
        
        auto prev_src1_write = rDep[reg1];
        auto prev_src2_write = rDep[reg2];

        if (prev_src1_write) {
            edge = std::make_pair(prev_src1_write,new_node);
            edge_set.insert(edge);
        }
        if (prev_src2_write) {
            edge = std::make_pair(prev_src2_write,new_node);
            edge_set.insert(edge);
        }

        set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
        if (dest_reg != reg1) {
            rReads[reg1].push_back(new_node);
        }
        if (dest_reg != reg2) {
            rReads[reg2].push_back(new_node);
        }
        add_edge_from_set(partition, new_node, edge_set);


#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
        std::cout << opcode << dest_reg << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
#endif
    };

    // void GraphFusion::handle_binop_scalar(const std::string &opcode, std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::ScalarRegisterFileType &srf, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_binop_scalar(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        auto instruction_copy = instruction;
        auto dest = get_dest(instruction);
        auto rns_id = get_rns_id(instruction);
        dest[0] = '0';
        auto dest_reg = std::stoul(dest);
        auto src1_str = get_src(instruction);
        auto src2_str = get_src(instruction);
        auto reg1 = get_register_idx(src1_str);
        // auto src1 = get_src_reg(reg1,rf,rfMem);
        // auto sca2 = get_register_scalar(src2_str, srf);

        // auto new_node = CUDA::createNode();
        auto new_node = std::make_shared<Node>(instruction_copy, opcode);
        // if (opcode == "ads") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].add(new_node, dest, src1, sca2, rns_id);
        // } else if (opcode == "sus") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].subtract(new_node, dest, src1, sca2, rns_id);
        // } else if (opcode == "mus") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].multiply(new_node, dest, src1, sca2, rns_id);
        // }
        // rfMem[dest_reg] = nullptr;

        auto prev_src1_write = rDep[reg1];
        if (prev_src1_write) {
            edge = std::make_pair(prev_src1_write,new_node);
            edge_set.insert(edge);
        }
        set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
        if (dest_reg != reg1) {
            rReads[reg1].push_back(new_node);
        }
        add_edge_from_set(partition, new_node, edge_set);

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
        std::cout << opcode << dest_reg << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
#endif
    };
    
    // void GraphFusion::handle_unop(const std::string &opcode, std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_unop(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        auto instruction_copy = instruction;
        auto dest = get_dest(instruction);
        auto rns_id = get_rns_id(instruction);
        dest[0] = '0';
        auto dest_reg = std::stoul(dest);
        auto src1_str = get_src(instruction);
        auto reg1 = get_register_idx(src1_str);
        // auto src1 = get_src_reg(reg1,rf,rfMem);
        // auto new_node = CUDA::createNode();
        auto new_node = std::make_shared<Node>(instruction_copy, opcode);
        // if (opcode == "con") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].conjugate(new_node, dest, src1, rns_id);
        // } else if (opcode == "neg") {
        //     auto & dest = rf[dest_reg];
        //     evaluators_[partition].negate(new_node, dest, src1, rns_id);
        // } else {
        //     throw std::runtime_error("Unimplemented");
        // }
        // rfMem[dest_reg] = nullptr;

        auto prev_src1_write = rDep[reg1];
        if (prev_src1_write) {
            edge = std::make_pair(prev_src1_write,new_node);
            edge_set.insert(edge);
        }
        set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
        if (dest_reg != reg1) {
            rReads[reg1].push_back(new_node);
        }
        add_edge_from_set(partition, new_node, edge_set);
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
        std::cout << opcode << dest_reg << " :" << src1 << ":" << rns_id << ":\n";
#endif
    };

    // void GraphFusion::handle_mov(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_mov(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, unop_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto dest_reg = std::stoul(match[1]);
            auto src_reg_str = std::string(match[2]);
            src_reg_str[0] = '0';
            auto src_reg = std::stoul(src_reg_str);
            if(dest_reg == src_reg) {
                throw std::runtime_error("Match");
            }
            // auto reg1 = get_register_idx(match[2], rf);
            // auto src1 = get_src_reg(src_reg,rf,rfMem); 
            auto rns_base_id = std::stoul(match[4]);
            // auto new_node = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instruction_copy, opcode);
            // if(match[3].length() == 0){
            //     auto & dest = rf[dest_reg];
            //     // evaluators_[partition].copy(new_node, dest, rf[reg1], rns_base_id);
            //     evaluators_[partition].copy(new_node, dest, src1, rns_base_id);
            //     // CUDA::deviceSynchronize();
            // } else {
            //     auto & dest = rf[dest_reg];
            //     // evaluators_[partition].mov(new_node, dest, rf[reg1], rns_base_id);
            //     evaluators_[partition].mov(new_node, dest, src1, rns_base_id);
            //     // CUDA::deviceSynchronize();
            // }
            // rfMem[dest_reg] = nullptr;
            
            auto prev_src_write = rDep[src_reg];
            if (prev_src_write) {
                edge = std::make_pair(prev_src_write,new_node);
                edge_set.insert(edge);
            }
            set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
            if (dest_reg != src_reg) {
                rReads[src_reg].push_back(new_node);
            }
            add_edge_from_set(partition, new_node, edge_set);
        } else {
            throw std::runtime_error("Invalid Instruction for mov: " + instruction);
        }
    };

    // void GraphFusion::handle_rot(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_rot(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, rot_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto rotation_amount = std::stoi(match[1]);
            auto dest_reg = std::stoul(match[2]);
            auto reg1 = get_register_idx(match[3]);
            // auto src1 = get_src_reg(reg1,rf,rfMem); 
            auto rns_base_id = std::stoul(match[5]);

            // auto & dest = rf[dest_reg];
            // auto new_node = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instruction_copy, opcode);
            // evaluators_[partition].rotate(new_node, dest, rf[reg1], rotation_amount, rns_base_id);
            // evaluators_[partition].rotate(new_node, dest, src1, rotation_amount, rns_base_id);
            // rfMem[dest_reg] = nullptr;

            auto prev_src_write = rDep[reg1];
            if (prev_src_write) {
                edge = std::make_pair(prev_src_write,new_node);
                edge_set.insert(edge);
            }
            set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
            if (dest_reg != reg1) {
                rReads[reg1].push_back(new_node);
            }
            add_edge_from_set(partition, new_node, edge_set);
        } else {
            throw std::runtime_error("Invalid Instruction for rot: " + instruction);
        }
    };

    // void GraphFusion::handle_ntt(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>> &bcu, 
    //                             GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_ntt(const std::string &opcode, const std::string &instruction, 
                                GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, ntt_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            // auto dest_bcu_id = std::stoul(match[1]);
            auto dest_reg = std::stoul(match[1]);
            auto rns_base_id = std::stoul(match[7]);
            // auto new_node1 = CUDA::createNode();
            // auto new_node2 = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instruction_copy, opcode);
            if(match[5].length() != 0) {
                auto src_bcu_id = std::stoul(match[5]);
                auto src_bcu_output_index = std::stoul(match[6]);
                if(opcode == "ntt") {
                    // auto & dest = rf[dest_reg];
                    // // evaluators_[partition].ntt(new_node1, new_node2, dest, *bcu[src_bcu_id], src_bcu_output_index, rns_base_id);
                    // evaluators_[partition].ntt(new_node, new_node, dest, *bcu[src_bcu_id], src_bcu_output_index, rns_base_id);
                    // rfMem[dest_reg] = nullptr;
                    // set_union(edge_set, add_write_dependencies(dest_reg, new_node1, new_node2, rDep, rReads));
                    set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
                    add_edge_from_set(partition, new_node, edge_set);
                } else {
                    throw std::runtime_error("Invalid Instruction for ntt" + instruction);
                }
            } else if(match[3].length() != 0){
                auto reg1 = get_register_idx(match[3]);
                // auto src1 = get_src_reg(reg1,rf,rfMem);
                if(opcode == "ntt") {
                    // auto & dest = rf[dest_reg];
                    // evaluators_[partition].ntt(new_node1, new_node2, dest, rf[reg1], rns_base_id);
                    // evaluators_[partition].ntt(new_node1, new_node2, dest, src1, rns_base_id);
                    // evaluators_[partition].ntt(new_node, new_node, dest, src1, rns_base_id);
                    // rfMem[dest_reg] = nullptr;
                    auto prev_src_write = rDep[reg1];
                    if (prev_src_write) {
                        // edge = std::make_pair(prev_src_write,new_node1);
                        edge = std::make_pair(prev_src_write,new_node);
                        edge_set.insert(edge);
                    }
                    // set_union(edge_set, add_write_dependencies(dest_reg, new_node1, new_node2, rDep, rReads));
                    set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
                    if (dest_reg != reg1) {
                        // rReads[reg1].push_back(new_node2);
                        rReads[reg1].push_back(new_node);
                    }
                    add_edge_from_set(partition, new_node, edge_set);
                } else if(opcode == "int") {
                    // auto & dest = rf[dest_reg];
                    // evaluators_[partition].inverse_ntt(new_node1, new_node2, dest, rf[reg1], rns_base_id);
                    // evaluators_[partition].inverse_ntt(new_node1, new_node2, dest, src1, rns_base_id);
                    // evaluators_[partition].inverse_ntt(new_node, new_node, dest, src1, rns_base_id);
                    // rfMem[dest_reg] = nullptr;
                    auto prev_src_write = rDep[reg1];
                    if (prev_src_write) {
                        // edge = std::make_pair(prev_src_write,new_node1);
                        edge = std::make_pair(prev_src_write,new_node);
                        edge_set.insert(edge);
                    }
                    // set_union(edge_set, add_write_dependencies(dest_reg, new_node1, new_node2, rDep, rReads));
                    set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
                    if (dest_reg != reg1) {
                        // rReads[reg1].push_back(new_node2);
                        rReads[reg1].push_back(new_node);
                    }
                    add_edge_from_set(partition, new_node, edge_set);
                } else {
                    throw std::runtime_error("Invalid Instruction for ntt/intt" + instruction);
                }
            } else {
                throw std::runtime_error("Invalid Instruction for ntt/intt" + instruction);
            }

        } else {
            throw std::runtime_error("Invalid Instruction for ntt: " + instruction);
        }
    };

    // void GraphFusion::handle_sud(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>> &bcu, 
    //                             GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_sud(const std::string &opcode, const std::string &instruction,
                                GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, sud_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            // auto dest_bcu_id = std::stoul(match[1]);
            auto dest_reg = std::stoul(match[1]);
            auto rns_base_id = std::stoul(match[9]);
            auto reg1 = get_register_idx(match[2]);
            // auto src1 = get_src_reg(reg1,rf,rfMem);
            // auto new_node1 = CUDA::createNode();
            // auto new_node2 = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instruction_copy, opcode);
            if(match[7].length() != 0) {
                // auto & dest = rf[dest_reg];
                // auto src_bcu_id = std::stoul(match[7]);
                // auto src_bcu_output_index = std::stoul(match[8]);
                // // evaluators_[partition].subtract_and_divide_modulus(new_node1, new_node2, dest, rf[reg1], *bcu[src_bcu_id], src_bcu_output_index, rns_base_id);
                // // evaluators_[partition].subtract_and_divide_modulus(new_node1, new_node2, dest, src1, *bcu[src_bcu_id], src_bcu_output_index, rns_base_id);
                // evaluators_[partition].subtract_and_divide_modulus(new_node, new_node, dest, src1, *bcu[src_bcu_id], src_bcu_output_index, rns_base_id);
                // rfMem[dest_reg] = nullptr;

                auto prev_op_write = rDep[reg1];
                if (prev_op_write) {
                    // edge = std::make_pair(prev_op_write,new_node1);
                    edge = std::make_pair(prev_op_write,new_node);
                    edge_set.insert(edge);
                }
                // set_union(edge_set, add_write_dependencies(dest_reg, new_node1, new_node2, rDep, rReads));
                set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
                if (dest_reg != reg1) {
                    // rReads[reg1].push_back(new_node2);
                    rReads[reg1].push_back(new_node);
                }
                add_edge_from_set(partition, new_node, edge_set);
            } else if(match[5].length() != 0){
                // auto & dest = rf[dest_reg];
                auto reg2 = get_register_idx(match[5]);
                // auto src2 = get_src_reg(reg2,rf,rfMem);
                // // evaluators_[partition].subtract_and_divide_modulus(new_node1, new_node2, dest, rf[reg1], rf[reg2], rns_base_id);
                // // evaluators_[partition].subtract_and_divide_modulus(new_node1, new_node2, dest, src1, src2, rns_base_id);
                // evaluators_[partition].subtract_and_divide_modulus(new_node, new_node, dest, src1, src2, rns_base_id);
                // rfMem[dest_reg] = nullptr;

                auto prev_src_write = rDep[reg2];
                auto prev_op_write = rDep[reg1];
                if (prev_src_write) {
                    // edge = std::make_pair(prev_src_write,new_node1);
                    edge = std::make_pair(prev_src_write,new_node);
                    edge_set.insert(edge);
                }
                if (prev_op_write) {
                    // edge = std::make_pair(prev_op_write,new_node1);
                    edge = std::make_pair(prev_op_write,new_node);
                    edge_set.insert(edge);
                }
                // set_union(edge_set, add_write_dependencies(dest_reg, new_node1, new_node2, rDep, rReads));
                set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
                if (dest_reg != reg1) {
                    // rReads[reg1].push_back(new_node2);
                    rReads[reg1].push_back(new_node);
                }
                if (dest_reg != reg2) {
                    // rReads[reg2].push_back(new_node2);
                    rReads[reg2].push_back(new_node);
                }
                add_edge_from_set(partition, new_node, edge_set);
            } else {
                throw std::runtime_error("Invalid Instruction for sud" + instruction);
            }
        } else {
            throw std::runtime_error("Invalid Instruction for sud: " + instruction);
        }

    };

    // void GraphFusion::handle_rsi(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_rsi(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instructions_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, rsi_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            for(auto i = 2; i < match.size(); i+=2) {
                auto dest_reg = std::stoul(match[i]);
                // auto & dest = rf[dest_reg];
                auto new_node = std::make_shared<Node>(instructions_copy, opcode);
                // auto new_node = CUDA::createNode();
                // evaluators_[partition].get_zero(new_node,dest);
                // rfMem[dest_reg] = nullptr;
                set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
            }
            auto dest_reg = std::stoul(match[match.size() - 1]);
            // auto new_node = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instructions_copy, opcode);
            // evaluators_[partition].get_zero(new_node,rf[dest_reg]);
            set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
            // rfMem[dest_reg] = nullptr;
            add_edge_from_set(partition, new_node, edge_set);
        } else {
            throw std::runtime_error("Invalid Instruction for rsi: " + instruction);
        }
    };

    // void GraphFusion::handle_rsv(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_rsv(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instructions_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, rsv_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            std::string dests_str(match[1]);
            std::string dest_rns_base_ids_str(match[4]);

            size_t pos = 0;
            // std::vector<LimbPtrT> dests;
            std::vector<unsigned long> dest_reg_idxs;
            unsigned long reg_idx;
            while((pos = dests_str.find(',')) != std::string::npos) {
                auto reg = dests_str.substr(0,pos);
                #ifdef BUTTERSCOTCH_DEBUG
                if(reg[0] != 'r') {
                    throw std::runtime_error("Invalid Instruction for rsv: " + instruction);
                }
                #endif
                reg[0] = '0';
                reg_idx = std::stoul(reg);
                // dests.push_back(rf[reg_idx]);
                dest_reg_idxs.push_back(reg_idx);
                dests_str.erase(0,pos+2);
            }
            #ifdef BUTTERSCOTCH_DEBUG
            if(dests_str[0] != 'r') {
                throw std::runtime_error("Invalid Instruction for rsv: " + instruction);
            }
            #endif
            dests_str[0] = '0';
            reg_idx = std::stoul(dests_str);
            // dests.push_back(rf[reg_idx]);
            dest_reg_idxs.push_back(reg_idx);

            pos = 0;
            std::vector<uint64_t> dest_rns_base_ids;
            while((pos = dest_rns_base_ids_str.find(',')) != std::string::npos) {
                auto id = dest_rns_base_ids_str.substr(0,pos);
                dest_rns_base_ids.push_back(std::stoul(id));
                dest_rns_base_ids_str.erase(0,pos+1);
            }
            dest_rns_base_ids.push_back(std::stoul(dest_rns_base_ids_str));

            auto src1 = get_register_idx(match[2]);
            // auto src_reg = get_src_reg(src1,rf,rfMem);
            auto rns_base_id = std::stoul(match[5]);
            // auto nodeIn = CUDA::createNode();
            // auto nodeOut = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instructions_copy, opcode);

            // evaluators_[partition].resolve_write(nodeIn,nodeOut,rf[src1],dests,dest_rns_base_ids,rns_base_id);
            // evaluators_[partition].resolve_write(nodeIn,nodeOut,src_reg,dests,dest_rns_base_ids,rns_base_id);
            // evaluators_[partition].resolve_write(new_node,new_node,src_reg,dests,dest_rns_base_ids,rns_base_id);

            auto prev_src1_write = rDep[src1];
            if (prev_src1_write) {
                // edge = std::make_pair(prev_src1_write,nodeIn);
                edge = std::make_pair(prev_src1_write,new_node);
                edge_set.insert(edge);
            }

            // Note: case on the length of dest to update last write (due to pass by reference)
            auto is_src_dest = false;
            for (auto & dest_idx : dest_reg_idxs) {
                // rfMem[dest_idx] = nullptr;
                // if (dests.size() == 2) {
                //     // set_union(edge_set, add_write_dependencies(dest_idx, nodeIn, nodeIn, rDep, rReads));
                //     set_union(edge_set, add_write_dependencies(dest_idx, new_node, new_node, rDep, rReads));
                // }
                // else {
                    // set_union(edge_set, add_write_dependencies(dest_idx, nodeIn, nodeOut, rDep, rReads));
                    set_union(edge_set, add_write_dependencies(dest_idx, new_node, new_node, rDep, rReads));
                // }
                is_src_dest |= (dest_idx == src1);
            }
            if (!is_src_dest) {
                // if (dests.size() == 2) {
                //     // rReads[src1].push_back(nodeIn);
                //     rReads[src1].push_back(new_node);
                // }
                // else {
                    // rReads[src1].push_back(nodeOut);
                rReads[src1].push_back(new_node);
                // }
            }
            add_edge_from_set(partition, new_node, edge_set);

        } else {
            throw std::runtime_error("Invalid Instruction for rsv: " + instruction);
        }
    };

    // void GraphFusion::handle_mod(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType &rf, GraphFusion::RegisterFileType &rfMem, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_mod(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
        GraphFusion::DependencySet edge_set;
        std::pair<NodePtr,NodePtr> edge;
        
        std::smatch match;
        auto instructions_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, mod_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto dest_reg = std::stoul(match[1]);
            auto rns_base_id = std::stoul(match[3]);

            std::string srcs_str(match[2]);
            size_t pos = 0;
            // std::vector<LimbPtrT> srcs;
            std::vector<size_t> srcs_idx;
            size_t src_idx;
            while((pos = srcs_str.find(',')) != std::string::npos) {
                auto reg = srcs_str.substr(0,pos);
                src_idx = get_register_idx(reg);
                // srcs.push_back(rf[src_idx]);
                srcs_idx.push_back(src_idx);
                srcs_str.erase(0,pos+2);
            }
            src_idx = get_register_idx(srcs_str); 
            // srcs.push_back(rf[src_idx]);
            srcs_idx.push_back(src_idx);

            // auto & dest = rf[dest_reg];
            // auto new_node = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instructions_copy, opcode);
            // evaluators_[partition].mod(new_node, dest, srcs, rns_base_id);
            // rfMem[dest_reg] = nullptr;

            for (auto & src_idx : srcs_idx) {
                auto prev_src_write = rDep[src_idx];
                if (prev_src_write) {
                    edge = std::make_pair(prev_src_write,new_node);
                    edge_set.insert(edge);
                }
            }
            set_union(edge_set, add_write_dependencies(dest_reg, new_node, new_node, rDep, rReads));
            for (auto & src_idx : srcs_idx) {
                if (dest_reg != src_idx) {
                    rReads[src_idx].push_back(new_node);
                }
            }
            add_edge_from_set(partition, new_node, edge_set);

        } else {
            throw std::runtime_error("Invalid Instruction for mod: " + instruction);
        }
    };

    // void GraphFusion::handle_bci(const std::string &opcode, std::string &instruction, std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>> &bcu, const size_t partition) {
//         auto dest = get_dest(instruction);
//         if(dest[0] != 'B'){
//             throw std::runtime_error("Invalid Dest for bci: " + dest);
//         }
//         dest[0] = '0';
//         auto dest_bcu_id = std::stoul(dest);

//         size_t pos = 0;
//         pos = instruction.find("],");
//         auto dest_base_ids_string = instruction.substr(1,pos-1);
//         instruction = instruction.erase(0,pos+4);
//         auto src_base_ids_string = instruction.erase(instruction.length()-1);

//         auto string_to_uint64_vec = [&](std::string & string) {
//             std::vector<uint64_t> result;
//             size_t pos = 0;
//             while ((pos = string.find(",")) != std::string::npos) {
//                 auto num = std::stoul(string.substr(0,pos));
//                 result.push_back(num);
//                 string.erase(0,pos+1);
//             }
//             auto num = std::stoul(string);
//             result.push_back(num);
//             return result;
//         };
        
//         auto dest_base_ids = string_to_uint64_vec(dest_base_ids_string);
//         auto src_base_ids = string_to_uint64_vec(src_base_ids_string);

// #ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
//         std::cout << "BCI B" << dest_bcu_id << ":" << src_base_ids_string << " | " << dest_base_ids_string << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
// #endif
//         bcu[dest_bcu_id]->init(graphs_[partition],src_base_ids,dest_base_ids,streams_[partition]);
//     };

    // void GraphFusion::handle_pl1(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType & rf, GraphFusion::RegisterFileType & rfMem, std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>> &bcu, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
     void GraphFusion::handle_pl1(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {

        std::smatch match;
        auto instructions_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, pl1_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto dest_bcu_id = std::stoul(match[1]);
            auto reg1 = get_register_idx(match[2]);
            // auto src1 = get_src_reg(reg1,rf,rfMem);
            auto rns_base_id = std::stoul(match[4]);

            // auto new_node1 = CUDA::createNode();
            // auto new_node2 = CUDA::createNode();
            auto new_node = std::make_shared<Node>(instructions_copy, opcode);
            // if(match[3].length() != 0){
            //     // evaluators_[partition].pl1(new_node1, new_node2, rf[reg1], *bcu[dest_bcu_id], rns_base_id);
            //     // evaluators_[partition].pl1(new_node1, new_node2, src1, *bcu[dest_bcu_id], rns_base_id);
            //     evaluators_[partition].pl1(new_node, new_node, src1, *bcu[dest_bcu_id], rns_base_id);
            // }
            // else {
            //     // evaluators_[partition].pl1(new_node1, new_node2, rf[reg1], *bcu[dest_bcu_id], rns_base_id);
            //     // evaluators_[partition].pl1(new_node1, new_node2, src1, *bcu[dest_bcu_id], rns_base_id);
            //     evaluators_[partition].pl1(new_node, new_node, src1, *bcu[dest_bcu_id], rns_base_id);
            // }

            auto prev_src_write = rDep[reg1];
            if (prev_src_write) {
                // CUDA::addSingleEdge(graphs_[partition],prev_src_write,new_node1);
                roots_[partition]->insert(prev_src_write); //??
                prev_src_write->insert(new_node);
            }
            rReads[reg1].push_back(new_node);

        } else {
            throw std::runtime_error("Invalid Instruction for pl1: " + instruction);
        }
    };

    // void GraphFusion::handle_bcw(const std::string &opcode, const std::string &instruction, GraphFusion::RegisterFileType & rf, GraphFusion::RegisterFileType & rfMem, std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>> &bcu, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {
    void GraphFusion::handle_bcw(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition) {

        std::smatch match;
        auto instruction_copy = instruction;
        if (std::regex_search(instruction.begin(), instruction.end(), match, bcw_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            for (auto m : match)
                std::cout << "  submatch " << m << '\n';
#endif

            auto dest_bcu_id = std::stoul(match[1]);
            auto reg1 = get_register_idx(match[2]);
            // auto src1 = get_src_reg(reg1,rf,rfMem);
            auto rns_base_id = std::stoul(match[4]);

            // auto new_node1 = CUDA::createNode();
            auto new_node1 = std::make_shared<Node>(instruction_copy, opcode);
            // evaluators_[partition].bcw(new_node1, src1, *bcu[dest_bcu_id], rns_base_id);
            auto prev_src_write = rDep[reg1];
            // if (!prev_src_write) {
            //     throw std::runtime_error("Src not found in rDep (in int op)");
            // }
            // CUDA::addSingleEdge(graphs_[partition],prev_src_write,new_node1);
            if (prev_src_write) {
                roots_[partition]->insert(prev_src_write); //??
                prev_src_write->insert(new_node1);
                // CUDA::addSingleEdge(graphs_[partition],prev_src_write,new_node1);
            }
            rReads[reg1].push_back(new_node1);

        } else {
            throw std::runtime_error("Invalid Instruction for pl1: " + instruction);
        }
    };


    void GraphFusion::run_program_multithread(const std::string &instruction_file_base, const uint8_t partitions, const uint32_t registers) {

        // CUDA::deviceSynchronize();

        // if(partitions > 1) {
        //     CUDA::ncclInit_graph();
        // }


        // bool BUTTERSCOTCH_USE_CUDA_GRAPHS = get_boolean_env_variable("BUTTERSCOTCH_USE_CUDA_GRAPHS");
        bool BUTTERSCOTCH_USE_CUDA_GRAPHS = true;
        bool BUTTERSCOTCH_PRINT_GRAPH = get_boolean_env_variable("BUTTERSCOTCH_PRINT_GRAPH");

        std::cout << "Starting Program\n" << std::flush;

        // std::vector<std::vector<std::shared_ptr<LimbT>>> register_files;
        // std::vector<std::vector<std::shared_ptr<LimbT>>> registers_in_memory;
        // std::vector<std::vector<LimbT::Element_t>> register_files_scalar;
        std::vector<uint64_t> sync_id;
        std::map<uint64_t,uint64_t> sync_count;
        // std::map<uint64_t,LimbPtrT> dis_value;
        std::map<uint64_t,uint64_t> dis_value_rns_base_id;
        std::map<uint64_t,size_t> dis_source;
        std::map<uint64_t,std::map<uint8_t,uint64_t>> rcv_pending;
        std::vector<bool> advance;

        // Vectors to store dependencies (nodes) depending on last writes to registers/base converters
        std::vector<std::vector<NodePtr>> depReg;
        std::vector<std::unordered_map<std::string, NodePtr>> depLocalMem;

        // Vectors to store all previous reads before a write (for false dependencies)
        std::vector<std::vector<std::vector<NodePtr>>> readNodesReg;
        std::vector<std::unordered_map<std::string, std::vector<NodePtr>>> readNodesLocalMem;

        auto handle_dis = [&](const std::string &opcode, const std::string &instruction, size_t partition) {
            GraphFusion::DependencySet edge_set;
            std::pair<NodePtr,NodePtr> edge;

            std::smatch match;
            auto instruction_copy = instruction;
            std::unique_lock<std::mutex> lock(sync_lock);

            if (std::regex_search(instruction.begin(), instruction.end(), match, dis_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                for (auto m : match)
                    std::cout << "  submatch " << m << '\n';
#endif
                auto sync_id_inst = std::stoul(match[1]);
                auto sync_count_inst = std::stoul(match[2]);
                auto reg1_idx = get_register_idx(match[3]);
                // auto reg1 = register_files[partition][reg1_idx];
                // auto reg1 = get_src_reg(reg1_idx,register_files[partition],registers_in_memory[partition]);
                // LimbPtrT dis_src = nullptr;
                // LimbPtrT dis_dest = nullptr;
                {
                    // std::lock_guard<std::mutex> lock(sync_lock);
                    if(sync_id[partition] == 0){
                        sync_id[partition] = sync_id_inst;
                        advance[partition] = false;
                        sync_count[sync_id_inst]++;
                        // if(dis_value.find(sync_id_inst) != dis_value.end()) {
                        //     throw std::runtime_error("dis_value is not nullptr");
                        // }
                        // // dis_value[sync_id_inst] = evaluators_[partition].copy(reg1, reg1->rns_base_id());
                        // dis_value[sync_id_inst] = reg1;
                        // dis_value_rns_base_id[sync_id_inst] = reg1->rns_base_id();
                        dis_source[sync_id_inst] = partition;
                        // dis_src = dis_value[sync_id_inst];
                        // dis_dest = register_files[partition][reg1_idx];

                        // auto destNode = CUDA::createNode();
                        auto destNode = std::make_shared<Node>(instruction_copy, opcode);
                        auto srcNode = depReg[partition][reg1_idx];
                        lock.unlock();
                        // auto nccl_edges = CUDA::ncclBroadcast_uint32_graph(srcNode,destNode,dis_src->data(),dis_dest->data(),dis_src->size(),partition,partition,streams_[partition],graphs_[partition]);
                        lock.lock();
                        // dis_value[sync_id_inst] = dis_dest;
                        // registers_in_memory[partition][reg1_idx] = nullptr;

                        // Moved nccl edge creation outside to avoid duplicate edges
                        // for (auto & p : nccl_edges) {
                        //     edge_set.insert(p);
                        // }
                        if (srcNode) {
                            edge = std::make_pair(srcNode,destNode);
                            edge_set.insert(edge);
                        }
                        readNodesReg[partition][reg1_idx].push_back(destNode);

                        advance[partition] = true;
                        sync_id[partition] = 0;
                        add_edge_from_set(partition, destNode, edge_set);
                    } else if(sync_id[partition] != sync_id_inst){
                        throw std::runtime_error("Mismatched sync_id");
                    }
                }
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                    std::cout << "DIS @ " << sync_id_inst << " : " << match[3] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
#endif

                {
                // std::lock_guard<std::mutex> lock(sync_lock);
                if(sync_count[sync_id_inst] == sync_count_inst) {
                    sync_count.erase(sync_id_inst);
                }
                }
            } else {
                throw std::runtime_error("Invalid Instruction for dis: " + instruction);
            }
        };

// #if 0
//         auto handle_dis_old = [&](const std::string &opcode, const std::string &instruction, size_t partition) {
//             std::smatch match;
//             std::lock_guard<std::mutex> lock(sync_lock);

//             if (std::regex_search(instruction.begin(), instruction.end(), match, dis_regex)) {

// #ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
//                 for (auto m : match)
//                     std::cout << "  submatch " << m << '\n';
// #endif
//                 auto sync_id_inst = std::stoul(match[1]);
//                 auto sync_count_inst = std::stoul(match[2]);
//                 auto reg1 = get_register(match[3], register_files[partition]);
//                 auto reg1_idx = get_register_idx(match[3]);

//                 if(sync_id[partition] == 0){
//                     sync_id[partition] = sync_id_inst;
//                     advance[partition] = false;
//                     sync_count[sync_id_inst]++;
//                     if(dis_value[sync_id_inst] != nullptr) {
//                         throw std::runtime_error("dis_value is not nullptr");
//                     }
//                     // if(match[4].length() == 0){
//                     //     dis_value[sync_id_inst] = std::move(reg1);
//                     // } else {
//                         // CUDA::deviceSynchronize();
//                         // dis_value[sync_id_inst] = evaluators_[partition].copy(reg1, reg1->rns_base_id());
//                         dis_value[sync_id_inst] = evaluators_[partition].copy(reg1, reg1->rns_base_id());
//                         // dis_value[sync_id_inst] = reg1;
//                         auto & src = dis_value[sync_id_inst];
//                         // CUDA::ncclBroadcast_uint32(src->data(),src->data(),src->size(),partition,partition,streams_[partition]);
//                         auto destNode = CUDA::createNode();
//                         auto srcNode = depReg[partition][reg1_idx];
//                         // CUDA::ncclBroadcast_uint32_graph(srcNode,destNode,src->data(),src->data(),src->size(),partition,partition,streams_[partition],graphs_[partition]);
//                         depReg[partition][reg1_idx] = destNode;

//                         // if (prev_src1_write) {
//                         //     CUDA::addSingleEdge(graphs_[partition],prev_src1_write,destNode);
//                         // }
//                     std::cout << "DIS @ " << sync_id_inst << " : " << match[3] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
// #ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
//                     std::cout << "DIS @ " << sync_id_inst << " : " << match[3] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
// #endif
//                         // auto reg1_host = dis_value[sync_id_inst]->move_to_host_ptr();
//                         // std::cout << "DIS: " << reg1_host->data()[reg1->size()-1] << "\n";
//                         // dis_value[sync_id_inst] = reg1;
//                         dis_source[sync_id_inst] = partition;
//                         // CUDA::deviceSynchronize();
//                     // }
//                     // partition = (partition + 1) % partitions;
//                     return;
//                 } else if(sync_id[partition] != sync_id_inst){
//                     throw std::runtime_error("Mismatched sync_id");
//                 }

//                 advance[partition] = true;
//                 sync_id[partition] = 0;
//                 if(sync_count[sync_id_inst] == sync_count_inst) {
//                     // advance[partition] = true;
//                     // sync_id[partition] = 0;
//                     // dis_value.erase(sync_id_inst);
//                     // dis_source.erase(sync_id_inst);
//                     sync_count.erase(sync_id_inst);
//                 }
//                 // partition = (partition + 1) % partitions;

//             } else {
//                 throw std::runtime_error("Invalid Instruction for dis: " + instruction);
//             }
//         };
//         #endif

        auto handle_rcv = [&](const std::string &opcode, const std::string &instruction, size_t partition) {
            GraphFusion::DependencySet edge_set;
            std::pair<NodePtr,NodePtr> edge;

            std::smatch match;
            auto instruction_copy = instruction;
            std::lock_guard<std::mutex> lock(sync_lock);
            if (std::regex_search(instruction.begin(), instruction.end(), match, rcv_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                for (auto m : match)
                    std::cout << "  submatch " << m << '\n';
#endif
                auto sync_id_inst = std::stoul(match[1]);
                auto sync_count_inst = std::stoul(match[2]);

                if(sync_id[partition] == 0){
                    sync_id[partition] = sync_id_inst;
                    advance[partition] = false;
                    // partition = (partition + 1) % partitions;
                    return;
                } else if(sync_id[partition] != sync_id_inst){
                    throw std::runtime_error("Mismatched sync_id");
                }

                // if(dis_value.find(sync_id_inst) == dis_value.end()){
                //     // partition = (partition + 1) % partitions;
                //     return;
                // }

                auto dest_reg = std::stoul(match[3]);
                // const auto & val = dis_value.at(sync_id_inst);
                // CUDA::deviceSynchronize();
                // auto & dest = register_files[partition][dest_reg];
                // auto size = val->size();
                auto src_device = dis_source.at(sync_id_inst);
                // CUDA::deviceSynchronize();
                // CUDA::memcpyPeer(dest->data(),partition,val->data(),src_device,sizeof(GraphFusionLimb::Element_t)*size);
                // CUDA::ncclBroadcast_uint32(nullptr,dest->data(),dest->size(),dis_source[sync_id_inst],partition,streams_[partition]);
                // auto destNode = CUDA::createNode();
                auto destNode = std::make_shared<Node>(instruction_copy, opcode);
                // auto nccl_edges = CUDA::ncclBroadcast_uint32_graph(nullptr, destNode, nullptr,dest->data(),dest->size(),dis_source[sync_id_inst],partition,streams_[partition],graphs_[partition]);
                // registers_in_memory[partition][dest_reg] = nullptr; // write to actual register
                // for (auto & p : nccl_edges) {
                //     edge_set.insert(p);
                // }
                set_union(edge_set, add_write_dependencies(dest_reg, destNode, destNode, depReg[partition], readNodesReg[partition]));
                add_edge_from_set(partition, destNode, edge_set);
                // auto reg1_host = dest->move_to_host_ptr();
                // std::cout << "DIS: " << reg1_host->data()[dest->size()-1] << "\n";

                // dest->set_rns_base_id(dis_value_rns_base_id.at(sync_id_inst));
                // if(val->is_ntt_form()) {
                //     dest->set_form_ntt();
                // } else {
                //     dest->set_form_coef();
                // }
                sync_count[sync_id_inst]++;
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
            std::cout << "RCV @ " << sync_id_inst << " R" << match[3] << ":\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
#endif

                sync_id[partition] = 0;
                advance[partition] = true;
                if(sync_count[sync_id_inst] == sync_count_inst) {
                    sync_count.erase(sync_id_inst);
                    // dis_value.erase(sync_id_inst);
                }
                // partition = (partition + 1) % partitions;

            } else {
                throw std::runtime_error("Invalid Instruction for dis: " + instruction);
            }
        };

//         // std::map<uint64_t,LimbPtrT> joi_value;
//         std::map<uint64_t,LimbPtr> joi_value;
        std::optional<std::pair<uint8_t,uint64_t>> joi_destination;
//         auto handle_joi = [&](const std::string &opcode, const std::string &instruction, size_t partition) {

//             std::smatch match;
//             std::lock_guard<std::mutex> lock(sync_lock);
//             if (std::regex_search(instruction.begin(), instruction.end(), match, joi_regex)) {

// #ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
//                 for (auto m : match)
//                     std::cout << "  submatch " << m << '\n';
// #endif
//                 auto sync_id_inst = std::stoul(match[1]);
//                 auto sync_count_inst = std::stoul(match[2]);

//                 auto rns_base_id = std::stoul(match[7]);

//                 if(sync_id[partition] == 0){
//                     sync_id[partition] = sync_id_inst;
//                     advance[partition] = false;
//                     sync_count[sync_id_inst]++;
//                     auto src_reg = get_register(match[5],register_files[partition]);
//                     assert(src_reg);
//                     // CUDA::deviceSynchronize();
//                     if(joi_value.find(sync_id_inst) == joi_value.end()){
//                         auto data = Util::allocate_uint2(src_reg->size());
//                         CUDA::memcpyGraphFusionToHostAsync(data.get(),src_reg->data(),src_reg->size()*sizeof(Limb::Element_t),streams_[partition]);
//                         joi_value[sync_id_inst] = std::make_shared<Limb>(std::move(data),src_reg->size(),rns_base_id,src_reg->is_ntt_form());
//                     } else {
//                         auto temp_data = Util::allocate_uint2(src_reg->size());
//                         CUDA::memcpyGraphFusionToHostAsync(temp_data.get(),src_reg->data(),src_reg->size()*sizeof(Limb::Element_t),streams_[partition]);
//                         auto joi_data = joi_value[sync_id_inst]->data();
//                         for(size_t i = 0; i < src_reg->size(); i++) {
//                             joi_data[i] = Util::add_uint_mod(joi_data[i],temp_data[i],context_.get_rns_modulus(rns_base_id));
//                         }

//                     }
//                     // CUDA::deviceSynchronize();
//                     if(match[3].length() == 0) {
//                         advance[partition] = true;
//                         sync_id[partition] = 0;
//                     }
//                     // partition = (partition + 1) % partitions;
//                     return;
//                 } else if(sync_id[partition] != sync_id_inst){
//                     throw std::runtime_error("Mismatched sync_id");
//                 }


//                 // sync_id[partition] = 0;
//                 if(sync_count[sync_id_inst] == sync_count_inst) {
//                     if(match[3].length() != 0){
//                         auto & dest_reg = register_files[partition][std::stoul(match[4])];
//                         auto & joi_val = joi_value.at(sync_id_inst);
//                         // CUDA::deviceSynchronize();
//                         // evaluators_[partition].copy(dest_reg,joi_val,joi_val->rns_base_id());
//                         CUDA::memcpyHostToGraphFusionAsync(dest_reg->data(),joi_val->data(),sizeof(GraphFusionLimb::Element_t)*joi_val->size(),streams_[partition]);
//                         // CUDA::deviceSynchronize();
//                         // CUDA::memcpyHostToGraphFusionAsync(dest_reg->data(),joi_val->data(),sizeof(GraphFusionLimb::Element_t)*joi_val->size());
//                         // CUDA::deviceSynchronize();
//                         dest_reg->set_rns_base_id(rns_base_id);
//                         if(joi_val->is_ntt_form()){
//                             dest_reg->set_form_ntt();
//                         } else {
//                             dest_reg->set_form_coef();
//                         }
//                         // CUDA::deviceSynchronize();
                        
//                         // dest = std::move(joi_value.at(sync_id_inst));
//                         advance[partition] = true;
//                         sync_id[partition] = 0;
//                         sync_count.erase(sync_id_inst);
//                         // joi_value.erase(sync_id_inst);
//                     }
//                 }
//                 // partition = (partition + 1) % partitions;

// #ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
//         if(match[3].length() == 0){
//             std::cout << "JOI @ " << sync_id_inst << " : " << match[5] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
//         } else {
//             std::cout << "JOI @ " << sync_id_inst << " R" << match[4] << " : " << match[5] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
//         }
// #endif

//             } else {
//                 throw std::runtime_error("Invalid Instruction for joi: " + instruction);
//             }
//         };

        auto handle_joi_nccl = [&](const std::string &opcode, const std::string &instruction, size_t partition) {
            GraphFusion::DependencySet edge_set;
            std::pair<NodePtr,NodePtr> edge;

            std::smatch match;
            // std::lock_guard<std::mutex> lock(sync_lock);
            auto instruction_copy = instruction;
            std::unique_lock<std::mutex> lock(sync_lock);
            if (std::regex_search(instruction.begin(), instruction.end(), match, joi_regex)) {

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                for (auto m : match)
                    std::cout << "  submatch " << m << '\n';
#endif
                auto sync_id_inst = std::stoul(match[1]);
                auto sync_count_inst = std::stoul(match[2]);

                auto rns_base_id = std::stoul(match[7]);

                if(sync_id[partition] == 0) {
                    sync_id[partition] = sync_id_inst;
                } 
                    advance[partition] = false;
                    {
                    // std::lock_guard<std::mutex> lock(sync_lock);
                    sync_count[sync_id_inst]++;
                    }
                    auto src_reg_idx = get_register_idx(match[5]);
                    // auto src_reg = get_src_reg(src_reg_idx,register_files[partition],registers_in_memory[partition]);
                    // assert(src_reg);
                    if(match[3].length() != 0){
                        auto dest_reg_idx = std::stoul(match[4]);
                        // auto & dest_reg = register_files[partition][dest_reg_idx];
                        // auto destNode = CUDA::createNode();
                        auto destNode = std::make_shared<Node>(instruction_copy, opcode);
                        auto srcNode = depReg[partition][src_reg_idx];
                        lock.unlock();
                        // auto nccl_edges = CUDA::ncclReduce_uint32_graph(srcNode,destNode,src_reg->data(),dest_reg->data(),src_reg->size(),partition,partition,streams_[partition],graphs_[partition]);
                        lock.lock();
                        // registers_in_memory[partition][dest_reg_idx] = nullptr; // write to actual register
                        // for (auto & p : nccl_edges) {
                        //     edge_set.insert(p);
                        // }
                        if (srcNode) {
                            edge = std::make_pair(srcNode,destNode);
                            edge_set.insert(edge);
                        }
                        set_union(edge_set, add_write_dependencies(dest_reg_idx, destNode, destNode, depReg[partition], readNodesReg[partition]));
                        if (src_reg_idx != dest_reg_idx) {
                            readNodesReg[partition][src_reg_idx].push_back(destNode);
                        }
                        add_edge_from_set(partition, destNode, edge_set);
                        // auto modNode = CUDA::createNode();
                        auto modNode = std::make_shared<Node>(instruction_copy, opcode);
                        // evaluators_[partition].mod(modNode,dest_reg,dest_reg,rns_base_id);
                        // CUDA::addSingleEdge(graphs_[partition],destNode,modNode); // NOTE: no other write-read or write-write or read-write relationships
                        destNode->insert(modNode);
                        depReg[partition][dest_reg_idx] = modNode;

                        // dest_reg->set_rns_base_id(rns_base_id);
                        // dest_reg->set_form_ntt();
                    } else {
                        // TODO: fix this to be a global partition, need to identify the source and destination
                        auto root = partition == 1 ? 0: 1;
                        // auto destNode = CUDA::createNode();
                        auto destNode = std::make_shared<Node>(instruction_copy, opcode);
                        auto srcNode = depReg[partition][src_reg_idx];
                        lock.unlock();
                        // auto nccl_edges = CUDA::ncclReduce_uint32_graph(srcNode,destNode,src_reg->data(),nullptr,src_reg->size(),root,partition,streams_[partition],graphs_[partition]);
                        lock.lock();
                        // for (auto & p : nccl_edges) {
                        //     edge_set.insert(p);
                        // }
                        if (srcNode) {
                            edge = std::make_pair(srcNode,destNode);
                            edge_set.insert(edge);
                        }
                        add_edge_from_set(partition, destNode, edge_set);
                        readNodesReg[partition][src_reg_idx].push_back(destNode);
                    }
                advance[partition] = true;
                sync_id[partition] = 0;


                // partition = (partition + 1) % partitions;

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
        if(match[3].length() == 0){
            std::cout << "JOI @ " << sync_id_inst << " : " << match[5] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
        } else {
            std::cout << "JOI @ " << sync_id_inst << " R" << match[4] << " : " << match[5] << "\n"; // << " :" << src1 << ":" << src2 << ":" << rns_id << ":\n";
        }
#endif

            } else {
                throw std::runtime_error("Invalid Instruction for joi: " + instruction);
            }
        };
 
        // end of helper definitions

        std::vector<uint64_t> instruction_count;

        const auto num_base_conversion_units = 2;
        // register_files.resize(partitions);
        // registers_in_memory.resize(partitions);
        // register_files_scalar.resize(partitions);
        // base_conversion_units_.resize(partitions);
        depReg.resize(partitions);
        depLocalMem.resize(partitions);
        readNodesReg.resize(partitions);
        readNodesLocalMem.resize(partitions);

        std::vector<std::ifstream> instruction_files;

        std::vector<std::string> instructions;
        std::vector<bool> completed;
        size_t complete_count = 0;

        auto rf_init_thread_fn = [&](size_t tid) {
            // CUDA::setGraphFusion(tid);
            // register_files.at(tid).resize(registers);
            // registers_in_memory.at(tid).resize(registers);
            // register_files_scalar.at(tid).resize(registers);
            // base_conversion_units_.at(tid).resize(num_base_conversion_units);
            depReg.at(tid).resize(registers);
            readNodesReg.at(tid).resize(registers);

            // for (size_t j = 0; j < num_base_conversion_units; j++) {
            //     base_conversion_units_[tid][j] = std::make_unique<GraphFusion::GraphFusionBaseConverterGraph>(context_);
            // }
            // size_t coeff_count_ = context_.n();
            // for (size_t j = 0; j < registers; j++) {
            //     auto data = GraphFusionUtils::allocate_uint(coeff_count_);
            //     register_files[tid][j] = std::make_shared<GraphFusionLimb>(std::move(data),coeff_count_,-1,false);
            // }
            // auto stream = CUDA::createStream();
            // auto graph = CUDA::createGraph();
            auto root_node = std::make_shared<Node>();
            {
                std::lock_guard<std::mutex> lock(sync_lock);
                // streams_.push_back(stream);
                // graphs_.push_back(graph);
                roots_.push_back(root_node);
                // evaluators_.push_back(std::move(EvaluatorGraph(context_,stream,graph)));
            }
            // CUDA::deviceSynchronize();
        };


        // std::vector<std::thread> rf_init_threads;
        // for(size_t tid = 0; tid < partitions; tid++) {
        //     rf_init_threads.push_back(std::thread(rf_init_thread_fn,tid));
        // }

        // for(size_t tid = 0; tid < partitions; tid++) {
        //     rf_init_threads[tid].join();
        // }

        for(size_t tid = 0; tid < partitions; tid++) {
            rf_init_thread_fn(tid);
        }

        for (size_t i = 0; i < partitions; i++) {
            auto instruction_file_name = instruction_file_base + std::to_string(i);
            std::ifstream ifile(instruction_file_name, std::ios::in);
            instruction_files.push_back(std::move(ifile));
            instruction_count.push_back(0);
            std::string instruction;
            std::getline(instruction_files[i], instruction);
            instructions.push_back("");
            advance.push_back(true);
            sync_id.push_back(0);
            completed.push_back(false);
        }


        auto thread_fn = [&](std::size_t partition){

            // CUDA::setGraphFusion(partition);
            // if(BUTTERSCOTCH_USE_CUDA_GRAPHS){
            //     CUDA::graphBeginCapture(streams_[partition],partition);
            // }
            std::string instruction;
            while(true){
                if(advance[partition]){
                    if(!std::getline(instruction_files.at(partition), instruction)){
                        // break;
                        std::lock_guard<std::mutex> lock(sync_lock);
                        if(completed[partition] == false){
                            complete_count++;
                            completed[partition] = true;
                        }
                        if(complete_count == partitions){
                            break;
                        } else {
                            // partition = (partition + 1) % partitions;
                            continue;
                        }
                    }
                    instruction_count[partition]++;
                    instructions[partition] = instruction;
                }
                instruction = instructions[partition];
                // auto &register_file = register_files[partition];
                // auto &register_local_mem = registers_in_memory[partition];
                // auto &register_file_scalar = register_files_scalar[partition];
                // auto &base_conversion_unit = base_conversion_units_[partition];
                auto &register_dep = depReg[partition];
                auto &local_mem_dep = depLocalMem[partition];
                auto &regReadNodes = readNodesReg[partition];
                auto &localMemReadNodes = readNodesLocalMem[partition];

#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                std::cout << "P" << partition << ": " <<  instruction << "\n";
#endif

                if(instruction_count[partition] % 100000 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - begin_).count();
                    std::cout << "HEARTBEAT:" << partition << " @ " << duration << " seconds " << instruction_count[partition]/(1000) << "K instructions\n" << std::flush;
                }
                auto opcode = get_opcode(instruction);
                if (opcode == "load") {
                    // handle_load(instruction, opcode, register_file, register_local_mem, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                    handle_load(instruction, opcode, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                } else if (opcode == "loas") {
                    // this case does not matter in our graph for now
                    // handle_loas(instruction, register_file_scalar, partition);
                } else if (opcode == "store") {
                    // handle_store(instruction, opcode, register_file, register_local_mem, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                    handle_store(instruction, opcode, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                } else if (opcode == "spill") {
                    // handle_store(instruction, opcode, register_file,  register_local_mem, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                    handle_store(instruction, opcode, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                } else if (opcode == "evg") {
                    // handle_evg(instruction, opcode, register_file, register_local_mem, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                    handle_evg(instruction, opcode, register_dep, local_mem_dep, regReadNodes, localMemReadNodes, partition);
                } else if (opcode == "add") {
                    // handle_binop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_binop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "ads") {
                    // handle_binop_scalar(opcode, instruction, register_file, register_local_mem, register_file_scalar, register_dep, regReadNodes, partition);
                    handle_binop_scalar(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "sub") {
                    // handle_binop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_binop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "sus") {
                    // handle_binop_scalar(opcode, instruction, register_file, register_local_mem, register_file_scalar, register_dep, regReadNodes, partition);
                    handle_binop_scalar(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "neg") {
                    // handle_unop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition);  
                    handle_unop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "mov") {
                    // handle_mov(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition);
                    handle_mov(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "mup") {
                    // handle_binop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_binop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "mus") {
                    // handle_binop_scalar(opcode, instruction, register_file, register_local_mem, register_file_scalar, register_dep, regReadNodes, partition);
                    handle_binop_scalar(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "mul") {
                    // handle_binop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_binop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "int") {
                    // handle_ntt(opcode, instruction, register_file, register_local_mem, base_conversion_unit, register_dep, regReadNodes, partition); 
                    handle_ntt(opcode, instruction, register_dep, regReadNodes, partition); 
                } else if (opcode == "ntt") {
                    // handle_ntt(opcode, instruction, register_file, register_local_mem, base_conversion_unit, register_dep, regReadNodes, partition);  
                    handle_ntt(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "sud") {
                    // handle_sud(opcode, instruction, register_file, register_local_mem, base_conversion_unit, register_dep, regReadNodes, partition); 
                    handle_sud(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "rot") {
                    // handle_rot(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition);
                    handle_rot(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "con") {
                    // handle_unop(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_unop(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "rsi") {
                    // handle_rsi(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_rsi(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "rsv") {
                    // handle_rsv(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_rsi(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "mod") {
                    // handle_mod(opcode, instruction, register_file, register_local_mem, register_dep, regReadNodes, partition); 
                    handle_mod(opcode, instruction, register_dep, regReadNodes, partition); 
                } else if (opcode == "bci") {
                    // handle_bci(opcode, instruction, base_conversion_unit, partition);
                } else if (opcode == "bcw") {
                    // handle_bcw(opcode, instruction, register_file, register_local_mem, base_conversion_unit, register_dep, regReadNodes, partition); 
                    handle_bcw(opcode, instruction, register_dep, regReadNodes, partition);
                } else if (opcode == "pl1") {
                    // handle_pl1(opcode, instruction, register_file, register_local_mem, base_conversion_unit, register_dep, regReadNodes, partition); 
                    handle_pl1(opcode, instruction, register_dep, regReadNodes, partition);
                // } else if (opcode == "pl2") {
                //     handle_pl2(opcode, instruction, register_file, base_conversion_unit);
                // } else if (opcode == "pl3") {
                //     handle_pl3(opcode, instruction, register_file, base_conversion_unit);
                // } else if (opcode == "pl4") {
                //     handle_pl4(opcode, instruction, register_file, base_conversion_unit);
                } else if (opcode == "dis") {
                    handle_dis(opcode, instruction, partition);
                } else if (opcode == "rcv") {
                    handle_rcv(opcode, instruction, partition);
                } else if (opcode == "joi") {
                    // handle_joi(opcode, instruction, partition);
                    // handle_joi_nccl(opcode, instruction, partition);
                } else {
                    throw std::runtime_error("Invalid opcode: " + opcode);
                }

                // CUDA::deviceSynchronize();
#ifdef BUTTERSCOTCH_INTERPRETER_VERBOSE
                std::cout << "\t" << opcode << ":"
                          << "\n";
                std::cout << "\t" << instruction << ":"
                          << "\n";
#endif
            }
            // CUDA::deviceSynchronize();
            // partition = (i + 1) % partitions;
            // if(BUTTERSCOTCH_USE_CUDA_GRAPHS){
            //     // CUDA::graphEndCapture(streams_[partition],partition);
            //     CUDA::explicitGraphInit(graphs_[partition],partition);
            // }
            // CUDA::deviceSynchronize();
        };

        begin_ = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        for(size_t tid = 0; tid < partitions; tid++) {
            threads.push_back(std::thread(thread_fn,tid));
        }

        for(size_t tid = 0; tid < partitions; tid++) {
            threads[tid].join();
        }

    if(BUTTERSCOTCH_USE_CUDA_GRAPHS){
        if (BUTTERSCOTCH_PRINT_GRAPH) {
            for(size_t tid = 0; tid < partitions; tid++) {
                // CUDA::graphDebugDotPrint(graphs_[tid],tid);
            }
        }
    }

    // auto graph_launch_thread_fn = [&](size_t tid) {
    //     CUDA::setGraphFusion(tid);
    //     CUDA::graphLaunch_graph(streams_[tid],tid);
    //     CUDA::deviceSynchronize();
    // };

    if(BUTTERSCOTCH_USE_CUDA_GRAPHS){
        // threads.clear();
        // std::cout << "Launching Graphs\n";
        // begin_ = std::chrono::steady_clock::now();
        // for(size_t tid = 0; tid < partitions; tid++) {
        //     threads.push_back(std::thread(graph_launch_thread_fn,tid));
        // }

        // for(size_t tid = 0; tid < partitions; tid++) {
        //     threads[tid].join();
        // }
    }

        end_ = std::chrono::steady_clock::now();
        // CUDA::deviceSynchronize();

    }

    bool GraphFusion::get_boolean_env_variable(const std::string & var) {
    if (const char *env_p = std::getenv(var.c_str())) {
      auto env_str = std::string(env_p);
      if(env_str == "1") {
        return true;
      } else {
        return false;
      }
    }
    return false;
  }

} // namespace GraphFusion