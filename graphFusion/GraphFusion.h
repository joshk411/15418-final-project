#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <variant>
#include <regex>
#include <chrono>
#include <unordered_set>
#include <iostream>
#include <mutex>
#include <functional>

// #include "ButterscotchFHE-Viewing/butterscotchFHE/context.h"
// #include "butterscotchFHE/limb.h"

// #include "butterscotchFHE/device/device_limb.h"
// #include "butterscotchFHE/cudagraph/cuda_memory.h"

namespace GraphFusion {

struct Node {
public:
    // default constructor
    Node() : instruction_(""), opcode_("") {}

    // Constructor to initialize the instruction
    Node(const std::string& instruction, const std::string& opcode) : instruction_(instruction), opcode_(opcode) {}

    // Method to insert a child node
    void insert(std::shared_ptr<Node> child) {
        children_.push_back(child);
    }

    // Getter for the instruction
    std::string getInstruction() const {
        return instruction_;
    }

    // Getter for the opcode
    std::string getOpcode() const {
        return opcode_;
    }

    const std::vector<std::shared_ptr<Node>>& getChildren() const {
        return children_;
    }

    bool isEmpty() const {
        return instruction_.empty() && opcode_.empty();
    }

    void printGraph(std::shared_ptr<Node> node, int indent = 0) {
        for (int i = 0; i < indent; ++i) {
            std::cout << "  ";
        }
        std::cout << "Instruction : " << node->getInstruction() << " Opcode: " << node->getOpcode() << std::endl;

        for (const auto& child : node->getChildren()) {
            printGraph(child, indent + 1);
        }
    }

    void fuseGraph(std::shared_ptr<Node> root) {

        std::vector<std::vector<std::shared_ptr<Node>>> levels;
        std::queue<std::shared_ptr<Node>> queue;
        queue.push(root);

        while (!queue.empty()) {
            int levelSize = queue.size();
            std::vector<std::shared_ptr<Node>> levelNodes;
            std::vector<std::shared_ptr<Node>> moveNodes;

            // get all our nodes for this level
            for (int i = 0; i < levelSize; ++i) {
                auto node = queue.front();
                queue.pop();
                levelNodes.push_back(node);
            }
            
            for (int i = 0; i < levelSize; ++i) {
                auto node = levelNodes[i];
                for (const auto& n : levelNodes) {
                    for (const auto& child : node->getChildren()) {
                        if (child == n) {
                            auto it = std::find(levelNodes.begin(), levelNodes.end(), n);
                            levelNodes.erase(it);
                            queue.push(n);
                            moveNodes.push_back(n);
                        }
                    }
                }
                
                node->seen_ = true;
                
                for (const auto& child : node->getChildren()) {
                    if (!child->seen_ && !(std::find(moveNodes.begin(), moveNodes.end(), child) != moveNodes.end())) {
                        queue.push(child);
                    }
                }
            }

            moveNodes.clear();
            levels.push_back(levelNodes);
        }

        // Process the nodes level by level
        for (const auto& level : levels) {
            for (const auto& node : level) {
                processNode(node);
            }
        }
    }

    void processNode(std::shared_ptr<Node> node) {
        // Here you can process or print the node's instruction and opcode
        std::cout << "Instruction: " << node->getInstruction() << ", Opcode: " << node->getOpcode() << std::endl;
    }

private:
    std::string instruction_;
    std::string opcode_;
    std::vector<std::shared_ptr<Node>> children_;
    bool seen_ = false;
};

// helper type for the visitor #4
template <class... Ts>
struct overloaded1 : Ts... {
    using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded1(Ts...) -> overloaded1<Ts...>;
    class GraphFusion {

    public:
        using NodePtr = std::shared_ptr<Node>;
        // using LimbT = GraphFusionLimb;
        // using LimbPtrT = GraphFusionLimbPtr;
        // using MessageType = std::variant<std::vector<double>, std::vector<std::complex<double>>, double>;
        // (Vector[double] | Vector[complex] | double , scale)
        // using RawInputType = std::pair<MessageType, double>;

        // using RegisterFileType = std::vector<std::shared_ptr<LimbT>>;
        // using ScalarRegisterFileType = std::vector<LimbT::Element_t>;
        using DependencyType = std::vector<NodePtr>;
        using LocalDepType = std::unordered_map<std::string, NodePtr>;
        using ReadNodeType = std::vector<std::vector<NodePtr>>;
        using LocalReadNodeType = std::unordered_map<std::string, std::vector<NodePtr>>;

        struct DependencyHash {
            std::size_t operator()(const std::pair<NodePtr, NodePtr>& p) const {
                // auto hash1 = std::hash<const void*>{}(p.first.get());
                // auto hash2 = std::hash<const void*>{}(p.second.get());
                std::hash<std::string> stringHash;
                auto hash1 = stringHash(p.first->getInstruction());
                auto hash2 = stringHash(p.second->getInstruction());
                return hash1 ^ (hash2 << 1);
            }
        };
        struct DependencyEqual {
            bool operator()(const std::pair<NodePtr, NodePtr>& p1, const std::pair<NodePtr, NodePtr>& p2) const {
                // return (p1.first.get() == p2.first.get()) && (p1.second.get() == p2.second.get());
                return (p1.first->getInstruction() == p2.first->getInstruction()) && (p1.second->getInstruction() == p2.second->getInstruction());
            }
        };
        using DependencySet = std::unordered_set<std::pair<NodePtr,NodePtr>,DependencyHash,DependencyEqual>;

        // GraphFusion(const Context &context) : context_(context) {
        //     bcw_regex = std::regex("B([0-1]+): (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     pl1_regex = std::regex("B([0-1]+): (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     pl2_regex = std::regex("B([0-1]+), r([0-9]+): b([0-9]+)\\{([0-9]+)\\}, (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     pl3_regex = std::regex("B([0-1]+): (r[0-9]+(\\[X\\])?), (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     pl4_regex = std::regex("r([0-9]+): b([0-9]+)\\{([0-9]+)\\}, (r[0-9]+(\\[X\\])?), (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     rot_regex = std::regex("(-?[0-9]+) r([0-9]+): (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     ntt_regex = std::regex("r([0-9]+): ((r[0-9]+(\\[X\\])?)|b([0-9]+)\\{([0-9]+)\\}) \\| ([0-9]+)");
        //     unop_regex = std::regex("r([0-9]+): (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        //     sud_regex = std::regex("r([0-9]+): (r[0-9]+(\\[X\\])?), ((r[0-9]+(\\[X\\])?)|b([0-9]+)\\{([0-9]+)\\}) \\| ([0-9]+)");
        //     rsi_regex = std::regex("\\{(r([0-9]+), )*r([0-9]+)\\}");
        //     rsv_regex = std::regex("\\{(.*)\\}: (r[0-9]+(\\[X\\])?): \\[(.*)\\] \\| ([0-9]+)");
        //     mod_regex = std::regex("r([0-9]+): \\{(.*)\\} \\| ([0-9]+)");
        //     load_regex = std::regex("r([0-9]+): (.+\\([0-9]+\\))(\\{F\\})?");
        //     loas_regex = std::regex("s([0-9]+): (.+\\([0-9]+\\))(\\{F\\})?");
        //     dis_regex = std::regex("@ ([0-9]+):([0-9]+) : (r[0-9]+(\\[X\\])?)");
        //     rcv_regex = std::regex("@ ([0-9]+):([0-9]+) r([0-9]+):");
        //     joi_regex = std::regex("@ ([0-9]+):([0-9]+) (r([0-9]+))?: (r[0-9]+(\\[X\\])?) \\| ([0-9]+)");
        // }
        // void generate_inputs(const std::string &inputs_file_name_, const std::unordered_map<std::string, RawInputType> &raw_inputs, CKKSEncryptor &encryptor);
        // void generate_inputs(const std::string &inputs_file_name, const std::string & evalkeys_file_name, const std::unordered_map<std::string, RawInputType> &raw_inputs, CKKSEncryptor &encryptor);
        // void copy_inputs_to_device(const std::string &local_inputs_file_base, size_t num_partitons);
        // void run_program(const std::string &instruction_file, const uint8_t partitions, const uint32_t registers);
        void run_program_multithread(const std::string &instruction_file, const uint8_t partitions, const uint32_t registers);

        // void generate_and_serialize_evalkeys(const std::string & output_file_name, const std::string & input_file_name, CKKSEncryptor &encryptor);
        // std::unordered_map<std::string, std::pair<RnsPolynomialPtr, RnsPolynomialPtr>> deserialize_evalkeys(const std::string & evalkey_file_name);

        // BUTTERSCOTCH_DATATYPE_TEMPLATE(T)
        // template <typename T>
        // std::map<std::string,std::vector<T>> get_decrypted_outputs(CKKSEncryptor & encryptor, std::unordered_map<std::string,double> output_scales);
        // void decrypt_and_print_outputs(CKKSEncryptor & encryptor, std::unordered_map<std::string,double> output_scales);
        // void set_program_memory(const std::unordered_map<std::string,std::shared_ptr<Limb>> & memory) {
        //     program_memory_ = memory;
        // }
        // auto get_program_memory() {
        //     return program_memory_;
        // }

        // std::vector<std::pair<NodePtr,NodePtr>> ncclBroadcast_uint32_graph(const NodePtr & sendNode, NodePtr & recvNode, const void * sendbuff, void * recvbuff,  size_t count, int root, size_t device, const StreamPtr & stream, GraphPtr & graph ) {
        //     std::vector<std::pair<NodePtr,NodePtr>> res;
        //     cudaStreamBeginCapture(stream->stream_, cudaStreamCaptureModeThreadLocal);
        //     CHECK_CUDA_ERROR();
        //     auto cmd = ncclBroadcast(sendbuff, recvbuff, count, ncclUint32, root, comms_graph[device], stream->stream_);
        //     NCCLCHECK(cmd);
        //     CHECK_CUDA_ERROR();
        //     cudaGraph_t graph_brc_;
        //     cudaStreamEndCapture(stream->stream_, &graph_brc_);
        //     cudaGraphAddChildGraphNode ( &recvNode->node_, graph->graph_, nullptr, 0, graph_brc_);
        //     if(prev_ncc[device]) {
        //         // addSingleEdge(graph,prev_ncc[device],recvNode);
        //         std::pair<NodePtr,NodePtr> edge = std::make_pair(prev_ncc[device],recvNode);
        //         res.push_back(edge);
        //     }
        //     prev_ncc[device] = recvNode;
        //     CHECK_CUDA_ERROR();
        //     return res;
        // }

    private:
        // roots for our node struct
        std::vector<std::shared_ptr<Node>> roots_;
        
        // struct EvalkeyInfo {
        //     enum KeyType {
        //         Mul,
        //         Rot,
        //         Con,
        //         Boot,
        //         Ephemeral,
        //         Boot2
        //     } key_type;
        //     uint32_t level;
        //     uint32_t extension_size;
        //     std::vector<uint64_t> digit_partition;
        //     int32_t rotation_amount;
        //     uint8_t ct_number;
        //     std::string id;

        //     EvalkeyInfo(std::string key_info);
        // };
        // using EvalKeyEntryType = std::tuple<std::string,EvalkeyInfo,std::vector<std::size_t>>;

        // const Context &context_;
        // // Evaluator evaluator_;
        // std::vector<CUDA::StreamPtr> streams_;
        // std::vector<CUDA::GraphPtr> graphs_; // this is the root
        // std::vector<GraphFusion::EvaluatorGraph> evaluators_;

        // std::vector<std::unordered_map<std::string,std::shared_ptr<LimbT>>> program_memory_local_;
        // // std::string inputs_file_name_;
        // // std::unordered_map<std::string, std::shared_ptr<LimbT>> program_memory_;
        // std::unordered_map<std::string, std::shared_ptr<Limb>> program_memory_;
        // std::unordered_map<std::string, LimbT::Element_t> program_memory_scalar_;
        
        std::vector<std::unordered_set<std::string>> program_outputs_local_;
        std::unordered_map<std::string, std::tuple<std::string, std::string, std::vector<uint64_t>>> program_outputs_;
        // std::vector<std::vector<std::unique_ptr<GraphFusion::GraphFusionBaseConverterGraph>>> base_conversion_units_;
        std::chrono::steady_clock::time_point begin_;
        std::chrono::steady_clock::time_point end_;
        
        std::mutex sync_lock;
        std::mutex mem_lock;

        std::regex mod_regex;
        std::regex rsv_regex;
        std::regex rsi_regex;
        std::regex sud_regex;
        std::regex unop_regex;
        std::regex ntt_regex;
        std::regex rot_regex;
        std::regex bcw_regex;
        std::regex pl1_regex;
        std::regex pl2_regex;
        std::regex pl3_regex;
        std::regex pl4_regex;
        std::regex load_regex;
        std::regex loas_regex;
        std::regex dis_regex;
        std::regex rcv_regex;
        std::regex joi_regex;

        // void handle_evalkey_stream(std::ifstream & input_file, CKKSEncryptor &encryptor);
        // void handle_evalkey_stream(std::ifstream & input_file, const std::string &evalkeys_file_name);
        // EvalKeyEntryType parse_evalkey(std::string &line);
        std::string evalkey_header = "ButterscotchFHE::Evalkeys\n";

        std::string get_opcode(std::string& instruction);
        std::string get_dest(std::string& instruction);
        std::string get_src(std::string& instruction);
        int get_rns_id(std::string& instruction);
        // LimbPtrT get_register(const std::string &reg_str_, GraphFusion::RegisterFileType &rf);
        // LimbT::Element_t get_register_scalar(const std::string &reg_str_, GraphFusion::ScalarRegisterFileType &srf);

        void add_edge_from_set(const size_t partition, std::shared_ptr<Node> curr_node, const DependencySet &edge_set);

        void handle_load(const std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                        GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition);
        void handle_loas(const std::string &instruction, const size_t partition);
        void handle_evg(const std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                        GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition);
        void handle_store(std::string &instruction, const std::string &opcode, GraphFusion::DependencyType &rDep, GraphFusion::LocalDepType &localDep, 
                        GraphFusion::ReadNodeType &rReads, GraphFusion::LocalReadNodeType &localReads, const size_t partition);
        void handle_binop(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_binop_scalar(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_unop(const std::string &opcode, std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_mov(const std::string &opcode, const std::string &instruction,GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_rot(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_ntt(const std::string &opcode, const std::string &instruction,
                        GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_sud(const std::string &opcode, const std::string &instruction, 
                        GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_rsi(const std::string &opcode, const std::string &instruction,GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_rsv(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_mod(const std::string &opcode, const std::string &instruction, GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        // void handle_bci(const std::string &opcode, std::string &instruction, const size_t partition);
        void handle_pl1(const std::string &opcode, const std::string &instruction,
                        GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        void handle_bcw(const std::string &opcode, const std::string &instruction,
                        GraphFusion::DependencyType &rDep, GraphFusion::ReadNodeType &rReads, const size_t partition);
        // void handle_pl3(const std::string &opcode, const std::string &instruction, const size_t partition);
        // void handle_pl4(const std::string &opcode, const std::string &instruction, const size_t partition);

        bool get_boolean_env_variable(const std::string & var);
    };

} // namespace GraphFusion