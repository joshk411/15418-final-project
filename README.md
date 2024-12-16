# 15418 Final Project: Accelerating Fully Homomorphic Encryption - Kernel Fusion

## Authors
Joshua Kim (jjkim3@andrew.cmu.edu) and Caleb Oh (choh@andrew.cmu.edu)

## Links
- [Link to Final Report](https://docs.google.com/document/d/1yCKJvy1uciiYdMDOvqjr0PVQJbHTDcozdU_APXeEuy4/edit?usp=sharing)

## Summary
We are going to create optimized implementations of fully homomorphic encryption operations on the GPU architecture, mainly focusing on kernel fusion techniques. We look to benchmark various fusion techniques against each other, including maximal fusion and data-dependent fusion.

## Background
In most modern systems, data is encrypted at rest and at transport, but is decrypted at use - making it vulnerable to leakage. Fully Homomorphic Encryption (FHE) is a privacy enhancing technology that tackles this privacy concern, allowing for computation on encrypted data without ever revealing its raw form. Though FHE meets the security requirements, it suffers from a huge performance overhead of close to 10,000 times when compared to plaintext on CPU. This performance deficit can be attributed to the increased width and depth in computation when working with encrypted data. Ciphertext is larger than plaintext (~10x larger), which creates more limbs, or independent components of the ciphertext. Additionally, as FHE applications grow in scale, we increase the multiplicative depth of our computation. Ciphertext multiplication accumulates errors, and eventually the errors create too much noise for an accurate computation. To refresh these errors, we utilize a method called bootstrapping - which is an extremely expensive operation that takes from 70% to 80% of execution time. Prior work on FHE, has identified program-level and limb-level parallelism, successfully creating instruction streams that minimize cross-stream dependencies. Additionally, more prior work (Josh’s research over the summer) created a baseline GPU implementation that targets the low-hanging fruits of the GPU architecture through mapping onto CUDA kernels, a kernel scheduling based on execution graphs, and multi-GPU support via communication methods. However, profiling the application with this baseline model revealed that we were spending approximately 60% of our execution time on various overheads. Of these overheads, one major source was kernel launch overhead, and thus this project attempts to tackle this issue by using kernel fusion.

For implementation background: The implementation of FHE usually involves data in the form of polynomials, which can be represented as vectors. Various operations are done on these vectors, including addition, FFT based multiplication, change in modulus base, and more. 

## Challenge
First, the inherent complexity of the theory behind the parallel system makes implementation difficult and time consuming. Second, prior profiling shows that kernel launch overheads in small FHE kernels account for approximately 60% of the execution time - making it hard to correctly analyze parallel performance. Third, as we scale to larger use cases and expand our parallelism horizontally, we find that we are inhibited by memory communications. Our plan is to tackle the dependencies in the workload and experiment with fusion strategies that maximally exploit data dependencies across small kernels to even out workload distribution and reduce memory transfer overhead. 

## Resources 
This project has been in progress for the past several months, during which we were able to create baseline implementations of these FHE operations on both the CPU and GPU (in C++ and CUDA). We will work off of this existing code base and look for various optimizations. In terms of compute resources, any system with an NVIDIA GPU will suffice (eg. GHC clusters), though we may look towards external systems if we wish to do multi-GPU tests. We are currently looking to implement optimizations of kernel fusion, of which we are gaining motivation from a general fusion paper, NNVM-fusion, and Triton's fusion techniques.

## Goals and Deliverables
We plan to achieve the following: 
- A complete implementation of a naive kernel fusion approach and at least one variation that build upon this naive approach. Some examples of variations include inner thread block vs inter thread block approaches (we are still investigating various fusion techniques utilized by various existing systems).
- A detailed analysis comparing our variations along with a FHE implementation in OpenAI's Triton language.

We hope to explore a wider range of fusion techniques as well as complete more testing at scale. If time permits, it would be great to target more sophisticated fusion techniques as well as fit our FHE operations into an end-to-end ML system. Additionally, if time permits, we would love to explore the memory inhibitions of our horizontal parallelism scheme, where we can explore various memory mapping and message passing techniques across GPUs.

Our analysis will center around comparing three main systems: the existing naive implementation, implementations using GPU-targeting DSLs, and our fusion based implementation. We will focus on various components, the main ones being differences in overhead (launch and memory), execution times, and speedups across these various implementations. This analysis will mainly be driven by the help of NVIDIA's profiling tool Nsight Systems. We will not have an interactive demo, as our project does not offer much to share in real-time, but we will display various comparative graphs based on the above metrics across the various systems.

## Platform Choice
We will be using C++ and CUDA for our system implementation as well as Python to interact with existing DSLs. We can complete most of our development and execution locally on GHC machines. However, when completing larger performence tests, especially ones that require multiple GPUs, we will require external systems - perhaps PSC or Entropy (if given permission).

## Schedule 
| Dates | Checkpoint |
| ----------- | ----------- |
| 11/10 - 11/16 | Explore a variety of papers and open-source repositories and identify the kernel techniques we want to explore. |
| 11/17 - 11/23 | Complete the triton implementation to fit into bootstrapping for testing and make progress on bringing manual fusion to bootstrapping |
| 11/24 - 11/30 | Implement the kernel fusion techniques (non-naive) and start to benchmark to identfy bottlenecks. |
| 12/01 - 12/07 | Continue the refinement process of targeting bottlenecks and benchmarking. |
| 12/08 - 12/15 | Poster presentation and Final report |
