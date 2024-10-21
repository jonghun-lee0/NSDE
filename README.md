# How does noise help robustness? Explanation and exploration under the neural SDE framework

This repository contains the implementation of the paper **"How does noise help robustness? Explanation and exploration under the neural SDE framework"**. The code has been written to recreate the pipeline and experiments from the original paper. This is not the official implementation, but rather a re-implementation based on the paper.

## Paper Reference

- **Title**: How does noise help robustness? Explanation and exploration under the neural SDE framework  
- **Authors**: 
  - Xuanqing Liu, UCLA, xqliu@cs.ucla.edu
  - Tesi Xiao, UC Davis, texiao@ucdavis.edu
  - Si Si, Google, sisidaisy@google.com
  - Qin Cao, Google, qincao@google.com
  - Sanjiv Kumar, Google, sanjivk@google.com
  - Cho-Jui Hsieh, UCLA, Google, chohsieh@cs.ucla.edu
- **Conference**: CVPR 2020  
- **Paper Link**: [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_How_Does_Noise_Help_Robustness_Explanation_and_Exploration_under_the_CVPR_2020_paper.pdf)

## Getting Started

To get started, clone this repository and follow the instructions below to set up the necessary environment and run the code.

### Prerequisites

+ Ubuntu 20.04.6 LTS
+ conda 23.7.4
+ cuda 12.1
+ Python 3.10.13
+ PyTorch 2.1.0
+ torchvision 0.16.0
+ scikit-learn 1.3.2
+ numpy 1.26.0
+ scipy 1.11.3
+ tqdm 4.66.1

### Running the Code

To run the experiments, simply execute the `run.sh` script:

    ```bash
    ./run.sh
    ```

## Acknowledgments

This code is based on the paper **"How does noise help robustness? Explanation and exploration under the neural SDE framework"** by Liu et al. (CVPR 2020).  

If you use this code, please consider citing both the original paper and this repository:

- **Original Paper**: Liu, Xuanqing, et al. "How does noise help robustness? Explanation and exploration under the neural SDE framework." CVPR 2020.  
- **GitHub Repository**: [Link](https://github.com/jonghun-lee0/NSDE)
