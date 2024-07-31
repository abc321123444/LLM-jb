# Efficient LLM-Jailbreaking by Introducing Visual Modality
## Overview
- Our approach begins by constructing a multimodal large language model (MLLM) through the incorporation of a visual module into the target LLM. Subsequently, we conduct an efficient MLLM-jailbreak to
generate jailbreaking embeddings embJS. Finally, we convert the embJS into text space to facilitate the jailbreaking of the target LLM.
- Compared to direct LLM-jailbreaking, our approach is more efficient, as MLLMs are more vulnerable to jailbreaking than pure LLM.
<p align="center">
  <img src="figs/fig1.png" width="500">
</p>

## Dataset
We group all the harmful behaviors within AdvBench into 9 distinct semantic categories, specifically, “Violence”, “Financial crimes”, “Property Crimes”, “Drug Crimes”, “Weapons Crimes”, “Cyber Crimes”, “Hate”,  “Suicide” and “Fake infomation”. At the same time, given that our method involves jailbreaking multimodal large language models, we provide up to thirty images which were retrieved from the Internet using the Google search engine for each class as initialization of the images during the jailbreak process.

## Getting Started

### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/abc03570128/LLM-jb.git
cd LLM-jb
conda env create -f environment.yml
conda activate LLM-jb
```

**2. Prepare the pretrained LLM weights**

Our experiments were mainly performed on LLaMA-2. Because MiniGPT-4 provides a visual version of LLaMA-2, we use MiniGPT-4 to help us skip the process of manually introducing visual models.

The model weights and tokenizer of LLaMA-2 can be downloaded at [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) 

Then, set the LLM path 
  [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.

**3. Prepare the pretrained model checkpoints**

Download the pretrained model checkpoints of MiniGPT-4 (LLaMA-2 Chat 7B) at  [Download](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing).

Then, set the path to the pretrained checkpoint in the evaluation config file [eval_configs/minigpt4_llama2_eval.yaml](eval_configs/minigpt4_llama2_eval.yaml#L10).   

**4. Prepare LLama-Guard-2**

Download the LLama-Guard-2 at  [Download](https://huggingface.co/meta-llama/Meta-Llama-Guard-2-8B).  

Then set the LLama-Guard2 path 
  [here](minigpt4/configs/models/minigpt4_llama2.yaml#L15) at Line 15.


If you want to jailbreak other large language models, you can use the suffix generated on LLaMA-2 to jailbreak in a black box manner, or use the MiniGPT-4 method to introduce vision modules to large language models and jailbreak them.



# Launching Demo Locally
First , run [best_init_img.ipynb](best_init_img.ipynb) to get a good initial img for LLM-jb in the next step.

Then , run the following command to jailbreak LLM.
```
python LLM_jb.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml --gpu-id 0 --class_tag S1 --attack_power 128 
```

## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) This repository is built upon MiniGPT-4!

+ [llm-attacks](https://github.com/llm-attacks/llm-attacks) Andy Zou’s outstanding work has found that a specific prompt suffix allows the jailbreaking of most popular LLMs. Don't forget to check this great open-source work if you don't know it before!

+ [adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch) Torchattacks is a PyTorch library that provides adversarial attacks to generate adversarial examples.

  
