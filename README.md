# MIRA-Bench: Mirror-Induced Reasoning Ambiguity Benchmark

> Evaluating perception, spatial reasoning, and action decisions in reflective scenes with vision-language models.

Hao Yin, Tianchen Guo, Heming Du, Yan Ke, [Yanbin Liu](https://csyanbin.github.io/), and [Xin Yu](https://xinyu-au.github.io/).

**[Dataset on Kaggle](https://www.kaggle.com/datasets/evayin/eval-mirror-dataset)** · **[Code](https://github.com/EvaYin-cloud/MIRA-Bench)**

## 🌟 Highlights

- **Real scenes with reflections:** **1,011 images** from **525 scenes**, covering **six reflective surface types** in indoor and outdoor environments under daytime and nighttime lighting conditions.
- **Hierarchical reasoning tasks:** **14,140 question–answer pairs** across **eight subtasks** and **three cognitive levels**: reflection-aware perception, spatial reasoning, and scene-level decision-making.
- **Detailed annotations:** Reflective surface masks, instance masks for directly observed and reflected objects, correspondences between objects and their reflections, and spatial relation annotations.
- **Controlled input comparisons:** Evaluation across **RGB, RGB-D, RGB-M, and RGB-DM** settings, using **five protocols** for referring to objects, supports analysis of depth information, mirror-region cues, and object referencing.
- **Performance gaps and depth limitations:** Evaluation of **10 VLMs** reveals a substantial gap from human performance. Adding depth does not consistently improve accuracy and can degrade performance on some tasks.

## 🚀 Quick start

### Environment

Use Python 3.10 or 3.11.

```bash
git clone https://github.com/EvaYin-cloud/MIRA-Bench.git
cd MIRA-Bench
conda create -n mira-bench python=3.11 -y
conda activate mira-bench
pip install -r requirements.txt
```

### Download and prepare the data

1. Download the annotations and self-collected RGB/depth images from [Kaggle](https://www.kaggle.com/datasets/evayin/eval-mirror-dataset). Extract them into `Dataset/`, with `annotations/` and `images/` directly inside it.
2. Obtain [RGBD-Mirror](https://mhaiyang.github.io/CVPR2021_PDNet/index.html) and [DVMD](https://github.com/UpChen/2025_DVMDNet) separately. Place or symlink them under `Dataset/images/RGBD-Mirror/` and `Dataset/images/Published_DVMD/`, preserving the original split and scene structure.
3. Prepare the normalized grayscale depth images referenced by the annotations before preprocessing. RGBD-Mirror uses `depth_normalized/`; DVMD uses `Depth_Norm_Per_Sequence/`. The script below consumes these inputs; it does not normalize raw depth. Self-collected grayscale depth is already included.

Arrange the inputs as follows. Keep the original filenames and DVMD scene names used in the annotations.

```text
Dataset/
├── annotations/
│   ├── questions.json
│   └── objects.json
└── images/
    ├── RGBD-Mirror/
    │   ├── train/
    │   │   ├── image/
    │   │   └── depth_normalized/
    │   └── test/
    │       ├── image/
    │       └── depth_normalized/
    ├── Published_DVMD/
    │   ├── train/
    │   │   ├── JPEGImages/<scene>/
    │   │   └── Depth_Norm_Per_Sequence/<scene>/
    │   └── test/
    │       ├── JPEGImages/<scene>/
    │       └── Depth_Norm_Per_Sequence/<scene>/
    └── RGBD_Ben_Self_collected/
        ├── rgb/
        └── depth/
```

Generate the evaluation inputs:

```bash
python scripts/data_process.py --data-root Dataset
```

This updates the annotation paths and adds the following directories alongside `images/`:

```text
Dataset/
├── depth_colorful/       # Colorized depth
├── mirror_marked/
│   ├── rgb/              # RGB with mirror overlays
│   └── depth/            # Depth with mirror overlays
├── _cache/              # Object-level visual prompts
└── _cache_mm/           # Visual prompts with mirror overlays
```

### Set up VLMs

#### Closed-source (API) VLMs

Set the API key for your provider: `OPENAI_API_KEY` for GPT or `GOOGLE_API_KEY` for Gemini.

```bash
export OPENAI_API_KEY="your_api_key"
```

```bash
export GOOGLE_API_KEY="your_api_key"
```

#### Open-source VLMs

Store model weights in separate subdirectories under `models/` at the repository root, using the default paths in the server scripts. To use weights stored elsewhere, set `MODEL_PATH` when starting the server.

Download the weights and start the corresponding server linked below. For Qwen3.5-9B, create a serving environment and download the model:

```bash
conda create -n mira-vllm python=3.11 -y
conda activate mira-vllm
pip install "vllm==0.19.0"
hf download Qwen/Qwen3.5-9B --local-dir models/Qwen3.5-9B
```

Start the server:

```bash
CUDA_VISIBLE_DEVICES=0,1 TENSOR_PARALLEL_SIZE=2 HOST=127.0.0.1 \
  bash scripts/serve/Serve_qwen3_vllm.sh
```

`CUDA_VISIBLE_DEVICES` selects the GPUs; `TENSOR_PARALLEL_SIZE` sets how many GPUs share the model. `MODEL_PATH` selects the local weight directory, and `PORT` sets the service port. Refer to the [vLLM installation guide](https://docs.vllm.ai/en/v0.19.0/getting_started/installation/gpu/) for CUDA requirements.

Keep the server running during evaluation.

### Run evaluation

| Parameter | Controls | Choices / default |
| --- | --- | --- |
| `--model` | Model preset | See the model table below. |
| `--mode` | Input setting | `rgb`, `rgbd`, `rgbm`, `rgbdm`; default: `rgbdm`. `d` adds depth; `m` overlays mirror-region masks on the input images. |
| `--data-root` | Prepared dataset directory | `Dataset` |
| `--qa-json` | Question annotation file | `Dataset/annotations/questions.json`; follows `--data-root`. |
| `--output-dir` | Predictions and reports directory | `Outputs/<model>_<mode>` |
| `--set` | Individual model or runtime setting | For example, `model.api_base=http://localhost:8000/v1`. |

For a closed-source model, run from the evaluation environment after setting its API key:

```bash
python -m src.main --model gpt-5.4 --mode rgbdm
```

For a local server, open another terminal at the repository root and activate the evaluation environment:

```bash
conda activate mira-bench
export OPENAI_API_KEY=EMPTY
python -m src.main --model qwen3 --mode rgbd \
  --set model.api_base=http://localhost:8000/v1
```

Select the model from the table and set `model.api_base` to its server endpoint.

Predictions, metrics, and an Excel summary are saved under `Outputs/`, grouped by model and input setting.

## 🤖 Supported models

| Model | `--model` | Official source | API / local server |
| --- | --- | --- | --- |
| GPT-5.4 | `gpt-5.4` | [API documentation](https://developers.openai.com/api/docs/models/gpt-5.4) | OpenAI API |
| Gemini 2.5 Pro | `gemini-2.5-pro` | [API documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro) | Gemini API |
| Gemini 2.5 Flash | `gemini-2.5-flash` | [API documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash) | Gemini API |
| Gemini 3.8 Flash | `gemini-3.8-flash` | [API documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) | Gemini API |
| Qwen3.5-9B | `qwen3` | [Weights](https://huggingface.co/Qwen/Qwen3.5-9B) | [Start server · 8000](scripts/serve/Serve_qwen3_vllm.sh) |
| Qwen3-VL-8B | `qwen3-vl-8b` | [Weights](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Start server · 8002](scripts/serve/Serve_qwen3_vl_8b_vllm.sh) |
| Qwen2.5-VL-7B | `qwen2.5-vl-7b` | [Weights](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | [Start server · 8003](scripts/serve/Serve_qwen2_5_vl_7b_vllm.sh) |
| LLaVA-OV-7B | `llava-ov-7b-hf` | [Weights (HF format)](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | [Start server · 8004](scripts/serve/Serve_llava_ov_7b_hf_vllm.sh) |
| InternVL3-8B | `internvl3` | [Weights](https://huggingface.co/OpenGVLab/InternVL3-8B) | [Start server · 8001](scripts/serve/Serve_internvl3_vllm.sh) |
| DeepSeek-VL2-Tiny | `deepseek-vl2-tiny-server` | [Weights](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) | [Start server · 8005](scripts/serve/Serve_deepseek_vl2_tiny_vllm.sh) |

## 📜 Citation

To be updated.

## 📄 License

- **Code:** licensed under the [Apache License 2.0](LICENSE).
- **Self-collected data and original annotations:** [CC BY-NC 4.0](LICENSE-DATA) for non-commercial use.

### Licenses for existing assets

[RGBD-Mirror](https://mhaiyang.github.io/CVPR2021_PDNet/index.html) and [DVMD-D](https://github.com/UpChen/2025_DVMDNet) are publicly available and remain subject to the applicable terms of use of their original sources.

## 🙏 Acknowledgements

We thank the creators of RGBD-Mirror and DVMD, and the open-source vision-language model community.

### Acknowledgements and disclosure of funding

This research is funded in part by ARC-Discovery grants (DP220100800, DE230100477, and FT260100691 to XY). Hao Yin acknowledges STSE support from The University of Queensland. Yanbin Liu is supported by the Australian Government through an ARC Discovery Early Career Researcher Award (DECRA), project number DE260100509. We also acknowledge support from the NVIDIA Academic Research Program. We thank the anonymous reviewers and area chairs for their constructive suggestions.
