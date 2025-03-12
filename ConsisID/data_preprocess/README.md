# <u>Data Preprocessing Pipeline</u> by *ConsisID* (Support Multi-ID Annotation)
This repo describes how to process your own data like [ConsisID-Preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data) datasets in the [ConsisID](https://arxiv.org/abs/2411.17440) paper.

## 🎉 Overview

<div align=center>
<img src="https://github.com/user-attachments/assets/f9a1dc9b-befb-44ee-8a4d-cef89e717dde">
</div>

## ⚙️ Requirements and Installation

We recommend the requirements as follows.

### Environment

```bash
git clone --depth=1 https://github.com/PKU-YuanGroup/ConsisID.git
cd ConsisID/data_preprocess
conda create -n consisid_data python=3.11.0
conda activate consisid_data
pip install -r requirements.txt
```

### Download Weight

The weights will be automatically downloaded, or you can download it with the following commands.

```bash
cd util
python download_weights_data.py
```

Once ready, the weights will be organized in this format:

```
📦 ConsisiID/
├── 📂 ckpts/
│   ├── 📂 data_process/
│       ├── 📂 Qwen2-VL-7B-Instruct
│       ├── 📄 step1_yolov8_face.pt
│       ├── 📄 step1_yolov8_head.pt
│       ├── 📄 yolov8l-worldv2.pt
│       ├── 📄 yolov8l-pose.pt
│       ├── 📄 sam2.1_hiera_large.pt
```

## 🗝️ Usage

### Step 0 - Split Transition (Optional)

To ensure data purity, we first use [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/tree/main) to split the video into multiple single-scene clips. (You can skip this step and directly use the multi-scene clips for training)

```bash
python step0_split_transition.py
```

### Step 1 - Multi-view Face Filtering

The purity of internet-sourced data is typically low, as full videos often include only brief segments featuring facial content. To address this, we use [YOLO](https://github.com/ultralytics/ultralytics) to obtain <u>bounding boxes</u> and <u>poses</u> for "*face*", "*head*", and "*person*", and then split the video based on this information.

```bash
python step1_get_bbox_pose.py
python step2_split_bbox_pose.py
```

### Step 2 - ID Verification

A video may include multiple individuals, necessitating the assignment of a unique identifier to each person for subsequent training. We utilize the previously obtained frame-by-frame *bbox* to compute a unique identifier for each individual.

```bash
python step3_get_refine_track.py
```

### Step 3 - Segmentation

To facilitate the application of *Dynamic Mask Loss*, we first input the highest-confidence *bbox* for each category obtained in the previous step into [SAM-2](https://github.com/facebookresearch/sam2/tree/main) to generate the corresponding masks for each person's "face," "head," and "person."

```bash
python step4_get_mask.py
```

### Step 4 - Captioning

We use [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) to generate captions and incorporate the *meta-info* of the video into the annotation.

```bash
python step5_get_caption.py
python step6_get_video_info.py
```

### Step 5 - Organize Data

If you want to train your own identity-preserving text-to-video generation model, you need to arrange all the dataset in this [format](https://github.com/PKU-YuanGroup/ConsisID/tree/main/asserts/demo_train_data/dataname):

```
📦 datasets/
├── 📂 captions/
│   ├── 📄 dataname_1.json
│   ├── 📄 dataname_2.json
├── 📂 dataname_1/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── 📂 dataname_2/
│   ├── 📂 refine_bbox_jsons/
│   ├── 📂 track_masks_data/
│   ├── 📂 videos/
├── ...
├── 📄 total_train_data.txt
```

### Step 6 - Get Summary Data Txt

Do not add spaces before and after the commas, otherwise you will get an error. An example can be found [here](https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/demo_train_data/total_train_data.txt).
```
#video_path,caption_path,dataset_name
dataname_videos,dataname_captions.json,dataname
dataname_1_videos,dataname_1_captions.json,dataname_1
...
dataname_n_videos,dataname_n_captions.json,dataname_n
```

## 🔒 Limitation

- Although the models used in the current data pipeline are lightweight (e.g., [YOLO](https://github.com/ultralytics/ultralytics), [SAM-2](https://github.com/facebookresearch/sam2/tree/main)), the GPU utilization is relatively low, resulting in longer processing times. We will continue to update the code in the future.
- The data preprocess code support multi-face annotation, but the training code only support single-face currently.
