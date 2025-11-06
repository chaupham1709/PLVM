# [CVPRW25] Implementation of PLVM: Personalize Large Vision-Language Model

Paper: https://arxiv.org/pdf/2412.17610

- [ ] Training code
- [x] Inference code
- [x] PLVM dataset

![PLVM](figures/intro.jpg)

**Abstract**: The personalization model has gained significant attention in image generation yet remains underexplored for large vision-language models (LVLMs).
Beyond generic ones, with personalization, LVLMs handle interactive dialogues using referential concepts (e.g, "Mike and Susan are talking.") instead of the generic form (e.g, a boy and a girl are talking.), making the conversation more customizable and referentially friendly. In addition, PLVM is equipped to continuously add new concepts during a dialogue without incurring additional costs, which significantly enhances the practicality. PLVM proposes Aligner, a pre-trained visual encoder to align referential concepts with the queried images. During the dialogues, it extracts features of reference images with these corresponding concepts and recognizes them in the queried image, enabling personalization. We note that the computational cost and parameter count of the Aligner are negligible within the entire framework. With comprehensive qualitative and quantitative analyses, we reveal the effectiveness and superiority of PLVM.

## Setup for PLVM

- Checkpoint path download: Please download the checkpoint path of the Aligner at this [link](https://drive.google.com/file/d/1_zdWlCXPem_RidqRW1Wt6yNe758Vovdv/view?usp=sharing).

- PLVM is install on top of [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install). Follow the instruction to install LLaVA for the LLaVA-Vicuna version.

- You can download the [YoLLaVA dataset](https://github.com/WisconsinAIVision/YoLLaVA) for testing PLVM with the human faces from YoLLaVA. We also included the photos from YoLLaVA dataset for human category in this repository, including the reference image for testing. Check out the ``yollava_dataset`` folder.

- Our evaluation dataset can be found at ``validation_recognition`` folder, we also include the json file for the VQA task. Please refer to ``non_query_ques.json`` and ``query_ques.json``.

## Running PLVM

- Our LVLM personalization can be run in ``run_script.sh``.

```python
python encoder_based_method.py \
    --model-path="/home/csgrad/haichaup/Code/LLaVA/llava-v1.5-7b" \
    --load-8bit \
    --output_dir="exp/"\
    --logging_dir="exp/"\
    --gradient_accumulation_steps=1 \
    --mixed_precision="fp16" \
    --num_train_steps=10000 \
    --img_dir="data/CelebAMask-HQ/CelebA-HQ-img" \
    --importance_weight=3.0 \
    --infer_ref_img="yollava_dataset/viruss/ref_img.png" \
    --infer_query_img="yollava_dataset/viruss/1.png" \
    --checkpoint_path="checkpoint.ckpt" \
    --question="Is <sks> in this photo?" \
    --task="infer"
```

If you find our work interesting, please considering cite:

## Citation
If you find our work interesting, please consider citing:
```
@inproceedings{pham2025plvm,
  title={PLVM: A tuning-free approach for Personalized Large Vision-Language Model},
  author={Pham, Chau and Phan, Hoang and Doermann, David and Tian, Yunjie},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3632--3641},
  year={2025}
}
```