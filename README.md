# DA-HOI
Official implementation of the ICLR 2026 paper: **"[Zero-shot HOI Detection with MLLM-based Detector-agnostic Interaction Recognition](https://arxiv.org/pdf/2602.15124)"**.

## 🛠 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/sy-xuan/DA-HOI.git
cd DA-HOI
pip install -r requirements.txt
```

## 📂 Data preparation

### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/file/d/1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS/view). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
data
 └─ hico
     |─ annotations
     |   |─ trainval_hico.json
     |   |─ test_hico.json
     |   └─ corre_hico.npy
     :
```

## 🧩 Pre-trained Detector

To guide the training, we utilize the pre-trained **DETR** detector. Please refer to [Gen-VLKT](https://github.com/YueLiao/gen-vlkt) for initial setup.

**Note:** Our model inherently decouples the detector from the interaction recognizer. Because of this detector-agnostic design, you are highly encouraged to plug in and replace DETR with any modern object detector of your choice.

## 🚀 Training
After the preparation, you can start training with the following commands.
### HICO-DET

```
<!-- The SAP Training -->
bash ./SAP/config/train.sh
<!-- The MLLM Training -->
bash train.sh
```

*Tip: You can easily configure the scripts to test different zero-shot settings (e.g., unseen-verb, unseen-object, etc.).*

## 📊 Evaluation

After training, you can evaluate the model on the HICO-DET using:

```bash
bash eval.sh

```

If you wish to test with different modern detectors, you can refer to and modify the following evaluation scripts:

* `qwenvl/eval/evaluate_dino.py`
* `qwenvl/eval/evaluate_gt.py`

## 🏆 Results

### Regular HOI Detection

Performance on the HICO-DET dataset:

| Model | Full (Default) | Rare (Default) | Non-rare (Default) |
| --- | --- | --- | --- |
| **DA-HOI** | **44.58** | **46.17** | **44.08** |

### Zero-shot HOI Detection

Performance across various zero-shot settings:

| Model | Setting Type | Unseen | Seen | Full |
| --- | --- | --- | --- | --- |
| **DA-HOI** | Rare First (RF-UC) | 41.79 | 44.01 | 43.56 |
| **DA-HOI** | Non-rare First (NF-UC) | 43.12 | 39.63 | 40.33 |
| **DA-HOI** | Unseen Object (UO) | **48.67** | 42.58 | **43.60** |
| **DA-HOI** | Unseen Verb (UV) | 36.89 | **43.84** | 42.88 |

## 📝 Citation

If you find this code or our paper useful for your research, please consider citing:

```bibtex
@inproceedings{xuan2026zeroshot,
  title={Zero-shot {HOI} Detection with {MLLM}-based Detector-agnostic Interaction Recognition},
  author={Shiyu Xuan and Dongkai Wang and Zechao Li and Jinhui Tang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=oHWg8cs5No}
}

```

## 🙏 Acknowledgments

Some portions of our code are built upon the following incredible open-source works. We thank the authors for their contributions to the community!
 * [Qwen-VL](https://github.com/QwenLM/Qwen3-VL)
 * [GEN-VLKT](https://github.com/YueLiao/gen-vlkt)
 * [DETR](https://github.com/facebookresearch/detr)
