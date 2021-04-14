# Attention_Backend_for_ASV
Attention Backend for Aotumatic Speaker Verification with Multiple Enrollment Utterances

It contains the official implementation of the paper [Attention Back-end for Automatic Speaker Verification with Multiple Enrollment Utterances](https://arxiv.org/abs/2104.01541)

## Authors

- Chang Zeng
- Xin Wang
- Erica Cooper
- [Junichi Yamagishi](https://nii-yamagishilab.github.io/)

## Requirements

1. Kaldi. And set kaldi path in `path.sh` and `run.sh` according to the instruction in these files.
2. Pytorch >= 1.0.0
3. Numpy

## Data

You can download the data from this [link](https://dubox.com/s/1m8n3h7zP4lr1UA64aFYPfQ)

Password: e2de.

It contains x-vectors extracted by the script of cnceleb example in Kaldi (train_xv, enroll_xv, eval_xv)

- x-vectors are extracted from the [CN-Celeb database](http://www.openslr.org/82/) owned by the Tsinghua University.

- x-vectors are distributed under the CC BY-SA license.

## Usage

1. Install all requirements.

    - Pytorch and Numpy for training

    - Kaldi for evaluation (EER and minDCF computation)

2. Create a directory for exp.

```
mkdir attention_backend_exp
```

3. Download the data (x-vectors) from above link and put is in `attention_backend_exp` directory.

4. Clone this repo 

```
cd attention_backend_exp
git clone https://github.com/nii-yamagishilab/Attention_Backend_for_ASV.git
```

5. Run the following command for training and scoring

```
python3 train.py
```

This script will generate exp log directory in `exp` for each training process. The directory is named by the time when you run this code. You can also resume training process by set the model path in exp log directory in `config.yaml` file.

6. EER and minDCF compuation

```
./run.sh exp_log_directory
```

## Results

![image-20210411134608804](https://i.loli.net/2021/04/11/hmEyBCFvSIbJ4Ro.png)

**Note**: This code only for X-Vectors extracted from TDNN.

## To do

- [ ] TD-ASV (RedDots)

- [ ] Change score method from cosine similarity to PLDA-like score

- [ ] Breakdown results per domain (genre) in CN-Celeb

## Acknowlegment

This study is supported by JST CREST Grants (JPMJCR18A6, JPMJCR20D3), MEXT KAKENHI Grants (16H06302, 18H04120, 18H04112, 18KT0051), Japan, and Google AI for Japan program.



This project used some code snippets from following repo:

- `dataset.py` implemented `BalancedBatchSampler` class from [siamese-triplet](https://github.com/adambielski/siamese-triplet). The corresponding license file is `siamese-triplet_license`.
- `attention.py` implemented `AttentionAlphaComponent` class from [asv-subtools](https://github.com/Snowdar/asv-subtools). The corresponding license file is `asv-subtools_license`.

## License

BSD 3-Clause License

Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.