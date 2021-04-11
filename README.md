# Attention_Backend_for_ASV
Attention Backend for Aotumatic Speaker Verification with Multiple Enrollment Utterances

It contains the official implementation of the paper [Attention Back-end for Automatic Speaker Verification with Multiple Enrollment Utterances](https://arxiv.org/abs/2104.01541)

## requirements

1. Kaldi. And set kaldi path in `path.sh` and `run.sh` according to the instruction in these files.
2. Pytorch >= 1.0.0
3. Numpy

## data

You can download the data from this [link](Link:https://dubox.com/s/1m8n3h7zP4lr1UA64aFYPfQ)

 Password: e2de.

It contains x-vectors extracted by the script of cnceleb example in Kaldi (train_xv, enroll_xv, eval_xv)

## results

![image-20210411134608804](https://i.loli.net/2021/04/11/hmEyBCFvSIbJ4Ro.png)

**Note**: This code only for X-Vectors extracted from TDNN.

## to do

- [ ] TD-ASV (RedDots)

- [ ] Change score method from cosine similarity to PLDA-like score

- [ ] Breakdown results per domain (genre) in CN-Celeb

## reference

[siamese-triplet](https://github.com/adambielski/siamese-triplet)

