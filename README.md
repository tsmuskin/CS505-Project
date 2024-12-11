# Text-to-Emoji: Custom Emoji Generation
Temima Muskin, Sujayanand Kingsly


## Setup the environment
First, install prerequisites with:

    conda env create -f environment.yml
    conda activate sd
  
Then, set up the configuration for your machine with:

    accelerate config

## Train a model
To train the model: We recommend using 2 GPU 8.9's to train the model and ensure `accelerate config` is set up to use multi-GPU.

    accelerate launch train.py --config='configs/train_emoji.json'

Instead of training the model you can download [this saved model](https://drive.google.com/file/d/1i1C_XDuIPZy_uVCn4DKGtPhu3elucWpu/view?usp=drive_link) and put it under models/emoji
## Evaluate a model
To Evalute/Obtain outputs: Make sure to recall `accelerate config` to use multi-CPU, but still use the GPU's from the SCC. 

    accelerate launch eval.py --config='configs/eval_emoji.json'

To change the prompt go to `configs/eval_emoji.json` and update the `prompt`. 

## Example Output
Prompt: Playing Soccer
![example output](/output/emoji2/1/200/image_2.png)

