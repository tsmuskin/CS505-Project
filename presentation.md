# Script for poster presentation:

Idea:
Wanted to generate two emojis that don't exist and that users can put in any prompt they'd like. 

### Current work:

**Apple AI**: Apple's emoji generation, just released but the code is not open source so we don't know what they did.

**Finetuning SD**: Finetuning stable diffusion model with a  stylizer to create new images in a specific style. 

### Methodology: 
#### Data:
Used data from a Kaggle which is a complete emoji dataset. Took one of each type of emoji and it's descriptiion and augmented the images and description to use similar words. 

### Training:
We followed the second approach of finetuning a Stable Diffusion model and added an emoji stalyzer to create new emoji images. We started with three pretrained models `UNet2DConditionModel` to de-noise the image, `AutoencoderKL (VAE)` to compress the images into latent space and reconstruct them and `CLIPTextModel` to encode the text prompts into embeddings. We then used our augmented images and descriptions and had the model learn the style of an emoji. We then calculate the content loss between the generated and ground-truth images to ensure the output is consistent with the prompt, and backpropagate the combined loss to optimize the model parameters. 

### Evaluation and Results 
we used Inception score and FID which are common GAN metrics. 

**Inception score**: Mean 1.4, Standard deviation 0.2 
Inception score can range from zero (worst) to infinity (best), but the highest score in literature is on the order of 10.
It takes into account the quality, how good the generated image is, and diversity, how diverse the generated image is. Since generated images produce highly varied images and have high randomness (entorpy).

**FID score**: 7.2 
 Frechet Inception Distance is a metric for evaluating the quality of generated images by evaluating synthetic images based on the statistics of a collection of synthetic images compared to the statistics of a collection of real images from the target domain.

 The range of FID is from zero (best) to infinity (worst). A lower FID indicates better-quality images, as a lower FID score means the generated images are more statistically similar to the real images.

### Future work

**Backgrouond Removal**: The generated images applies a green background to most images we would like to remove that background so that the images are transparent and are like real emojis which don't have backgrounds.

**Different augmentations**: Instead of augmenting the images by stretching and rotating and cutting of different parts on the iamges, we would like to make augmentations differenly by changing the colors, flipping the image, saturating the images ect. We would like to try this because we have noticed from of our images generate errors (like serving a head on a platter) and believe these errors are happening becuase we augmenting the photos to be cut off. 

**Training without augmentations**: We would like to see how images would be generated if we used all of our images from our dataset and didn't do any augmentations.  




