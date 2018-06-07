# t-SNE
This code can do visuarization what recognition result distribution of Discriminator's hidden layer with t-SNE.
Discriminator learned with Generative Adversarial Networks (GANs).  The GANs's Discriminator require that facial attribute recognition branch.

```sh
python3 t-SNE.py
```

# What is doing so
First, input the face images in Discriminator. Note, Dsicriminator use pre-trained model.
Then, extracted feature maps from hidden layer of Discriminator. 
Next, transform dimension of feature map from n-dim to 1-dim. Then, it stack vertial direction.
Stacked item reduce dimention by t-SNE. Paste a image coresponding the distribution. 
Distinguish facial attribute by new RGB image.


# Reference
https://github.com/danielfrg/tsne
