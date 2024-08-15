import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, pretrained_dir, image_encoder='base', feature_dim=768):
        """
        image_encoder: base / large
        """
        super(ImageEncoder, self).__init__()

        assert image_encoder in ['vit-base', 'vit-large']

        tokenizer = ViTFeatureExtractor
        model = ViTModel

        if image_encoder == 'vit-base':
            config = f'{pretrained_dir}/vit-base/config.json'
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit-base/')
            self.model = model.from_pretrained(pretrained_dir+'/vit-base/', config=config, add_pooling_layer=False)
            self.model.config.hidden_size = feature_dim  
        else:
            config = f'{pretrained_dir}/vit-large/config.json'
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit-large/')
            self.model = model.from_pretrained(pretrained_dir+'/vit-large/', config=config, add_pooling_layer=False)
            self.model.config.hidden_size = feature_dim  

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, pixel_values):
        last_hidden_states = self.model(pixel_values=pixel_values).last_hidden_state
        return last_hidden_states


if __name__ == "__main__":
    vit_normal = ImageEncoder()
