import torch
import torch.nn as nn
from transformers import BertTokenizer, VisualBertModel, DeiTFeatureExtractor
from PIL import Image

alphabet = 'abcdefghijklmnopqrstuvwxyz'
data_path = '../data/asl_alphabet_train/asl_alphabet_train'

class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = VisualBertModel.from_pretrained('uclanlp/visualbert-vqa-coco-pre')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')

    def forward(self, token, image):
        # inputs = self.tokenizer(token, return_tensors="pt")
        features = self.feature_extractor([image])
        return features

    def process_letter(self, letter):
        images_outputs = []
        for i in range(1, 3001):
            path = f'{data_path}/{letter.upper()}/{letter.upper()}{i}.jpg'
            print(path)
            image = Image.open(path)
            output = self.forward(letter, image)
            images_outputs.append(output)
        return images_outputs

    def process_alphabet(self):
        for letter in alphabet:
            outputs = self.process_letter(letter)
            print(outputs)
            break


text_to_image = BERTEncoder()
text_to_image.process_alphabet()

