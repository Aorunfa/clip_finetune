import torch
#import clip
from PIL import Image
import time
from model import CLIP
import copy
from typing import Union, List
from dataset import CsvDataset
from torch.utils.data import DataLoader

def _transform(n_px):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    try:
        from torchvision.transforms import InterpolationMode
        BICUBIC = InterpolationMode.BICUBIC
    except ImportError:
        BICUBIC = Image.BICUBIC

    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def build_model_transform(model_hyp):
    model = CLIP(**model_hyp)
    return model, _transform(model.visual.input_resolution)
        
def build_tokenizer():
    from simple_tokenizer import SimpleTokenizer
    _tokenizer = SimpleTokenizer()
    def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
    return tokenize

def collect_fun(batch):
    prev_text = []
    img_tensors, text_tensors = [], []
    for b in batch:
        if b['text'] not in prev_text:
            prev_text.append(b['text'])
            img_tensors.append(b['img_tensor'])
            text_tensors.append(b['text_tensor'])
    img_tensors = torch.stack(img_tensors)
    text_tensors = torch.stack(text_tensors)
    return img_tensors, text_tensors

def bulid_dataloader(args, data_hyp, transform, tokenizer, shuffle=True):
    ds = CsvDataset(

            **data_hyp,
            transforms=transform,
            tokenizer=tokenizer
        )
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collect_fun, shuffle=shuffle)


def build_clip_loss(loss_hyp):
    from loss import ClipLoss
    return ClipLoss(**loss_hyp)

if __name__ == '__main__':
    args = Config()
    model, transform = build_model_transform(args)
    tokenizer = build_tokenizer()
    # res = tokenizer(["a diagram", "a dog", "a cat"])
    trian_dl = bulid_dataloader(args, transform, tokenizer)
    loss = build_clip_loss(args)


    




        
