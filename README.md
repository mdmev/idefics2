# Finetuning Chatty Model from idefics2

This repository was used to perform finetuning on the Chatty model from idefics2. I conducted a full finetuning using the split_image flag. However, when saving the model's checkpoint, I encountered a specific issue during inference with the `eval_ocr.py` script, particularly at line 43 when calling `generate`. For some reason, this function is called twice internally, resulting in a shape mismatch of the tensors and preventing the model from making predictions.

The training was based on images and the corresponding text to train an OCR capable of extracting text from a given image without an additional prompt.
