import torch
import pandas as pd

from torch import tensor 
from transformers import DistilBertTokenizer, BertTokenizer
from transformers.pipelines import TextClassificationPipeline
from captum.attr import IntegratedGradients, LayerIntegratedGradients, TokenReferenceBase

import matplotlib.pyplot as plt

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""
    
    def __init__(self, pipeline: TextClassificationPipeline, device: str, output_path: str, algorithms: list, model: str):
        self.__pipeline = pipeline
        self.__device = device
        self.output_path = output_path
        self.algorithms = algorithms
        self.model = model
    
    def forward_func(self, inputs: tensor, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs,
                       attention_mask=torch.ones_like(inputs))
        return pred[position]
        
    def visualize(self, inputs: list, attributes: list, index: int=0, output: bool=False):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        attr_sum = attributes.sum(-1) 
        
        attr = attr_sum / torch.norm(attr_sum)
        
        a = pd.Series(attr.numpy()[0], 
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().numpy()[0]))
        
        a.plot.barh(figsize=(10,20))
        plt.show()
        if output:
            plt.savefig(f'{self.output_path}/viz-{index}.png', bbox_inches='tight')
                      
    def explain(self, text: str, visualize: bool=False, index: int=0):
        """
            Main entry method. Passes text through series of transformations and through the model. 
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])
        
        if 'lig' in self.algorithms:
            lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, self.model).embeddings)

            attributes, delta = lig.attribute(inputs=inputs,
                                    baselines=baseline,
                                    target = self.__pipeline.model.config.label2id[prediction[0]['label']], 
                                    return_convergence_delta = True)
            print(inputs, attributes, prediction)
            print(inputs.shape, attributes.shape, delta)
            if visualize:
                self.visualize(inputs, attributes, i)

        if 'ig' in self.algorithms:
            print(prediction)
            print(prediction[0]['label'], type(self.__pipeline.model.config.label2id[prediction[0]['label']]))
            ig = IntegratedGradients(self.forward_func, getattr(self.__pipeline.model, self.model).embeddings)
            attributes, delta = ig.attribute(inputs=inputs,
                                    baselines=baseline,
                                    target = torch.tensor(self.__pipeline.model.config.label2id[prediction[0]['label']]),
                                    return_convergence_delta = True)
            print(inputs, attributes, prediction)
            print(inputs.shape, attributes.shape, delta)
        
    def generate_inputs(self, text: str) -> tensor:
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False), device = self.__device).unsqueeze(0)
    
    def generate_baseline(self, sequence_len: int) -> tensor:
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """        
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)
