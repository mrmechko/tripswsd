from typing import Iterator, List, Dict

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy

CUDA_DEVICE=0

def full_mask(sentence, labels, skip=0):
    mask = get_text_field_mask(sentence).long()
    if labels is not None:
        mask = torch.where(
                labels != skip, 
                mask, 
                torch.zeros(labels.shape).long().cuda(CUDA_DEVICE)
            )
    return mask.cuda(CUDA_DEVICE)

@Model.register("BaseLSTM")
class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.dbg_ctr = 0
        self.word_embeddings = word_embeddings
        self.encoder = encoder.cuda(CUDA_DEVICE)
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels')).cuda(CUDA_DEVICE)
        self.accuracy = CategoricalAccuracy()
        self.blank_index = vocab.get_token_index("_")

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        self.dbg_ctr += 1
        mask = get_text_field_mask(sentence).cuda(CUDA_DEVICE)
        embeddings = self.word_embeddings(sentence).cuda(CUDA_DEVICE)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        mask = full_mask(sentence, labels)
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


