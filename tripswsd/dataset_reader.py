from typing import Iterator, List, Dict, Union
import json

from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


from allennlp.data.tokenizers import Token, Tokenizer

@DatasetReader.register("wsd")
class WSDReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"token": SingleIdTokenIndexer()}
        self.count = 0

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                # What does Token do here?
                yield self.text_to_instance([Token(word) for word in sentence], tags)




@DatasetReader.register("jswsd")
class JsonWSDReader(WSDReader):
    def __init__(self, 
            tokenizer: Tokenizer = None ,
            label_field : str = "lftype",
            token_indexers: Dict[str, TokenIndexer] = None, 
            entry_transform: Dict[str, str] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer 
        self.label_tokenizer = lambda sentence: ["_"] + [x.get(label_field, "_") for x in sentence] + ["_"]
        self.token_indexers = token_indexers or {"token": SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for sentence in json.load(f):
                s, t = self.tokenizer.tokenize(sentence), self.label_tokenizer(sentence)
                yield self.text_to_instance(s, t)

