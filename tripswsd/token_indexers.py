from typing import List, Dict

from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

@Tokenizer.register("dicts")
class DictTokenizer(Tokenizer):
    """
    a ``DictTokenizer`` takes data in the form of a list of dictionaries and transforms
    each dictionary into a single token


    Parameters
    ----------

    text_id
    lex_id
    pos_id
    meta_id

    start_token
    end_token
    """

    def __init__(self,
            meta_ids: Dict[str, str] = None,
            start_token="<start>",
            end_token="<end>"):
        if not meta_ids:
            meta_ids = {"text": "lex"}
        self._meta_ids = meta_ids

        self._start_token = Token(
            **{i: start_token for i in self._meta_ids}
            )
        self._end_token = Token(
            **{i: end_token for i in self._meta_ids}
            )

    @overrides
    def tokenize(self, sentence: List[Dict[str, str]]) -> List:
        """
        convert a list of dicts into a list of token elements
        """
        sentence = [
                Token(
                    **{i: w.get(j, "<unk>") 
                        for i, j in self._meta_ids.items()})
                for k, w in enumerate(sentence)
                ]
        sentence.insert(0, self._start_token)
        sentence.append(self._end_token)
        return sentence

    @overrides 
    def batch_tokenize(self, sentences: List[List[Dict[str, str]]]) -> List[List]:
        return sum([self.tokenize(s) for s in sentences], [])

