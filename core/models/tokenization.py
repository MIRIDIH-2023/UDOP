from transformers import T5Tokenizer, PreTrainedTokenizer
import re
import sentencepiece as spm

# The special tokens of T5Tokenizer is hard-coded with <extra_id_{}>
# Created another class UDOPTokenizer extending it to add special visual tokens like <loc_{}>, etc.

class UdopTokenizer(T5Tokenizer):

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=100,
        loc_extra_ids=501,
        other_extra_ids=200,
        additional_special_tokens=[],
        sp_model_kwargs=None,
        **kwargs
    ):
        # Add extra_ids to the special token list
        if extra_ids > 0 and not "<extra_id_0>" in additional_special_tokens:
            additional_special_tokens = ["<extra_id_{}>".format(i) for i in range(extra_ids)]
            additional_special_tokens.extend(["<extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_l_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["<extra_t_id_{}>".format(i) for i in range(extra_ids)])
            additional_special_tokens.extend(["</extra_t_id_{}>".format(i) for i in range(extra_ids)])

        if loc_extra_ids > 0 and not "<loc_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<loc_{}>".format(i) for i in range(loc_extra_ids)])

        if other_extra_ids > 0 and not "<other_0>" in additional_special_tokens:
            additional_special_tokens.extend(["<other_{}>".format(i) for i in range(other_extra_ids)])
        print(additional_special_tokens)
        PreTrainedTokenizer.__init__(
            self,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.vocab_file = vocab_file
        self._extra_ids = extra_ids
        self._loc_extra_ids = loc_extra_ids
        self._other_extra_ids = other_extra_ids

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)
        
    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids + self._other_extra_ids

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(
            i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4
        elif token.startswith("<extra_l_id_"):
            match = re.match(r"<extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3
        elif token.startswith("</extra_l_id_"):
            match = re.match(r"</extra_l_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2
        elif token.startswith("<extra_t_id_"):
            match = re.match(r"<extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids - self._extra_ids
        elif token.startswith("</extra_t_id_"):
            match = re.match(r"</extra_t_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids - self._loc_extra_ids
        elif token.startswith("<loc_"):
            match = re.match(r"<loc_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1 - self._other_extra_ids
        elif token.startswith("<other_"):
            match = re.match(r"<other_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            
            if index > self.sp_model.get_piece_size() + self._extra_ids * 5 + self._loc_extra_ids - 1:
                index_loc = self.vocab_size - 1 - index
                token = f"<other_{index_loc}>"           
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 5 - 1:
                index_loc = self.vocab_size - self._other_extra_ids - 1 - index
                token = f"<loc_{index_loc}>"   
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 4 - 1:
                token = "</extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 3 - 1:
                token = "<extra_t_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids * 2 - 1:
                token = "</extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 2 - 1 - index)
            elif index > self.sp_model.get_piece_size() + self._extra_ids - 1:
                token = "<extra_l_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 3 - 1 - index)
            elif index > self.sp_model.get_piece_size() - 1:
                token = "<extra_id_{}>".format(self.vocab_size - self._other_extra_ids - self._loc_extra_ids - self._extra_ids * 4 - 1 - index)
            else:
                raise
        return token

