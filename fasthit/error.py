class TransformerImportError(ImportError):
    def __str__(self):
        return ("transformers which contained ProtTrans not installed. "
                "Source code are available at "
                "https://github.com/huggingface/transformers")


class EsmImportError(ImportError):
    def __str__(self):
        return ("fair-esm not installed. "
                "Source code are available at "
                "https://github.com/facebookresearch/esm")


class TapeImportError(ImportError):
    def __str__(self):
        return ("tape-proteins not installed. "
                "Source code are available at "
                "submodule ./fasthit/encoders/tape")
