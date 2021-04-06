from arch.Models import Transformer, MultiEncTransformer, MultiDecTransformer, ModelingTransformer

AVAIABLE_MODELS = dict(
    standard=Transformer,
    multi_encoder=MultiEncTransformer,
    multi_decoder=MultiDecTransformer,
    modeling=ModelingTransformer,
)
