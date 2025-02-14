import torch
import whisper
import torch.nn as nn
import pickle
from ModelInterfaces import IASRModel
from AIModels import NeuralASR 

def getASRModel(language: str) -> whisper.Whisper:
    if language == 'de':

        model = whisper.load_model("medium", )
        supported_languages ='de'
        if language not in supported_languages:
            raise ValueError(f"Language {language} is not supported. Supported languages are: {supported_languages}")

    elif language == 'en':
        model = whisper.load_model("medium", )
        supported_languages = 'en'
        if language not in supported_languages:
            raise ValueError(f"Language {language} is not supported. Supported languages are: {supported_languages}")

    elif language == 'fr':
        model = whisper.load_model("medium", )
        supported_languages ='fr'
        if language not in supported_languages:
            raise ValueError(f"Language {language} is not supported. Supported languages are: {supported_languages}")
    return (model)



def getTTSModel(language: str) -> nn.Module:

    if language == 'de':

        speaker = 'thorsten_v2'  # 16 kHz
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)

    elif language == 'en':
        speaker = 'lj_16khz'  # 16 kHz
        model = torch.hub.load(repo_or_dir='snakers4/silero-models',
                               model='silero_tts',
                               language=language,
                               speaker=speaker)
    else:
        raise ValueError('Language not implemented')

    return model


def getTranslationModel(language: str) -> nn.Module:
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    if language == 'de':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        # Cache models to avoid Hugging face processing
        with open('translation_model_de.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_de.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
    else:
        raise ValueError('Language not implemented')

    return model, tokenizer
