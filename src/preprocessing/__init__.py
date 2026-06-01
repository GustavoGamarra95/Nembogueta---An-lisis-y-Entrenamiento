def process_letter_data(*args, **kwargs):
    from . import letter_preprocessor
    return letter_preprocessor.process_data(*args, **kwargs)


def process_word_data(*args, **kwargs):
    from . import word_processor
    return word_processor.process_data(*args, **kwargs)


def process_phrase_data(*args, **kwargs):
    from . import phrase_processor
    return phrase_processor.process_data(*args, **kwargs)
