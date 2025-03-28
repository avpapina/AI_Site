from vosk import Model, KaldiRecognizer, SetLogLevel 


def create_recognizers() -> dict:
    rus_model = Model(lang="ru")
    en_model = Model(lang="en-us")
    rec_ru = KaldiRecognizer(rus_model, 16000) # 16000 is sample rate
    rec_ru.SetWords(True)
    rec_ru.SetPartialWords(True)
    rec_en = KaldiRecognizer(en_model, 16000) # 16000 is sample rate
    rec_en.SetWords(True)
    rec_en.SetPartialWords(True)
    recs = {
        'ru': rec_ru,
        'en': rec_en
    }
    return recs


def stt(rec, sound_bytes) -> str:
    print('stting')
    if len(sound_bytes) == 0:
        return 'failed to recognize'
    if rec.AcceptWaveform(sound_bytes):
        print(rec.Result())
    else:
        print(rec.PartialResult())

def tts(text):
    pass

