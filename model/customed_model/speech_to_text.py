import speech_recognition as sr


def speech_to_text(self):
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        print("listening...")
        audio = recognizer.listen(mic)
    try:
        self.text = recognizer.recognize_google(audio)
        print("me --> ", self.text)
    except:
        print("me --> ERROR")
