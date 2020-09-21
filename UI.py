import kivy
from kivy.app import App
from kivy.uix.label import Label

class MyApp(App):
    def build(selfself):
        return Label(text="Cool it works!")

if __name__=="__main__":
    MyApp().run()