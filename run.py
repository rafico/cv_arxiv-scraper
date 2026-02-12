from threading import Timer
import webbrowser

from app import create_app

APP_URL = "http://127.0.0.1:5000"


app = create_app()


def _open_browser():
    webbrowser.open(APP_URL)


if __name__ == "__main__":
    Timer(1.0, _open_browser).start()
    app.run(debug=True)
