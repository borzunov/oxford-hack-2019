from flask import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/layout')
def layout():
    return render_template("layout.html")


@app.route('/aboutus')
def about():
    return render_template("about.html")


@app.route('/upload')    # ('/') - WORK
def upload():
    return render_template("upload.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return render_template("success.html", name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)