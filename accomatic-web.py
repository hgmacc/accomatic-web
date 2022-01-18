from flask import Flask, render_template, url_for


app = Flask(__name__)  # Special python variable to pass in name
app.config['SECRET KEY'] = "secret123"


@app.route("/")  # '/' for the root page of our website
@app.route("/index")
def index():
    return render_template('index.html')


@app.route("/contact")  # '/' for the root page of our website
def contact():
    return render_template('contact.html')


@app.route("/walkthrough")  # '/' for the root page of our website
def walkthrough():
    return render_template('walkthrough.html')


if __name__ == "__main__":
    app.run(debug=True)

