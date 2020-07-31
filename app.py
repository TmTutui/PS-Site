from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/score/', methods=['GET', 'POST'])
def score():
  data = request.form['input_name']

  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)