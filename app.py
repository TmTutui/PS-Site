from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/score', methods=['GET', 'POST'])
def score():
  default_value = "1"

  canal = int(request.form.get('canal_de_vendas_ord', default_value ))
  subcanal = int(request.form.get('subcanal_de_vendas_ord', default_value ))
  segmento = int(request.form.get('segmento_ord', default_value ))
  regiao = int(request.form.get('regiao_ord', default_value ))
  cidade = int(request.form.get('cidade_ord', default_value ))
  bairro = int(request.form.get('bairro_ord', default_value ))
  uf = int(request.form.get('uf_ord', default_value ))
  mcc = int(request.form.get('mcc_ord', default_value ))
  mei = int(request.form.get('mei', default_value ))
  tpv = int(request.form.get('canal_de_vendas_ord', "5763" ))

  score = .8
  limite = 10000
  rotativo = "5%"
  parcelas = "4%"

  if(mei == 1):
    # se for MEI
    aprovado = True if score > .75 else False
  else:
    # se não for MEI
    aprovado = True if score > .85 else False

  return render_template('score.html', score = score, limite = limite, rotativo = rotativo, parcelas = parcelas, aprovado = aprovado)


if __name__ == '__main__':
  app.run(debug=True)