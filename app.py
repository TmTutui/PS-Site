from flask import Flask, render_template, request

from ML.FINAL_DA_WEB import ml_model
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/score', methods=['GET'])
def score():
  default_value = "1"

  canal = int(request.args.get('canal_de_vendas_ord', default_value ))
  subcanal = int(request.args.get('subcanal_de_vendas_ord', default_value ))
  segmento = int(request.args.get('segmento_ord', default_value ))
  regiao = int(request.args.get('regiao_ord', default_value ))
  cidade = int(request.args.get('cidade_ord', default_value ))
  bairro = int(request.args.get('bairro_ord', default_value ))
  uf = int(request.args.get('uf_ord', default_value ))
  mcc = int(request.args.get('mcc_ord', default_value ))
  mei = int(request.args.get('mei', default_value ))
  tpv = int(request.args.get('tpv', "5763" ))

  inputs = [tpv,segmento,mcc,bairro,cidade,regiao,uf,subcanal]

  print(inputs)

  limite, parcelas, score = ml_model(inputs)

  score = score.round(2)
  print(score)
  limite = limite.round(0)
  rotativo = str((1.5*parcelas).round(2)) + "%"
  parcelas = str(parcelas.round(2)) + "%"

  if(mei == 1):
    # se for MEI
    aprovado = True if score > .75 else False
  else:
    # se não for MEI
    aprovado = True if score > .85 else False

  return render_template('score.html', score = score, limite = limite, rotativo = rotativo, parcelas = parcelas, aprovado = aprovado)


if __name__ == '__main__':
  app.run(debug=True)