from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import matplotlib
import plotly.express as px
matplotlib.use('Agg')  # Use o
import matplotlib.pyplot as plt
app = Flask(__name__)

cols_float=['idade_diagnostico', 'INR', 'AFP', 'Hemoglobina', 'plaquetas', 'gama_glutamil_transferase',
            'biliburrina_total', 'albumina', 'fosfatase_alcalina', 'maior_diametro_nodulo', 'biliburrina_direta', 'ferro', 'ferritina']
cols_bin = ['sintomas','trombose_veia_porta', 'metastases_hepatica']
dic_nomes = {"idade_diagnostico": "Idade do diagnóstico (Anos)", "INR": "Razão Normalizada Internacional", "AFP":"Alfa-Fetoproteína (ng/mL)", "Hemoglobina": "Hemoglobina (g/dL)",
            "plaquetas": "Plaquetas (G/L)", "gama_glutamil_transferase":"Gama Glutamil Transferase (U/L", "biliburrina_total" : "Bilirrubina Total (mg/dL)", 
            "albumina": "Albumina (mg/dL)", "fosfatase_alcalina" : "Fosfatase Alcalina (U/L)", "maior_diametro_nodulo": "Tamanho do maior nódulo (cm)", "biliburrina_direta" : "Bilirrubina Direta (mg/dL)",
            "ferro" : "Ferro (mcg/dL)", "ferritina" : "Ferritina (ng/mL)", "sintomas" : "sintomas", "trombose_veia_porta" : "Trombose da Veia Porta", "metastases_hepatica": "Metástase Hepática", 
            "grau_ascite_1.0" : "Grau 1 de Ascite", "grau_ascite_2.0" : "Grau 2 de Ascite", "grau_ascite_3.0" : "Grau 3 de Ascite"} 
base_value = 0.38181818181818183


@app.route('/')
def index():
    return render_template('formulario.html',  cols_bin = cols_bin, cols_float = cols_float, dic_nomes = dic_nomes)

@app.route('/result', methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
            rcf = joblib.load('modelo_random_forest.joblib')
            sc = joblib.load('scaler.pkl')
            cols_modelo = rcf.feature_names_in_.tolist()
            df_resp = pd.DataFrame(request.form.items()).T
            df_resp.columns = df_resp.iloc[0]
            df_resp = df_resp.drop(0) 
            df_resp = df_resp.drop(columns = 'confirmacao')
            df_resp[cols_float + ['grau_ascite']] = df_resp[cols_float + ['grau_ascite']].astype(float)
            df_resp = pd.get_dummies(df_resp, columns=['grau_ascite'], dtype=int)
            for col in cols_modelo:
                if col not in df_resp.columns:
                    #como o  formulario nao vai aceitar valores nulos, as flags dummies para valores nulos sera 0
                    df_resp[col] = 0
            df_resp = df_resp[cols_modelo]
            df_resp_t = df_resp.copy().T.reset_index()
            df_resp_data = sc.transform(df_resp)
            df_resp = pd.DataFrame(df_resp_data,columns=cols_modelo)
            probabilidade_final = f"{int(round(rcf.predict_proba(df_resp)[0][1],2) * 100)}%"
            explainer = shap.TreeExplainer(rcf)
            #valores das importancias dos atributos
            shap_values = explainer.shap_values(df_resp)
            df_resp_t.columns = ['col','valor_paciente']
            df_means = pd.read_csv('medias.csv').round(2)

            df_means = pd.merge(df_means,df_resp_t, on = 'col', how= 'left')


            shap.plots._waterfall.waterfall_legacy(expected_value=base_value, shap_values=shap_values[1][0], feature_names=cols_modelo, show=False)
            plt.savefig('static/grafico.png', bbox_inches='tight')
            plt.clf()
            return render_template('result.html', prob_final = probabilidade_final, df_medias = df_means, dic_nomes = dic_nomes)
    return render_template('formulario.html', cols_bin = cols_bin, cols_float = cols_float, dic_nomes = dic_nomes)


if  __name__ == '__main__':
    app.run(debug=True)
