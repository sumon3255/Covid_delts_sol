from flask import Flask,render_template,request
import pickle
app = Flask(__name__)


file = open('corona_virus_sol.pkl','rb') 
model = pickle.load(file)
@app.route("/",methods=['GET','POST'])
def index():
    if request.method == 'POST':
                  dict = request.form 
                  fever = int(dict['fever'])
                  age = int(dict['age'])
                  headach = int(dict['headach'])
                  runny_nose = int(dict['runny nose'])
                  persistent_Cough = int(dict['persistent Cough'])
                  sore_throat = int(dict['sore throatr'])
                  diffBreath = int(dict['diffBreath'])
                  loss_of_smell = int(dict['loss of smell'])
                  input_user = [fever,age,headach,runny_nose,persistent_Cough,sore_throat,diffBreath,loss_of_smell]
                  inf_prob = model.predict_proba([input_user])[0][1]
                  inf_pred = model.predict([input_user])
                  return render_template("select_data.html", inf = round(inf_prob*100),pred=inf_pred,dict=dict)
    return  render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
    