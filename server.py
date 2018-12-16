#from flask import Flask
import numpy as np
import torch.nn.functional as F
import flask
import torch
from form import TipForm
from flask import render_template
#from flask_wtf.csrf import CsrfProtect

app = flask.Flask(__name__)
app.config['WTF_CSRF_SECRET_KEY'] = 'IT 666 to 709 kill pic'
#CsrfProtect(app)
#WTF_CSRF_ENABLED = True

#SECRET_KEY = 'this-is-a-secret-key'
model = None


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
	self.hidden1 = torch.nn.Linear(n_feature, n_hidden)   # hidden layer 1
        #self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)   # hidden layer 2
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        out = F.relu(self.hidden1(x))         # activation function for hidden layer
        #out = F.relu(self.hidden2(out))      # activation function for hidden layer
        output = self.predict(out)             # linear output
        return output


def load_model():
        """Load the pre-trained model, you can use your model just as easily.

    """
        global model
        model = Net(n_feature=6, n_hidden=8, n_output=2)
        model.load_state_dict(torch.load('./trainedModel.pt'))
        model.eval()

@app.route("/tip", methods=['GET','POST'])
def start_page():
    form = TipForm(csrf_enabled=False)
    if form.validate_on_submit():
        return redirect('/predict')
    return render_template('tip.html', title='Tip',form = form)
    
        
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view
    data = {}
    if flask.request.method == 'POST':
            """
            num_of_clients = int(flask.request.args.get('num'))
            distance = float(flask.request.args.get('dis'))
            fare_amunt = 2.5+distance/0.2*0.5+0.8
            tolls = float(flask.request.args.get('tolls'))
            duration = float(flask.request.args.get('dur'))
            is_weekend = bool(flask.request.args.get('is_w'))
            """
            num_of_clients = int(request.form['num_of_clients'])
            distance = float(request.form['distance'])
            tolls = float(request.form['tolls'])
            duration = float(request.form['duration'])
            is_weekend = bool(request.form['is_weekend'])
            fare_amunt = 2.5+distance/0.2*0.5+0.8
            
            #prepare_data
            a = [num_of_clients,distance,fare_amunt,tolls,duration,is_weekend]
            a = torch.from_numpy(np.array(a))
            #a = a.reshape(-1,1)
            a = torch.autograd.Variable(a.float())
            
            #start prediction

            #return value
            result = model(a).data
            data['tips'] =  float(result[1])
            data['fare'] = float(result[0])
            data['total'] = float(result[1])+float(result[0])

    return flask.jsonify(data)
            


if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run(host='0.0.0.0')
