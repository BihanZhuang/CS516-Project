#from flask import Flask
import numpy as np
import torch.nn.functional as F
import flask
import torch
app = flask.Flask(__name__)
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
        model = Net(n_feature=6, n_hidden=8, n_output=1)
        model.load_state_dict(torch.load('./trainedModel.pt'))
        model.eval()
        
@app.route("/predict", methods=["GET"])
def predict():
    # Initialize the data dictionary that will be returned from the view
    data = {}
    if flask.request.method == 'GET':
            num_of_clients = int(flask.request.args.get('num'))
            distance = float(flask.request.args.get('dis'))
            fare_amunt = 2.5+distance/0.2*0.5+0.8
            tolls = float(flask.request.args.get('tolls'))
            duration = float(flask.request.args.get('dur'))
            is_weekend = bool(flask.request.args.get('is_w'))

            #prepare_data
            a = [num_of_clients,distance,fare_amunt,tolls,duration,is_weekend]
            a = torch.from_numpy(np.array(a))
            #a = a.reshape(-1,1)
            a = torch.autograd.Variable(a.float())
            
            #start prediction

            #return value
            result = model(a).data
            data['tips'] =  float(result)
            data['fare'] = fare_amunt
            data['total'] = fare_amunt+float(result)

    return flask.jsonify(data)
            


if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run(host='0.0.0.0')
