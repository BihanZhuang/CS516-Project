PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'

import sys, getopt
from flask import request

def main(argv):
    #get the parameters
    a = ''
    b = ''
    try:
        opts, args = getopt.getopt(argv,"i:o:",["a=","b="])
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i"):
            a = arg
        elif opt in ("-o"):
            b = arg

    #send the request
    
    #check if we send request succssfully 
               
               
               
if __name__ == "__main__":
    main(sys.argv[1:])
