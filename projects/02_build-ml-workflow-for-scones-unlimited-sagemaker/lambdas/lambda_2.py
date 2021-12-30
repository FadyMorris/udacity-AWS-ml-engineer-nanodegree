import json
import sagemaker
import base64
import os
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = os.environ['ENDPOINT']## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])## TODO: fill in)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT) ## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)## TODO: fill in

    # We return the data back to the Step Function    
    event["body"]["inferences"] = json.loads(inferences)
    return {
        'statusCode': 200,
        'body': event['body']
    }

