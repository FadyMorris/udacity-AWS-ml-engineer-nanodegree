# Image Classification using AWS Sagemaker <br/> Dog Identification App using Transfer Learning (Pytorch ResNet50)


In theis project we use AWS Sagemaker to train a pretrained Resnet50 model that can perform image classification on the provided dog breed classication data set. We also use the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. 

Our training scripts can take advantage of Amazon EC2 GPU-based parallel compute capabilities using Pytorch and CUDA. They can utilize Multi-GPU insances using Pytorch `DataParallel()` model.

## Project Set Up and Installation
This project was developed and tested locally on local jupyter server that connects to AWS cloud using AWS access key credentials. It was created from starter files from [this Git repository](https://github.com/udacity/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter).
### Local environment setup:
- Python 3.8.12
- Pytorch with CUDA 1.10.0+cu102 ([link](https://pytorch.org/get-started/locally/))
- Pandas >= 1.2.4
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html).
- Amazon SageMaker Debugger client library, SMDebug
- [Install and configure aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html) to interact 
with AWS. Configuration includes your security credentials, the default output format, and the default AWS Region

Or alternatively you can use AWS SageMaker Studio.

### AWS Execution Role:
The AWS execution role used for the project should have the following access:
- AmazonSageMakerFullAccess
- AmazonS3FullAccess

## Dataset
The provided dataset is the dogbreed classification dataset which can be downloaded from this [link](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project. The dataset file tree should have the following format:

```
data-dir
├── train
│   ├── Label 001
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   ├── Label 002
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   └── ...
├── valid
│   ├── Label 001
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   ├── Label 002
│   │   ├── image_01.jpg
│   │   └── image_02.jpg
│   └── ...
└── test
    ├── Label 001
    │   ├── image_01.jpg
    │   └── image_02.jpg
    ├── Label 002
    │   ├── image_01.jpg
    │   └── image_02.jpg
    └── ...

```

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
If you have set up aws-cli you can execute the following commands:  

```sh
mkdir data
aws s3 cp s3://udacity-aind/dog-project/dogImages.zip ./data/ --no-sign-request
cd data
unzip dogImages.zip
aws s3 sync dogImages/  s3://<default_s3_bucket>/data/ 
```

## Hyperparameter Tuning

We use Pytorch pretreained ResNet50 model for transfer learning. Resnet 50 is a pretrained convolutional neural network. We fine tune Resnet 50 using transfer learning techniques to classify dog breeds from images.


In this project we tune `learning_rate` and `batch_size` parameters. Both of these two parameters can affect the speed of conversion and the accuracy of the model.

`learning_rate` is in the range (0.001, 0.1) and `batch_size` is one from 5 values ([32, 64, 128, 256, 512]).

We run a hyperparameter tuning job that randomly selects parameters from the search space run a training job, then try to guess the next hyperparameters to pick for subsequent training jobs to improve the `Test Loss` metric.

The best training hyperparameters are those that _minimize_ the `Test Loss` metric.

- A screenshot of completed hyperparameter tuning job:

![Hyperparameter Tuning Job](./screenshots/hyperparameter-tuning-job_all-jobs.png?raw=true "Completed Hyperparameter Tuning Job")
- Logged metrics during the training process:

| TrainingJobName                               | batch_size   |   learning_rate |   Test Loss |   TrainingTime |
|:----------------------------------------------|:-------------|----------------:|------------:|---------------:|
| pytorch-dog-hpo-tuni-220117-0020-004-30d11567 | "256"        |      0.0141198  |     2.33193 |           1365 |
| pytorch-dog-hpo-tuni-220117-0020-003-8cc229d5 | "32"         |      0.00288517 |     1.17126 |           1222 |
| pytorch-dog-hpo-tuni-220117-0020-002-79a70b44 | "512"        |      0.00126656 |     2.49791 |           1381 |
| pytorch-dog-hpo-tuni-220117-0020-001-ceff5c55 | "128"        |      0.0425335  |     4.85937 |           1370 |

- Best hyperparameters: `batch_size` = 32 and `learning_rate` = 0.00288517

## Debugging and Profiling

Model debugging and profiling was done using Amazon SageMaker Debugger client library `SMDebug`. Sagemaker debugger is used to monitor machine learning training performance, record training and evaluation metrics and plot learning curves. It can also check for other problems like overfitting, overtraining, poor weight initialization and vanishing gradients.

Profiling is used to give insights about compute instance resource utilization statistics, bottlenecks in algorihtm training, rules detailed analysis and summary statistics.


### Results

The algorithm can be improved by having more training time. From the profiler it was also shown that the compute instance's GPU was underutilized, perhaps becacuse of the small batch size used (32). This problem could have happened because we used `ml.m5.2xlarge` instances for hyperparameter tuning and a different instance (`ml.p2.xlarge`) instance for model training. The hyperparameter tuning job didn't take into account the presense of the GPU.

Profiler resutls are included and can be found in [ProfilerReport/profiler-output/](./ProfilerReport/profiler-output/)


## Model Deployment

The model was deployed to a Sagemaker endpoint for inference. We picked `ml.t2.medium` instance as it has a low cost.
The [inference script](code/inference.py) can take eitehr the image binary data or the image URL as input. It ouptuts the logits  from the final fully connected custom layer of our model as a vector with length 133 corresponding to dog breeds in the dataset.

The deployed endpoint can be queried using one of the following methdos:

- Using Sagemaker SDK:

 + Deployment:
```Python
predictor = sagemaker.predictor.Predictor(endpoint_name, 
                                            sagemaker_session=None, 
                                            serializer=jpeg_serializer, 
                                            deserializer=json_deserializer,
                                           )
```

 + Query using image binary data:  
```Python
response=predictor.predict(image_bytes, initial_args={"ContentType": "image/jpeg"})
```
 + Query using image URL:
```Python
response=predictor.predict(json.dumps(request_dict), initial_args={"ContentType": "application/json"})
```


- Boto3 library:
```Python
client = boto3.client('sagemaker-runtime')
content_type = "image/jpeg"                   #"application/json"      
payload = image_bytes                         #json.dumps(request_dict)  
response_boto3 = client.invoke_endpoint(
                                EndpointName=endpoint_name, 
                                ContentType=content_type,
                                Body=payload,
                                )
response = json.loads(response_boto3['Body'].read())
```

- AWS CLI:

 + Encode image binary in base64 encoding:  
```Python
image_base64 = base64.b64encode(image_bytes).decode("utf-8") 
```

 + In bash shell run the command:  
```bash
 aws sagemaker-runtime invoke-endpoint \
             --endpoint-url https://runtime.sagemaker.us-east-1.amazonaws.com \
             --endpoint-name $endpoint_name \
             --body <image_base64> \
             --content-type "image/jpeg" \
             ./output/predictions2.txt
```

![Deployed Endpoint](./screenshots/endpoint-1.png?raw=true "Deployed and Active Endpoint")

## Standout Suggestions
The project can be further improved by running the hyperparameter tuning jobs on GPU insantces. The model can also be packaged as a docker container.
