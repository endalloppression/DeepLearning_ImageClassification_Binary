using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Vision;
using DeepLearning_ImageClassification_Binary;

var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
var assetsRelativePath = Path.Combine(projectDirectory, "assets");
MLContext mlContext = new MLContext();


 IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    //get all file paths from the subdirectories
    var files = Directory.GetFiles(folder, "*", searchOption:
    SearchOption.AllDirectories);
    //iterate through each file
    foreach (var file in files)
    {
        //Image Classification API supports .jpg and .png formats; check img formats
        if ((Path.GetExtension(file) != ".jpg") &&
         (Path.GetExtension(file) != ".png"))
            continue;
        //store filename in a variable, say ‘label’
        var label = Path.GetFileName(file);
        /* If the useFolderNameAsLabel parameter is set to true, then name 
           of parent directory of the image file is used as the label. Else label is expected to be the file name or a a prefix of the file name. */
        if (useFolderNameAsLabel)
            label = Directory.GetParent(file).Name;
        else
        {
            for (int index = 0; index < label.Length; index++)
            {
                if (!char.IsLetter(label[index]))
                {
                    label = label.Substring(0, index);
                    break;
                }
            }
        }
        //create a new instance of ImgData()
        yield return new ImageData()
        {
            ImagePath = file,
            Label = label
        };
    }

}
IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: assetsRelativePath, useFolderNameAsLabel: true);
IDataView imageData = mlContext.Data.LoadFromEnumerable(images);

IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);


var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(
        inputColumnName: "Label",
        outputColumnName: "LabelAsKey")
    .Append(mlContext.Transforms.LoadRawImageBytes(
        outputColumnName: "Image",
        imageFolder: assetsRelativePath,
        inputColumnName: "ImagePath"));

/*
 * Use the Fit method to apply the data to the preprocessingPipeline EstimatorChain followed by the Transform method, which returns an IDataView containing the pre-processed data.
 */
IDataView preProcessedData = preprocessingPipeline
                    .Fit(shuffledData)
                    .Transform(shuffledData);

/*To train a model, it's important to have a training dataset as well as a validation dataset. The model is trained on the training set. How well it makes predictions on unseen data is measured by the performance against the validation set. Based on the results of that performance, the model makes adjustments to what it has learned in an effort to improve. The validation set can come from either splitting your original dataset or from another source that has already been set aside for this purpose. In this case, the pre-processed dataset is split into training, validation and test sets.
 */
TrainTestData trainSplit = mlContext.Data.TrainTestSplit(data: preProcessedData, testFraction: 0.3);
TrainTestData validationTestSplit = mlContext.Data.TrainTestSplit(trainSplit.TestSet);

/*Assign the partitions their respective values for the train, validation and test data.
 */
IDataView trainSet = trainSplit.TrainSet;
IDataView validationSet = validationTestSplit.TrainSet;
IDataView testSet = validationTestSplit.TestSet;

/*
 * Create a new variable to store a set of required and optional parameters for an ImageClassificationTrainer.
 */
var classifierOptions = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = "Image",
    LabelColumnName = "LabelAsKey",
    ValidationSet = validationSet,
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
    MetricsCallback = (metrics) => Console.WriteLine(metrics),
    TestOnTrainSet = false,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true
};

/*Define the EstimatorChain training pipeline that consists of both the mapLabelEstimator and the ImageClassificationTrainer.
 */
var trainingPipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(classifierOptions)
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

/*Use the Fit method to train your model.
 */
ITransformer trainedModel = trainingPipeline.Fit(trainSet);

/*Create a new utility method called OutputPrediction to display prediction information in the console.
 */
 static void OutputPrediction(ModelOutput prediction)
{
    string imageName = Path.GetFileName(prediction.ImagePath);
    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}

void ClassifySingleImage(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    PredictionEngine<ModelInput, ModelOutput> predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);
    ModelInput image = mlContext.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).First();
    ModelOutput prediction = predictionEngine.Predict(image);
    Console.WriteLine("Classifying single image");
    OutputPrediction(prediction);
    ClassifySingleImage(mlContext, testSet, trainedModel);
}

void ClassifyImages(MLContext mlContext, IDataView data, ITransformer trainedModel)
{
    IDataView predictionData = trainedModel.Transform(data);
    IEnumerable<ModelOutput> predictions = mlContext.Data.CreateEnumerable<ModelOutput>(predictionData, reuseRowObject: true).Take(10);
    Console.WriteLine("Classifying multiple images");
    foreach (var prediction in predictions)
    {
        OutputPrediction(prediction);
    }
    ClassifyImages(mlContext, testSet, trainedModel);
}
