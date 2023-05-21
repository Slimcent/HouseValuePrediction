using HouseValuePrediction.Model;
using Microsoft.ML;

Console.WriteLine("Hello, World!");


MLContext context = new MLContext();

IDataView data = context.Data.LoadFromTextFile<HousingData>("./housing.csv", hasHeader : true, separatorChar : ',' );

DataOperationsCatalog.TrainTestData split = context.Data.TrainTestSplit(data, 0.2);

var features = split.TrainSet.Schema
    .Select(col => col.Name)
    .Where(colName => colName != "Label" && colName != "OceanProximity")
    .ToArray();

var pipline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
    .Append(context.Transforms.Concatenate("Features", features))
    .Append(context.Transforms.Concatenate("Feature", "Features", "Text"))
    .Append(context.Regression.Trainers.LbfgsPoissonRegression());

var model = pipline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);

var metrics = context.Regression.Evaluate(predictions);

Console.WriteLine($"R^2 - {metrics.RSquared}");