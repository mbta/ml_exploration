```
$ python3
>>> from subway_pipeline import SubwayPipeline
>>> p = SubwayPipeline(
...     actuals_path="datasets/pa_datapoints.csv",
...     locations_path="datasets/locations.csv",
...     patterns_path="datasets/patterns.csv",
...     terminals_path="datasets/terminal_datapoints.csv",
...     vehicles_path="datasets/vehicle_datapoints.csv"
... )
>>> result = p.load()
>>> result
array([[-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.89696695],
       [-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.95945863],
       [-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.84628768],
       ...,
       [-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.20118781],
       [-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.15051468],
       [-0.1264922 , -0.12878912, -0.04834095, ...,  0.        ,
         0.        , -0.09983732]])
```
