# ml_exploration
Trying out Machine Learning models for prediction accuracy

# Set Up

Install python3 (which includes pip3).

```
$ brew install python3
```

Install virtualenv, which lets you install deps just for this project.

```
$ pip3 install --user --upgrade virtualenv
```

Create and activate the virtualenv. (Note that if you use an unusual shell like `fish` or `csh`, there are alternate `activate` scripts in `env/bin/` to run. Check the directory to see what's supported.)

```
$ virtualenv env
$ source env/bin/activate
$ pip3 install --upgrade pip
$ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
```

Test that it worked. The following should run with no output and no errors.

```
$ python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
```
