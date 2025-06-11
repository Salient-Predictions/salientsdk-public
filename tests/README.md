

From the commmand line, the best way to test /examples is with 

```
# Clear cached data - equivalent of setting force=True in the notebook file
find ./examples/ -type d -name '*_example' -exec rm -rf {} +
pytest --nbmake "./examples"
```

```
cd ~/salientsdk
pytest --cov=. --cov-report=term

```

